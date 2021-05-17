from manimlib.imports import *

COLOR_MAP = {
    "S" : "#007f5f",
    "I" : "#ff0000",
    "R" : "#242423",
    "V" : "#0000FF",
    "A" : "#deff0a"
}

def update_time(m, dt):
    m.time += dt

class City(VGroup):
    CONFIG = {
        "size" : 7,
        "color" : GREY
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.time = 0.0
        self.add_body()
        self.people = VGroup()

        self.add_updater(update_time)
    
    def add_body(self):
        city = Square()
        city.set_height(self.size)
        city.set_stroke(self.color, 2.0, opacity=0.9)
        self.body = city
        self.add(self.body)

DEFAULT_CITY = City(size=10)

class Person(VGroup):
    CONFIG = { 
        "size" : 0.2,
        "max_speed" : 1,
        "wall_buff" : 0.4,
        "random_walk_interval" : 1.0,
        "step_size" : 1.5,
        "gravity_strength" : 1.0,
        "infection_ring_style" : {
            "stroke_color" : COLOR_MAP["I"],
            "stroke_opacity" : 1,
            "stroke_width" : 1.0
        },
        "infection_ring_anim_time" : 0.6,
        "vaccination_blink_time" : 0.6,
        "infection_radius" : 0.3,
        "infection_prob" : 0.2,
        "infection_time" : 8,
        "social_distance_factor" : 0.0,
        "obey_social_distancing" : True
    }

    def __init__(self, city=DEFAULT_CITY, **kwargs):
        super().__init__(**kwargs)

        self.status = "S"
        self.time = 0.0
        self.last_step_update = -1.0
        self.infected_time = -np.inf
        self.recovered_time = -np.inf
        self.vaccinated_time = -np.inf
        self.gravity_center = None
        self.isUpdating = True
        self.isTravelling = False
        self.underQuarantine = False
        self.isVaccinated = False
        self.symptoms = True
        self.destination = None
        self.velocity = np.zeros(3)
        self.city = city

        self.add_body()
        self.add_infection_ring()

        #updaters 
        self.add_updater(update_time)
        self.add_updater(lambda m, dt : m.update_position(dt))
        self.add_updater(lambda m, dt : m.update_infection_ring(dt))
        self.add_updater(lambda m, dt : m.update_color(dt))
        self.add_updater(lambda m, dt : m.update_status(dt))
        self.add_updater(lambda m, dt : m.travel(dt))

    def add_infection_ring(self):
        ring = Circle(radius=self.size/2.5)
        ring.set_style(**self.infection_ring_style)
        ring.move_to(self.get_center())
        self.add_to_back(ring)
        self.infection_ring = ring

    def add_body(self):
        body = self.get_body()
        body.set_height(self.size)
        body.set_color(COLOR_MAP[self.status])
        body.move_to(self.get_center())
        self.body = body
        self.add(self.body)

    def get_body(self):
        return Dot()

    def set_status(self, status):
        self.status = status
        if status == "I":
            self.infected_time = self.time
        elif status == "R":
            self.recovered_time = self.time
        elif status == "V":
            self.vaccinated_time = self.time

    def pause_updation(self):
        self.isUpdating = False
    
    def resume_updation(self):
        self.isUpdating = True

    def change_city(self, city):
        self.city = city

    def update_position(self, dt):
        total_force = np.zeros(3)
        if self.isUpdating:
            c = self.get_center()
            

            #updating gravity center. This is a nicer version of random_walk
            if (self.time - self.last_step_update) >= self.random_walk_interval:
                self.last_step_update = self.time
                random_vec = rotate_vector(RIGHT, TAU * random.random())
                self.gravity_center = c + random_vec * self.step_size
            
            #gravity
            if self.gravity_center is not None:
                f = self.gravity_center - c
                r = np.linalg.norm(f)
                if r > 0.0:
                    total_force += f/r**2 * self.gravity_strength


            #walls
            dl = self.city.get_corner(DL)
            ur = self.city.get_corner(UR)
            wall_force = np.zeros(3)

            for i in range(2):
                to_dl = c[i] - dl[i] - self.size/2.0
                to_ur = ur[i] - c[i] - self.size/2.0

                if to_dl < 0.0:
                    self.velocity[i] *= -1
                    self.set_coord(dl[i] + self.size/2.0, i)
                
                if to_ur < 0.0:
                    self.velocity[i] *= -1
                    self.set_coord(ur[i] - self.size/2.0, i)
                
                #dl force should be +ve, the other -ve
                wall_force[i] += max(0, (1.0/to_dl - 1.0/self.wall_buff))
                wall_force[i] -= max(0, (1.0/to_ur - 1.0/self.wall_buff))
            
            total_force += wall_force

            #social distancing
            if self.social_distance_factor > 0.0 and self.obey_social_distancing and not self.underQuarantine:
                people = self.city.people
                repulsion_force = np.zeros(3)
                for other in people:
                    if other != self:
                        vec = self.get_center() - other.get_center()
                        d = np.linalg.norm(vec)
                        if d > 0.0:
                            repulsion_force += vec/d**3 * self.social_distance_factor
                total_force += repulsion_force

        #update velocity
        self.velocity += total_force * dt

        #limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.max_speed * self.velocity / speed

        #update postion
        self.shift(self.velocity * dt)

    def update_infection_ring(self, dt):
        if self.status == "I":
            if self.symptoms:
                col = COLOR_MAP["I"]
            else:
                col = COLOR_MAP["A"]
            if self.time - self.infected_time <= self.infection_ring_anim_time:
                alpha = (self.time - self.infected_time)/self.infection_ring_anim_time
                if 0.0 <= alpha <= 1.0:
                    self.infection_ring.set_width(alpha * self.size * 8 + (1.0-alpha)*self.size)
                    self.infection_ring.set_style(stroke_color=col, stroke_opacity=there_and_back(alpha), stroke_width=there_and_back(alpha)*5)
            else:
                self.infection_ring.set_style(stroke_opacity=0.0, stroke_width=0.0)
        
    def update_color(self, dt):
        if self.status == "I":
            if self.symptoms:
                col = COLOR_MAP["I"]
            else:
                col = COLOR_MAP["A"]
            if self.time - self.infected_time <= self.infection_ring_anim_time:
                alpha = (self.time - self.infected_time)/self.infection_ring_anim_time
                if 0.0 <= alpha < 1.0:
                    self.body.set_color(interpolate_color(COLOR_MAP["S"], col, alpha))
                else:
                    self.body.set_color(col)
                

        if self.status == "R":
            if self.time - self.recovered_time <= self.infection_ring_anim_time:
                alpha = (self.time - self.recovered_time)/self.infection_ring_anim_time
                if 0.0 <= alpha < 1.0:
                    self.body.set_color(interpolate_color(COLOR_MAP["I"], COLOR_MAP["R"], alpha))
                else:
                    self.body.set_color(COLOR_MAP["R"])
                    self.body.set_opacity(0.4)

        if self.status == "V":
            if self.time - self.vaccinated_time <= self.vaccination_blink_time:
                alpha = (self.time - self.vaccinated_time)/self.vaccination_blink_time
                if 0.0 <= alpha <= 1.0:
                    self.body.set_color(interpolate_color(COLOR_MAP["S"], COLOR_MAP["V"], alpha))
            else:
                self.vaccinated_time = self.time


    def update_status(self, dt):
        people = self.city.people
        infected_people = list(filter(lambda m : m.status == "I", people))

        if self.status == "S":
            for other in infected_people: #use a Quadtree maybe??
                if other != self and not other.isTravelling:
                        d = np.linalg.norm(self.get_center() - other.get_center())
                        if d < self.infection_radius and random.random() < self.infection_prob * dt:
                            self.set_status("I")
        elif self.status == "I":
            if (self.time - self.infected_time) > self.infection_time:
                    self.set_status("R")

    def start_journey(self, city):
        if not self.isTravelling:
            self.city.remove(self)
            city.people.add(self)
            self.isTravelling = True
            self.pause_updation()
            self.destination = city

    def travel(self, dt):
        if self.isTravelling:
            vec = self.get_center() - self.destination.get_center()
            d = np.linalg.norm(vec)
            if d <= self.destination.get_width()/2.0:
                self.resume_updation()
                self.isTravelling = False
                self.city = self.destination

            elif d > 0.0:
                self.shift(-vec/d * dt * 7)
                # self.velocity += -vec/d * dt * 10

    def go_quarantine(self, Qzone):
        if not self.isTravelling:
            self.city.remove(self)
            Qzone.people.add(self)
            self.isTravelling = True
            self.pause_updation()
            self.destination = Qzone


class SIRSimulation(VGroup):
    CONFIG = {
        "n_cities" : 1,
        "city_size" : 7,
        "n_citizen_per_city" : 50,
        "social_distance_obedience" : 0.0, #anything between 0 and 1
        "person_config" : {
            "infection_prob" : 0.5,
            "social_distance_factor" : 0.2,
        },
        "travel_rate" : 0.02,
        "quarantine" : True,
        "time_for_infections_to_start" : 4,
        "prob_symptoms" : 0.8,
        "vaccine_efficacy" : 0.0,
        "vaccination_frequency" : 2.0,
        "vaccine_per_day" : 10,
        "include_vaccination" : True
    
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.time = 0.0
        self.last_travel_time = -np.inf
        self.last_vaccinated_time = -np.inf
        self.add_cities()
        self.add_people()
        self.infect_one_person()

        self.add_updater(update_time)
        self.add_updater(lambda m, dt : m.travel(dt))
        self.add_updater(lambda m, dt : m.put_under_quarantine(dt))
        self.add_updater(lambda m, dt : m.vaccinate(dt))

    def add_cities(self):
        self.cities = VGroup()
        for _ in range(self.n_cities):
            city = City(size=self.city_size)
            self.cities.add(city)

        self.cities.arrange_in_grid(buff=LARGE_BUFF)
        self.add(self.cities)

        if self.quarantine:
            Qzone = City(size=self.cities.get_height()*0.2, color=RED)
            Qzone.next_to(self.cities.get_corner(DL) + UP * Qzone.get_height()/2.0, LEFT, buff=0.1*self.cities.get_width())
            self.Qzone = Qzone
            self.add(Qzone)

    def add_people(self):
        self.people = VGroup()
        for city in self.cities:
            for _ in range(self.n_citizen_per_city):
                if random.random() < self.social_distance_obedience:
                    obey_social_distancing = True
                else:
                    obey_social_distancing = False
                p = Person(city=city, **self.person_config, obey_social_distancing=obey_social_distancing)
                if random.random() > self.prob_symptoms:
                    p.symptoms = False

                dl = city.get_corner(DL)
                ur = city.get_corner(UR)
                x = random.uniform(dl[0], ur[0])
                y = random.uniform(dl[1], ur[1])
                p.move_to(np.array([x, y, 0.0]))
                self.people.add(p)
                city.people.add(p)

        self.add(self.people)
    
    def infect_one_person(self):
        # for _ in range(10):
            p = random.choice(self.people)
            p.set_status("I")

    def travel(self, dt):
        if self.travel_rate > 0.0:
            for p in self.people:
                if random.random() < self.travel_rate * dt and not p.underQuarantine:
                    new_city = random.choice(self.cities)
                    if new_city != p.city:
                        p.start_journey(new_city)

    def put_under_quarantine(self, dt):
        if self.quarantine:
            i_people = list(filter(lambda m : m.status == "I", self.people))
            for p in i_people:
                if p.time - p.infected_time > self.time_for_infections_to_start and not p.underQuarantine and p.symptoms:
                    p.isTravelling = False
                    p.underQuarantine = True
                    p.go_quarantine(self.Qzone)

    def vaccinate(self, dt):
        if self.include_vaccination:
            if self.time - self.last_vaccinated_time > self.vaccination_frequency:
                self.last_vaccinated_time = self.time
                s_people = [p for p in self.people if p.status=="S" and not p.isVaccinated]
                candidates_for_vaccine = []
                if len(s_people) > 0:
                    for _ in range(self.vaccine_per_day):
                        p = random.choice(s_people)
                        if p not in candidates_for_vaccine:
                            candidates_for_vaccine.append(p)
                for p in candidates_for_vaccine:
                    if random.random() < self.vaccine_efficacy:
                        p.set_status("V")
                        p.isVaccinated = True
                    else:
                        p.isVaccinated = True


    def get_counts(self):
        return np.array(
            [
                len(list(filter(lambda m : m.status == status, self.people))) for status in "SIRV"
            ]
        )

    def get_normalised_data(self):
        counts = self.get_counts()
        return counts/sum(counts)

    def set_social_distancing(self, factor, prob):
        for p in self.people:
            if random.random() < prob:
                p.social_distance_factor = factor
                p.obey_social_distancing = True

    def set_travel_rate(self, rate):
        self.travel_rate = rate


#most of this is similar to that of 3B1B          
class SIRGraph(VGroup):
    CONFIG = {
        "width" : 7,
        "height" : 5,
        "update_frequency" : 1/30.0,
        "include_r_graph" : False,
    }

    def __init__(self, simulation, **kwargs):
        super().__init__(**kwargs)
        self.time = 0.0
        self.last_update_time = -np.inf
        self.simulation = simulation
        self.data = [self.simulation.get_normalised_data()]

        self.add_axes()
        self.add_x_labels()
        self.add_y_labels()
        self.add_graph()

        self.add_updater(update_time)
        self.add_updater(lambda m, dt : m.update_graph(dt))
        self.add_updater(lambda m, dt : m.update_labels(dt))

    def add_axes(self):
        axes = Axes(
            y_min=0.0,
            y_max = 1.0,
            y_axis_config = {
                "tick_frequency" : 0.1
            },
            x_min = 0,
            x_max = 1,
            axis_config = {
                "include_tip" : False
            }
        )

        origin = axes.coords_to_point(0, 0)
        axes.x_axis.set_width(self.width, about_point=origin, stretch=True)
        axes.y_axis.set_height(self.height, about_point=origin, stretch=True)
        self.axes = axes
        self.add(self.axes)

    def add_graph(self):
        self.graph = self.get_graph()
        self.add(self.graph)

    def add_x_labels(self):
        self.x_ticks = VGroup()
        self.x_labels = VGroup()
        self.add(self.x_ticks, self.x_labels)

    def add_y_labels(self):
        xs = np.arange(0.2, 1.1, 0.2)
        y_labels = VGroup()
        for x in xs:
            label = DecimalNumber(x, num_decimal_places=1)
            label.set_height(self.height * 0.05)
            label.move_to(self.axes.coords_to_point(0, x) + LEFT * label.get_width())
            y_labels.add(label)
        self.y_labels = y_labels
        self.add(self.y_labels)

    def get_graph(self):
        axes = self.axes
        data = self.data
        i_points = []
        r_points = []

        for x, counts in zip(np.linspace(0, 1, len(data)), data):
            i_points.append(axes.coords_to_point(x, counts[1]))
            r_points.append(axes.coords_to_point(x, counts[2]))

        i_lines = VGroup()
        for i in range(len(i_points)-1):
            i_lines.add(Line(i_points[i], i_points[i+1], color=COLOR_MAP["I"], stroke_width=2.0))

        if self.include_r_graph:
            r_lines = VGroup()
            for i in range(len(r_points)-1):
                r_lines.add(Line(r_points[i], r_points[i+1], color=COLOR_MAP["R"], stroke_width=1.0))
            return VGroup(i_lines, r_lines)
        else:
            return i_lines

    def update_graph(self, dt):
        if self.time - self.last_update_time > self.update_frequency:
            self.data.append(self.simulation.get_normalised_data())
            self.graph.become(self.get_graph())
            self.last_update_time = self.time

    def update_labels(self, dt):
        tick_height = 0.05 * self.height
        tick_template = Line(UP, DOWN).set_height(tick_height)

        if self.time < 10:
            tick_range = range(1, int(self.time) + 1, 1)
        elif self.time < 50:
            tick_range = range(5, int(self.time) + 1, 5)
        elif self.time < 100:
            tick_range = range(10, int(self.time) + 1, 10)
        else:
            tick_range = range(20, int(self.time) + 1, 20)

        def get_tick(x):
            tick = tick_template.copy()
            tick.move_to(self.axes.coords_to_point(x/self.time, 0))
            return tick
        
        def get_tick_label(x, tick):
            label = Integer(x)
            label.set_height(tick_height)
            label.next_to(tick, DOWN, buff=0.2*tick_height)
            return label
        x_ticks = VGroup()
        x_labels = VGroup()
        for x in tick_range:
            tick = get_tick(x)
            label = get_tick_label(x, tick)
            x_ticks.add(tick)
            x_labels.add(label)

        self.x_ticks.become(x_ticks)
        self.x_labels.become(x_labels)

    def add_h_line(self, h):
        line = Line(
            self.axes.c2p(0, h),
            self.axes.c2p(1, h),
            color=YELLOW,
            stroke_width=2.0
        )
        self.add(line)


class GeneralSimulation(ZoomedScene):
    CONFIG = {
        "simulation_config" : {
            "n_cities" : 1,
            "city_size" : 7,
            "n_citizen_per_city" : 150,
            "social_distance_obedience" : 0.0, #anything between 0 and 1
            "person_config" : {
                "max_speed" : 0.5,
                "gravity_strength" : 0.2,
                "infection_radius" : 0.5,
                "infection_prob" : 0.2,
                "infection_time" : 8,
                "social_distance_factor" : 2.0,
        
            },
            #travel
            "travel_rate" : 0.02,
            #quarantine
            "quarantine" : False,
            #infection stuff
            "time_for_infections_to_start" : 4,
            "prob_symptoms" : 1.0,
            #vaccine stuff
            "include_vaccination" : False,
            "vaccine_efficacy" : 1.0,
            "vaccination_frequency" : 2.0,
            "vaccine_per_day" : 10,
            
        },

        "graph_config" : {
            "update_frequency" : 1/30.0,
            "include_r_graph" : False
        },
        "width_to_frameheight_ratio" : 0.5,
        "height_to_frameheight_ratio" : 0.4,
        
    }

    def setup(self):
        super().setup()
        self.add_simulation()
        self.position_camera()
        self.add_graph()
        self.add_n_cases()
        self.n_cases[1].add_updater(lambda m, dt : m.set_value(self.simulation.get_counts()[1]))

    def add_simulation(self):
        self.simulation = SIRSimulation(**self.simulation_config)
        self.add(self.simulation)

    def add_n_cases(self):
        text = VGroup(TextMobject("Active Cases ="))
        text.add(Integer(1).next_to(text[0], RIGHT))
        text.set_width(self.graph.get_width() * 0.6)
        text.next_to(self.graph, DOWN, buff=self.graph.get_height() * 0.15)
        text.set_color(COLOR_MAP["I"])
        self.n_cases = text
        self.add(self.n_cases)

    def add_graph(self):
        frame = self.camera_frame
        h = frame.get_height()
        graph = SIRGraph(
            self.simulation, 
            width=h*self.width_to_frameheight_ratio, 
            height=h*self.height_to_frameheight_ratio,
            **self.graph_config
        )
        graph.next_to(frame.get_left(), RIGHT, buff=0.2*graph.get_width())
        graph.shift(UP * h/5.0)
        self.graph = graph
        self.add(self.graph)
    

    def position_camera(self):
        cities = self.simulation.cities
        frame = self.camera_frame
        height = cities.get_height() + 1
        width = cities.get_width() * 2

        if frame.get_width() < width:
            frame.set_width(width)
        if frame.get_height() < height:
            frame.set_height(height)
               
        frame.next_to(cities.get_right(), LEFT, buff=-0.1*cities.get_width())

    def construct(self):
        self.wait_until(self.count, max_time=180)
        self.wait(5)
    
    def count(self):
        c = 0
        for p in self.simulation.people:
            if p.status == "I":
                c += 1
        return (c == 0)

class IntroSim(GeneralSimulation):
    CONFIG = {
        "graph_config" : {
            "update_frequency" : 1/60.0
        }
    }
            
class ControlCity(GeneralSimulation):
    CONFIG = {
        "random_seed" : 54,
        "simulation_config" : {
            "n_citizen_per_city" : 300,
            "person_config" : {
                "size" : 0.1,
                "max_speed" : 0.5,
                "gravity_strength" : 0.3,
                "infection_radius" : 0.35,
                "infection_time" : 8,
            }
        },
        "graph_config" : {
            "update_frequency" : 1/60.0
        }
    }

class HygienicCity(ControlCity):
    CONFIG = {
        "random_seed" : 47, 
        "simulation_config" : {
            "person_config" : {
                "infection_prob" : 0.1
            }
        }
    }

class CityWithSocialDistancing(GeneralSimulation):
    CONFIG = {
        "random_seed" : 42,
        "social_distancing_starts_at" : 15,
        "simulation_config" : {
            "n_citizen_per_city" : 300,
            "person_config" : {
                "size" : 0.1,
                "max_speed" : 0.5,
                "gravity_strength" : 0.3,
                "infection_radius" : 0.35,
                "infection_time" : 8,
            }
        },
        "graph_config" : {
            "update_frequency" : 1/60.0
        }
    }

    def construct(self):
        def till_threshold():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c >= self.social_distancing_starts_at)
        self.wait_until(till_threshold)
        self.simulation.set_social_distancing(0.9, 1.0)
        self.wait_until(self.count, max_time=180)
        self.wait(5)


class CityWithSocialDistancing2(CityWithSocialDistancing):
    CONFIG = {
        "social_distancing_starts_at" : 30
    }

    def construct(self):
        def till_threshold():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c >= self.social_distancing_starts_at)
        self.wait_until(till_threshold)
        self.simulation.set_social_distancing(0.25, 1.0)
        self.wait_until(self.count, max_time=180)
        self.wait(5)

class CityWithQuarantine(ControlCity):
    CONFIG = {
        "random_seed" : 51,
        "simulation_config" : {
                    "quarantine" : True,
                    "time_for_infections_to_start" : 4
        }    
    }

class CityWithQuarantine2(ControlCity):
    CONFIG = {
        "random_seed" : 63,
        "simulation_config" : {
                    "quarantine" : True,
                    "time_for_infections_to_start" : 2
        }    
    }

class CityWithVaccination100(ControlCity):
    CONFIG = {
        "random_seed" : 2,
        "vaccination_starts_at" : 15,
        "simulation_config" : {
        "vaccine_efficacy" : 1.0,
        "vaccination_frequency" : 1.0,
        "vaccine_per_day" : 15,
        }
    }

    def construct(self):
        def till_threshold():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c >= self.vaccination_starts_at)
        self.wait_until(till_threshold)
        self.simulation.include_vaccination = True
        self.wait_until(self.count, max_time=180)
        self.wait(5)

class CityWithVaccination96(CityWithVaccination100):
    CONFIG = {
        "random_seed" : 103,
        "vaccine_efficacy" : 0.96,
        }

class CityWithVaccination60(CityWithVaccination100):
    CONFIG = {
        "random_seed" : 1,
        "vaccine_efficacy" : 0.6,
        }

class CityWithSocialDistancing60p(CityWithSocialDistancing):
    CONFIG = {
        "social_distancing_starts_at" : 30,
    }
    def construct(self):
        def till_threshold():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c >= self.social_distancing_starts_at)
        self.wait_until(till_threshold)
        self.simulation.set_social_distancing(0.25, 0.6)
        self.wait_until(self.count, max_time=180)
        self.wait(5)

class SecondWave(GeneralSimulation):

    CONFIG = {
        "simulation_config" : {
            "n_cities" : 9,
            "city_size" : 7,
            "n_citizen_per_city" : 100,
            "person_config" : {
                "max_speed" : 0.5,
                "gravity_strength" : 0.2,
                "infection_radius" : 0.5,
                "infection_prob" : 0.2,
                "infection_time" : 8,
                "social_distance_factor" : 2.0,
        
            },
            #travel
            "travel_rate" : 0.02,   
        },

        "graph_config" : {
            "update_frequency" : 1/60.0,
            "include_r_graph" : False
        },
        "restrictions_start_at" : 90, #cases
        "restrictions_relaxed_after" : 20 #cases
    }

    def construct(self):
        def till_threshold():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c >= self.restrictions_start_at)
        self.wait_until(till_threshold)
        self.simulation.set_social_distancing(0.2, 1.0)
        self.simulation.set_travel_rate(0.0)
        def till_threshold_lower():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c <= self.restrictions_relaxed_after)
        self.wait_until(till_threshold_lower)
        self.simulation.set_travel_rate(0.02)
        self.simulation.set_social_distancing(0.0, 1.0)
        for p in self.simulation.people:
            p.obey_social_distancing = False
        self.wait_until(self.count, max_time=180)
        self.wait(5)

class SecondWavePrevention(SecondWave):
    CONFIG = {
        "simulation_config" : {
        "vaccine_per_day" : 45,
        "vaccine_efficacy" : 1.0,
        "vaccination_frequency" : 1
        },
        "vaccine_starts_after" : 3,
    }

    def construct(self):
        def till_threshold():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c >= self.restrictions_start_at)
        self.wait_until(till_threshold)

        self.simulation.set_social_distancing(0.25, 1.0)
        self.simulation.set_travel_rate(0.0)

        self.wait(self.vaccine_starts_after)
        self.simulation.include_vaccination = True

        def till_threshold_lower():
            c = 0
            for p in self.simulation.people:
                if p.status == "I":
                    c += 1
            return (c <= self.restrictions_relaxed_after)
        self.wait_until(till_threshold_lower)

        self.simulation.set_travel_rate(0.02)
        self.simulation.set_social_distancing(0.0, 1.0)
        for p in self.simulation.people:
            p.obey_social_distancing = False

        self.wait_until(self.count, max_time=180)
        self.wait(5)

class ExpvLinear(GraphScene):
    def construct(self):
        self.setup_axes(animate=True)
        exp = self.get_graph(lambda x : np.exp(x), color=COLOR_MAP["I"])
        exp_text = TextMobject("Infections").set_color(COLOR_MAP["I"]).scale(0.7).next_to(self.input_to_graph_point(2.1, exp), LEFT)
        linear = self.get_graph(lambda x : 2*x, color=COLOR_MAP["V"])
        linear_text = TextMobject("Vaccinations").set_color(COLOR_MAP["V"]).scale(0.7).next_to(self.input_to_graph_point(2, linear), RIGHT)
        self.play(
            ShowCreation(exp), Write(exp_text)
        )
        self.play(
            ShowCreation(linear),  Write(linear_text)
        )
        self.wait()

class IntroSIR(Scene):
    CONFIG = {
        "random_seed" : 2
    }
    def construct(self):
        self.wait(2)
        p = Person(city= City(size=6), infection_time=90, max_speed=1.5).scale(1.5)
        susceptible = TextMobject("Susceptible").set_color(COLOR_MAP["S"])
        susceptible.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        S = TextMobject("S").set_color(COLOR_MAP["S"])
        S.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        self.play(
            GrowFromCenter(p)
        )
        self.wait(9)
        self.play(
            Write(susceptible)
        )
        self.wait(1)
        self.play(
            ReplacementTransform(susceptible, S)
        )
        self.wait(6)
        I = TextMobject("I").set_color(COLOR_MAP["I"])
        I.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        infected = TextMobject("Infected").set_color(COLOR_MAP["I"])
        infected.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        p.set_status("I")
        self.play(
            ReplacementTransform(S, infected)
        )
        self.wait(6)
        
        self.play(
            ReplacementTransform(infected, I)
        )
        self.wait(10)
        R = TextMobject("R").set_color(COLOR_MAP["R"])
        R.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        removed = TextMobject("Removed").set_color(COLOR_MAP["R"])
        removed.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        p.set_status("R")
        self.play(
            ReplacementTransform(I, removed)
        )
        self.wait()
        self.play(
            ReplacementTransform(removed, R)
        )
        self.wait(5)
        self.play(
            FadeOut(p),
            FadeOut(R)
        ) 
        self.wait(3.5)
        prob_inf = TexMobject("\\text{P(}", "\\text{infection}", ")", "=", "20 \\%").to_edge(LEFT)
        prob_inf[1].set_color(COLOR_MAP["I"])
        self.play(
            FadeInFromDown(prob_inf)
        )
        self.wait()

class IntroVaccine(Scene):
    CONFIG = {
        "random_seed" : 1
    }
    def construct(self):
        p = Person(city= City(size=7)).scale(1.5)
        S = TextMobject("S").set_color(COLOR_MAP["S"])
        # S.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        self.play(
            GrowFromCenter(p),
        )
        self.wait(6)
        
        # self.wait(2)
        # V = TextMobject("V").set_color(COLOR_MAP["V"])
        # V.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        p.set_status("V")
        # self.play(
        #     ReplacementTransform(S, V)
        # )
        self.wait(3)
        # Vaccinated = TextMobject("Vaccinated").set_color(COLOR_MAP["V"])
        # Vaccinated.add_updater(lambda m : m.next_to(p.get_center(), UP, buff=MED_LARGE_BUFF))
        # self.play(
            # ReplacementTransform(V, Vaccinated)
        # )
        # self.wait(2)
