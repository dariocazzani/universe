import random
from sklearn.metrics import pairwise_distances
import pygame
from math import degrees, atan2, cos, sin, radians
import numpy as np 

pygame.init()
clock = pygame.time.Clock()
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class Odin(object):
    G = 6.674 * 1e-1
    BigBang = False
    max_speed = 3
    max_mass = 4
    max_density = 1
    initial_max_speed = 1e-1
    num_particles = 30
    universe_size = 1000


class Particle(object):
    def __init__(self, 
            speed:float=0.0, 
            direction:float=0.0,
            position:tuple=(0,0),
            mass:float=1.0,
            density:float=1.0):
        self.speed = speed 
        self.direction = direction # -180 to 180.0 --> 2D universe
        self.position = position # --> 2D universe
        self.mass = mass
        self.density = density
        self.size = self.mass / self.density
        self.momentum = self.get_momentum()
        self.color = color = WHITE
        # self.color = (int(self.get_mass() / Odin.max_mass * 255), color[1], color[2])

    def get_x(self) -> int:
        return self.position[0]

    def get_y(self) -> int:
        return self.position[1]

    def set_x(self, x:int):
        self.position = (x, self.position[1])

    def set_y(self, y:int):
        self.position = (self.position[0], y)

    def get_mass(self) -> float:
        return self.mass

    def set_mass(self, mass:float):
        self.mass = mass

    def get_density(self) -> float:
        return self.density

    def set_density(self, density:float):
        self.density = density

    def get_size(self) -> float:
        return self.size

    def set_size(self, size:float):
        self.size = size

    def get_position(self) -> tuple:
        return self.position

    def set_position(self, position:tuple):
        self.position = position

    def get_direction(self) -> float:
        return self.direction

    def get_speed(self) -> float:
        return self.speed

    def set_speed(self, speed:float):
        self.speed = speed

    def get_momentum(self) -> tuple:
        return (cos(radians(self.get_direction()))*self.get_speed(), sin(radians(self.get_direction()))*self.get_speed())

    def set_momentum(self, momentum:tuple):
        self.momentum = momentum
        # update speed and direction
        self.direction = degrees(atan2(momentum[1], momentum[0]))
        self.speed = min(Odin.max_speed, np.sqrt(momentum[0]**2 + momentum[1]**2))
    
    def move(self):
        x_offset, y_offset = self.get_momentum()
        self.position = (self.get_x() + x_offset, self.get_y() + y_offset)

    def __str__(self) -> str:
        return str(self.position)

    def __repr__(self) -> str:
        return self.__str__()


class FlatUniverse(object):
    def __init__(self,
            num_particles:int=100,
            size:int=1000):
        self.num_particles = num_particles
        self.size = size
        self._init_locations()

    def __str__(self) -> str:
        return str(self.particles)
        
    def _init_locations(self):
        self.particles = list()
        pos = random.sample(range(self.size ** 2), self.num_particles)
        for p in pos:
            if Odin.BigBang:
                x = int(self.size/2)
                y = int(self.size/2)
            else:
                x = p % self.size
                y = p / self.size
            speed = np.random.rand() * Odin.initial_max_speed
            mass = max(1, random.random()*Odin.max_mass)
            density = max(1, random.random()*Odin.max_density)
            direction = np.random.rand() * 360 - 180
            new_p = Particle(position=(x, y), mass=mass, speed=speed, direction=direction, density=density)
            self.particles.append(new_p)

    def random_plank_step(self):
        for idx, p in enumerate(self.particles):
            new_y = p.get_y() + random.randint(-1, 1)
            new_x = p.get_x() + random.randint(-1, 1)
            p.set_x(new_x)
            p.set_y(new_y)
            self.particles[idx] = p

    def plank_step(self):
        for idx1, p1 in enumerate(self.particles):
            mass1 = p1.get_mass()
            size1 = p1.get_size()
            momentum1 = p1.get_momentum()
            other_forces = list()
            for idx2, p2 in enumerate(self.particles):
                if idx1 != idx2:
                    angle = atan2(p2.get_y()-p1.get_y(), p2.get_x()-p1.get_x()) # radians
                    mass2 = p2.get_mass()
                    size2 = p2.get_size()
                    distance = max(np.linalg.norm(np.array(p1.get_position()) - np.array(p2.get_position())), (size1 + size2)/2)
                    force = Odin.G * mass1 * mass2 / (distance**2)
                    vector_force = (cos(angle)*force, sin(angle)*force)
                    other_forces.append(np.array(vector_force))
            other_forces.append(np.array(momentum1))
            new_momentum1 = tuple(sum(other_forces))
            p1.set_momentum(new_momentum1)
        
        for idx, p in enumerate(self.particles):
            p.move()
            self.particles[idx] = p


    def show_live(self):
        done = False
        screen = pygame.display.set_mode((self.size, self.size))
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            pygame.display.flip()
            screen.fill((0,0,0))
            for p in self.particles:
                x = p.get_x()
                y = p.get_y()
                if (x > 0) and (y > 0) and (x < self.size) and (y < self.size):
                    pygame.draw.circle(screen, p.color, (p.get_x(), p.get_y()), p.get_mass())
            self.plank_step()
            # self.random_plank_step()
            clock.tick(120)

if __name__ == "__main__":
    home = FlatUniverse(Odin.num_particles, Odin.universe_size)
    home.show_live()