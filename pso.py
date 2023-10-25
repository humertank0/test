import numpy as np

class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_position = np.copy(self.position)
        self.best_score = -np.inf

    def update_velocity(self, global_best, w=0.7, c1=1.4, c2=1.4):
        r1 = np.random.uniform(size=len(self.position))
        r2 = np.random.uniform(size=len(self.position))
        social = c1 * r1 * (self.best_position - self.position)
        cognitive = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + social + cognitive

    def update_position(self, minx, maxx):
        self.position += self.velocity
        self.position = np.clip(self.position, minx, maxx)

class PSO:
    def __init__(self, dim, minx, maxx, n_particles):
        self.particles = [Particle(dim, minx, maxx) for _ in range(n_particles)]
        self.global_best = np.random.uniform(low=minx, high=maxx, size=dim)
        self.global_best_score = -np.inf

    def optimize(self, function, iterations):
        for _ in range(iterations):
            for particle in self.particles:
                score = function(particle.position)
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best = particle.position.copy()
            for particle in self.particles:
                particle.update_velocity(self.global_best)
                particle.update_position(minx=-10, maxx=10)

def sphere(x):
    return -sum([i**2 for i in x])

pso = PSO(dim=2, minx=-10, maxx=10, n_particles=30)
pso.optimize(sphere, iterations=1000)
print(pso.global_best_score)
print(pso.global_best)
