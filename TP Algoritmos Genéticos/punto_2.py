import random

POP_SIZE = 10
GENS = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
BITS = 5 # 2 ejercicio a, 5 ejercicio b

# EJERCICIO a
#def fitness(ind):
#
#    x0, x1 = ind
#
#    clauses = [
#        (x0 or x1),
#        (x0 or not x1),
#        (not x0 or x1),
#        (not x0 or not x1)
#    ]
#
#    return sum(clauses)

# EJERCICIO b
def fitness(ind):
    x1, x2, x3, x4, x5 = ind

    clauses = [
        (x4 or x2 or x3),
        (x5 or x1 or x2),
        (x4 or x1 or not x3),
        (x3 or x1 or x2),
        (x4 or x1 or not x2),
        (not x5 or not x1 or x4)
    ]

    return sum(clauses)

def create_individual():
    return [random.randint(0,1) for _ in range(BITS)]

def selection(pop):
    return max(random.sample(pop, 2), key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, BITS-1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2

def mutate(ind):
    return [bit if random.random() > MUTATION_RATE else 1-bit for bit in ind]

population = [create_individual() for _ in range(POP_SIZE)]

best_global = None

for gen in range(GENS):
    new_population = []

    while len(new_population) < POP_SIZE:
        p1 = selection(population)
        p2 = selection(population)

        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)

        new_population.extend([c1, c2])

    population = new_population
    best = max(population, key=fitness)
    if best_global is None or fitness(best) > fitness(best_global):
        best_global = best
    
    print(f"Gen {gen}: {best}, Clausuras: {fitness(best)}")

print("\nMejor solución encontrada:", best_global, fitness(best_global))
