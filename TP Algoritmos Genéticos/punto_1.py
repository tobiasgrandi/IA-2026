import random
import matplotlib.pyplot as plt

# Parámetros
POP_SIZE = 20
GENS = 50
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.8
BITS = 6  # porque 2^6

# Función objetivo
def fitness(x):
    return x**5 - x**3 - 2*x**2

# Generar individuo (cadena binaria)
def create_individual():
    return ''.join(random.choice('01') for _ in range(BITS))

# Decodificar binario a entero
def decode(ind):
    return int(ind, 2)

# Selección por torneo
def selection(pop):
    a, b = random.sample(pop, 2)
    return a if fitness(decode(a)) > fitness(decode(b)) else b

# Cruce de un punto
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, BITS-1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2

# Mutación bit a bit
def mutate(ind):
    new_ind = ''
    for bit in ind:
        if random.random() < MUTATION_RATE:
            new_ind += '1' if bit == '0' else '0'
        else:
            new_ind += bit
    return new_ind

# Inicialización
population = [create_individual() for _ in range(POP_SIZE)]

best_fitness_history = []
avg_fitness_history = []
best_global = None
best_global_fitness = float('-inf')

for gen in range(GENS):
    new_population = []
    
    # Reproducción
    while len(new_population) < POP_SIZE:
        p1 = selection(population)
        p2 = selection(population)
        
        c1, c2 = crossover(p1, p2)
        
        c1 = mutate(c1)
        c2 = mutate(c2)
        
        new_population.extend([c1, c2])
    
    population = new_population[:POP_SIZE]

    fitness_values = [fitness(decode(ind)) for ind in population]

    best_fitness = max(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)

    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)

    gen_best = max(population, key=lambda ind: fitness(decode(ind)))
    gen_best_fitness = fitness(decode(gen_best))

    if gen_best_fitness > best_global_fitness:
        best_global = gen_best
        best_global_fitness = gen_best_fitness

plt.figure()
plt.plot(best_fitness_history, label="Mejor fitness")
plt.plot(avg_fitness_history, label="Fitness promedio")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del algoritmo genético")
plt.legend()
plt.grid()

plt.show()

# Resultado final
x_best = decode(best_global)
print(f"Mejor solución global: x = {x_best}, f(x) = {best_global_fitness}")