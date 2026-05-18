import random
import math

POP_SIZE = 1500
GENS = 2000
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.3
BITS = 8


def fitness(individual):
    conflicts = 0
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            #misma fila
            if individual[i] == individual[j]:
                conflicts += 1
            # diagonal
            elif abs(individual[i] - individual[j]) == abs(i - j):
                conflicts += 1

    # El máximo número de pares sin conflictos es de 28 (8C2)
    return 28 - conflicts

def create_individual():
    return [random.randint(0, BITS) for _ in range(BITS)]

def selection(population):
    return max(random.sample(population, 3), key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1)-1)
        return p1[:point] + p2[point:], p1[point:] + p2[:point]
    return p1, p2

def mutate(ind):
    return [
        random.randint(0, BITS) if random.random() < MUTATION_RATE else g
        for g in ind
    ]

def run_ga():
    population = [create_individual() for _ in range(POP_SIZE)]
    
    best_history = []

    for _ in range(GENS):
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
        best_history.append(fitness(best))
        
        if fitness(best) == 28:
            break
    
    best = max(population, key=fitness)
    return best, best_history

def run_tabu(max_iter=GENS, tabu_tenure=7):
    current = create_individual()
    best = current[:]

    tabu_list = []
    best_history = []

    for _ in range(max_iter):
        neighbors = []

        # Generar vecinos cambiando las posiciones de las reinas
        for i in range(len(current)):
            for q in range(BITS):
                if q != current[i]:
                    neighbor = current[:]
                    old_pos = neighbor[i]
                    neighbor[i] = q
                    move = (i,old_pos)
                    neighbors.append((neighbor, move))

        best_candidate = None
        best_move = None

        for neighbor, move in neighbors:
            if move not in tabu_list or fitness(neighbor) > fitness(best):
                if best_candidate is None or fitness(neighbor) > fitness(best_candidate):
                    best_candidate = neighbor
                    best_move = move

        current = best_candidate[:]

        #Actualizar lista tabu
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        if fitness(current) > fitness(best):
            best = current[:]

        best_history.append(fitness(best))
        
        if fitness(best) == len(best):
            break

    return best, best_history

def run_sa(max_iter=GENS, T0=10, alpha=0.995):
    current = create_individual()
    best = current[:]

    T = T0
    best_history = []

    for _ in range(max_iter):
        i = random.randint(0, BITS-1)
        new_pos = random.randint(0, BITS)

        neighbor = current[:]
        neighbor[i] = new_pos

        delta = fitness(neighbor) - fitness(current)

        if delta > 0 or random.random() < math.exp(delta / T):
            current = neighbor

        if fitness(current) > fitness(best):
            best = current[:]

        best_history.append(fitness(best))

        T *= alpha

        if fitness(best) == len(best):
            break

    return best, best_history

def solve(alg):

    print(f'*************** ALGORITMO: {alg} ***************')
        
    if alg == 'GA':
        solution, history = run_ga()
    elif alg == 'TS':
        solution, history = run_tabu()
    elif alg == 'SA':
        solution, history = run_sa()
    
    if fitness(solution) == 28:
        print(f"Solución encontrada")
        return solution, history
    print(f"Solución no encontrada")
    return [], []


solution, history = solve(alg='GA')

print("Solution:", solution)
print("Fitness:", fitness(solution))