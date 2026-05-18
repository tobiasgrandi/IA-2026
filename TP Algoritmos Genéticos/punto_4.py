import random
import matplotlib.pyplot as plt
import networkx as nx
import math

POP_SIZE = 150
GENS = 2000
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.1

states = [
    "Schleswig-Holstein", "Hamburg", "Mecklenburg-Vorpommern",
    "Lower Saxony", "Bremen", "Brandenburg", "Berlin",
    "Saxony-Anhalt", "North Rhine-Westphalia", "Hesse",
    "Thuringia", "Saxony", "Rhineland-Palatinate",
    "Baden-Württemberg", "Bavaria", "Saarland"
]

idx = {s:i for i,s in enumerate(states)}

edges = [
    ("Schleswig-Holstein","Hamburg"), ("Schleswig-Holstein","Lower Saxony"), ("Schleswig-Holstein","Mecklenburg-Vorpommern"),

    ("Hamburg","Lower Saxony"),

    ("Mecklenburg-Vorpommern","Brandenburg"),("Mecklenburg-Vorpommern","Saxony-Anhalt"),

    ("Lower Saxony","Bremen"), ("Lower Saxony","North Rhine-Westphalia"), ("Lower Saxony","Hesse"), ("Lower Saxony","Saxony-Anhalt"), ("Lower Saxony","Thuringia"),

    ("Brandenburg","Berlin"), ("Brandenburg","Saxony"), ("Brandenburg","Saxony-Anhalt"),

    ("Saxony-Anhalt","Thuringia"), ("Saxony-Anhalt","Saxony"),

    ("North Rhine-Westphalia","Hesse"), ("North Rhine-Westphalia","Rhineland-Palatinate"),

    ("Hesse","Thuringia"), ("Hesse","Bavaria"), ("Hesse","Rhineland-Palatinate"), ("Hesse","Baden-Württemberg"),

    ("Thuringia","Saxony"), ("Thuringia","Bavaria"),

    ("Saxony","Bavaria"),

    ("Rhineland-Palatinate","Baden-Württemberg"), ("Rhineland-Palatinate","Saarland"),

    ("Baden-Württemberg","Bavaria"),
]

edges_idx = [(idx[a], idx[b]) for a,b in edges]

def fitness(individual):
    conflicts = sum(1 for a,b in edges_idx if individual[a] == individual[b])
    return len(edges_idx) - conflicts

def create_individual(n_colors):
    return [random.randint(0, n_colors-1) for _ in states]

def selection(population):
    return max(random.sample(population, 3), key=fitness)

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(p1)-1)
        return p1[:point] + p2[point:], p1[point:] + p2[:point]
    return p1, p2

def mutate(ind, n_colors):
    return [
        random.randint(0, n_colors-1) if random.random() < MUTATION_RATE else g
        for g in ind
    ]

def run_ga(n_colors):
    population = [create_individual(n_colors) for _ in range(POP_SIZE)]
    
    best_history = []

    for _ in range(GENS):
        new_population = []
        
        while len(new_population) < POP_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, n_colors)
            c2 = mutate(c2, n_colors)
            new_population.extend([c1, c2])
        
        population = new_population
        
        best = max(population, key=fitness)
        best_history.append(fitness(best))
        
        if fitness(best) == len(edges_idx):
            break
    
    best = max(population, key=fitness)
    return best, best_history

def run_tabu(n_colors, max_iter=GENS, tabu_tenure=7):
    current = create_individual(n_colors)
    best = current[:]

    tabu_list = []
    best_history = []

    for _ in range(max_iter):
        neighbors = []

        # Generar vecinos cambiando el color de un nood
        for i in range(len(states)):
            for c in range(n_colors):
                if c != current[i]:
                    neighbor = current[:]
                    old_color = neighbor[i]
                    neighbor[i] = c
                    move = (i,old_color)
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
        
        if fitness(best) == len(edges_idx):
            break

    return best, best_history

def run_sa(n_colors, max_iter=GENS, T0=10, alpha=0.995):
    current = create_individual(n_colors)
    best = current[:]

    T = T0
    best_history = []

    for _ in range(max_iter):
        i = random.randint(0, len(states)-1)
        new_color = random.randint(0, n_colors-1)

        neighbor = current[:]
        neighbor[i] = new_color

        delta = fitness(neighbor) - fitness(current)

        if delta > 0 or random.random() < math.exp(delta / T):
            current = neighbor

        if fitness(current) > fitness(best):
            best = current[:]

        best_history.append(fitness(best))

        T *= alpha

        if fitness(best) == len(edges_idx):
            break

    return best, best_history

def find_min_colors(start_colors, alg):
    results = {}
    
    k = start_colors
    print(f'*************** ALGORITMO: {alg} ***************')
    
    while k >= 1:
        print(f"\nProbando con {k} colores")
        
        if alg == 'GA':
            solution, history = run_ga(k)
        elif alg == 'TS':
            solution, history = run_tabu(k)
        elif alg == 'SA':
            solution, history = run_sa(k)
        
        if fitness(solution) == len(edges_idx):
            print(f"Solución encontrada con {k} colores")
            results[k] = (solution, history)
            k -= 1
        else:
            print(f"Solución no encontrada con {k} colores")
            break
    
    return results


def plot_coloring(solution, k):
    G = nx.Graph()
    G.add_edges_from(edges)
    
    colors = solution
    
    plt.figure()
    plt.title(f"Coloreo con {k} colores")
    nx.draw(
        G,
        with_labels=True,
        node_color=colors,
        cmap=plt.cm.Set3,
        node_size=800,
        font_size=7
    )
    plt.show()

results = find_min_colors(start_colors=6, alg='SA')

for k, (solution, history) in results.items():
    print(f"\nColores usados con k={k}: {set(solution)}")
    print("Asignación de colores:", solution)
    plot_coloring(solution, k)