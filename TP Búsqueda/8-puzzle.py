from collections import deque

GOAL = (1,2,3,
        4,5,6,
        7,8,0)

# Movimientos posibles
MOVES = {
    0: [1,3],
    1: [0,2,4],
    2: [1,5],
    3: [0,4,6],
    4: [1,3,5,7],
    5: [2,4,8],
    6: [3,7],
    7: [4,6,8],
    8: [5,7]
}

# Intecambio de posiciones
def cambio(state, i, j):  
    state = list(state)  
    state[i], state[j] = state[j], state[i] 
    return tuple(state)  

# Obtención de lista de sucesores
def obtenerSucesor(state): 
    idx = state.index(0) 
    sucesores = [] 
    for move in MOVES: 
        sucesores.append(cambio(state, idx, move)) 
    return sucesores 

# Búsqueda en anchura
def busAnch(start): 
    queue = deque([(start, [])]) 
    visited = set() 

    while queue:
        current, path = queue.popleft() 

        if current == GOAL:
            return path + [current] 
        
        visited.add(current) 

        for neighbor in obtenerSucesor(current):  
            if neighbor not in visited:
                queue.append((neighbor, path + [current])) 
    
    return None

# Casos de prueba 
start1 = (1,2,3,
        4,5,6,
        7,0,8)


start2 = (1,5,2,
        4,0,3,
        7,8,6)

start3 = (5,8,3,
         0,6,1,
         2,4,7)

def print_solution(path): 
    for step in path: 
        print(step[0:3]) 
        print(step[3:6]) 
        print(step[6:9]) 
        print("------")

solucion = busAnch(start3) 
print_solution(solucion)
print("Cantidad de pasos:", len(solucion)-1) 