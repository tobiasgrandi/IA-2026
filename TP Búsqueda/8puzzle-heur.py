from simpleai.search import SearchProblem, astar
import time

GOAL = (1,2,3,
        4,5,6,
        7,8,0)

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

class PuzzleProblem(SearchProblem):

    def actions(self, state):
        idx = state.index(0)
        return MOVES[idx]

    def result(self, state, action):
        state = list(state)
        idx = state.index(0)
        state[idx], state[action] = state[action], state[idx]
        return tuple(state)

    def is_goal(self, state):
        return state == GOAL

    def heuristic(self, state):
        return self.h_manhattan(state)
    
    def h_mal_colocada(self, state):
        return sum(1 for i in range(9) 
                   if state[i] != 0 and state[i] != GOAL[i])
    
    def h_manhattan(self, state):
        distance = 0
        for i in range(9):
            if state[i] != 0:
                valor = state[i]
                goal_index = GOAL.index(valor)

                x1, y1 = divmod(i, 3)
                x2, y2 = divmod(goal_index, 3)

                distance += abs(x1 - x2) + abs(y1 - y2)
        return distance
    

# Casos de prueba
start1 = (1,2,3,
        4,5,6,
        7,0,8)


start2 = (1,5,2,
        4,0,3,
        7,8,6)

start3 = (2,1,4,
        7,0,5,
        8,3,6)

problem = PuzzleProblem(start3)
inicio = time.time()
result1 = astar(problem, graph_search=True)
fin = time.time()


def print_result(result):
    for action, state in result.path():
        print(state[0:3])
        print(state[3:6])
        print(state[6:9])
        print("------")

print("Fichas mal colocadas:")
print_result(result1)
print("Pasos:", len(result1.path()) - 1)
print("Costo total:", result1.cost)
print("Tiempo de ejecución:", fin - inicio, "segundos")

