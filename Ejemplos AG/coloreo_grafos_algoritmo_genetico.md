# Implementación del problema de Coloreo de Grafos con Algoritmos Genéticos en Python

## 1. Introducción

El **problema de coloreo de grafos** consiste en asignar colores a los nodos de un grafo de modo que **dos nodos adyacentes no compartan el mismo color**.

Es un problema clásico de optimización combinatoria y teoría de grafos, con aplicaciones en:

- planificación de horarios,
- asignación de frecuencias,
- asignación de recursos,
- compiladores,
- partición de problemas.

En su versión de optimización, el objetivo es:

- minimizar la cantidad de colores usados,
- evitando conflictos entre nodos vecinos.

En esta práctica se trabajará con una versión donde se fija una cantidad máxima de colores y se busca encontrar una asignación válida con la menor cantidad efectiva posible de colores y sin conflictos.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque permiten explorar muchas coloraciones posibles sobre espacios de búsqueda grandes y con múltiples óptimos locales.

---

## 2. Idea de modelado con Algoritmos Genéticos

En esta implementación:

- cada **gen** representa el color asignado a un nodo,
- el **cromosoma completo** representa una coloración del grafo.

### Ejemplo de cromosoma

```python
[0, 1, 2, 0, 1, 2]
```

Esto significa:

- nodo 0 → color 0
- nodo 1 → color 1
- nodo 2 → color 2
- nodo 3 → color 0
- nodo 4 → color 1
- nodo 5 → color 2

### Objetivo

Se busca una solución que:

- minimice los conflictos entre nodos adyacentes,
- y secundariamente minimice la cantidad de colores realmente utilizados.

---

## 3. Instancia de ejemplo

Se usará un grafo pequeño de ejemplo con 8 nodos.

```python
aristas = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 4),
    (2, 3), (2, 5),
    (3, 5), (3, 6),
    (4, 5), (4, 7),
    (5, 6), (5, 7),
    (6, 7)
]
```

Se trabajará inicialmente con un máximo de 4 colores posibles.

---

## 4. Código Python completo

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =========================
# DATOS DEL PROBLEMA
# =========================
n_nodos = 8
aristas = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 4),
    (2, 3), (2, 5),
    (3, 5), (3, 6),
    (4, 5), (4, 7),
    (5, 6), (5, 7),
    (6, 7)
]

max_colores = 4

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 100
n_generaciones = 120
prob_cruce = 0.8
prob_mutacion = 0.08
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# GRAFO
# =========================
G = nx.Graph()
G.add_nodes_from(range(n_nodos))
G.add_edges_from(aristas)

# =========================
# REPRESENTACIÓN
# =========================
def crear_individuo():
    return np.random.randint(0, max_colores, size=n_nodos)

# =========================
# FITNESS
# =========================
def contar_conflictos(individuo):
    conflictos = 0
    for u, v in aristas:
        if individuo[u] == individuo[v]:
            conflictos += 1
    return conflictos

def cantidad_colores_usados(individuo):
    return len(set(individuo))

def fitness(individuo):
    conflictos = contar_conflictos(individuo)
    colores_usados = cantidad_colores_usados(individuo)
    return -(100 * conflictos + colores_usados)

# =========================
# OPERADORES GENÉTICOS
# =========================
def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_un_punto(padre1, padre2):
    if random.random() < prob_cruce:
        punto = random.randint(1, n_nodos - 1)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    return padre1.copy(), padre2.copy()

def mutar(individuo):
    mutado = individuo.copy()
    for i in range(n_nodos):
        if random.random() < prob_mutacion:
            mutado[i] = random.randint(0, max_colores - 1)
    return mutado

# =========================
# INICIALIZACIÓN
# =========================
poblacion = [crear_individuo() for _ in range(tam_poblacion)]

mejores_fitness = []
promedios_fitness = []
mejores_conflictos = []
mejores_colores = []

mejor_individuo_global = None
mejor_fit_global = -float("inf")

# =========================
# BUCLE EVOLUTIVO
# =========================
for generacion in range(n_generaciones):
    nueva_poblacion = []

    while len(nueva_poblacion) < tam_poblacion:
        padre1 = seleccion_torneo(poblacion)
        padre2 = seleccion_torneo(poblacion)

        hijo1, hijo2 = cruce_un_punto(padre1, padre2)
        hijo1 = mutar(hijo1)
        hijo2 = mutar(hijo2)

        nueva_poblacion.append(hijo1)
        if len(nueva_poblacion) < tam_poblacion:
            nueva_poblacion.append(hijo2)

    poblacion = nueva_poblacion

    fitness_poblacion = [fitness(ind) for ind in poblacion]
    mejor_generacion = max(poblacion, key=fitness)
    mejor_fit = fitness(mejor_generacion)
    promedio_fit = np.mean(fitness_poblacion)

    mejores_fitness.append(mejor_fit)
    promedios_fitness.append(promedio_fit)
    mejores_conflictos.append(contar_conflictos(mejor_generacion))
    mejores_colores.append(cantidad_colores_usados(mejor_generacion))

    if mejor_fit > mejor_fit_global:
        mejor_fit_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
conflictos_finales = contar_conflictos(mejor_individuo_global)
colores_finales = cantidad_colores_usados(mejor_individuo_global)

print("Mejor cromosoma encontrado:", mejor_individuo_global.tolist())
print("Conflictos finales:", conflictos_finales)
print("Cantidad de colores usados:", colores_finales)

for nodo, color in enumerate(mejor_individuo_global):
    print(f"Nodo {nodo} -> color {color}")

# =========================
# GRÁFICO 1: CONVERGENCIA DEL FITNESS
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_fitness, label="Mejor fitness")
plt.plot(promedios_fitness, label="Fitness promedio")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del algoritmo genético")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 2: CONFLICTOS POR GENERACIÓN
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_conflictos, label="Conflictos del mejor individuo")
plt.xlabel("Generación")
plt.ylabel("Cantidad de conflictos")
plt.title("Evolución de conflictos")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 3: GRAFO COLOREADO
# =========================
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

palette = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "pink"]
node_colors = [palette[c % len(palette)] for c in mejor_individuo_global]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    node_size=900,
    font_size=10,
    edgecolors="black"
)

plt.title("Mejor coloreo encontrado")
plt.show()

# =========================
# GRÁFICO 4: COLORES ASIGNADOS POR NODO
# =========================
plt.figure(figsize=(8, 5))
plt.bar([f"N{i}" for i in range(n_nodos)], mejor_individuo_global)
plt.xlabel("Nodo")
plt.ylabel("Código de color")
plt.title("Color asignado a cada nodo")
plt.grid(axis="y")
plt.show()
```

---

## 5. Explicación paso a paso

### 5.1 Representación del cromosoma
Cada posición del cromosoma representa un nodo del grafo y el valor almacenado indica el color asignado.

### 5.2 Función de fitness
La calidad de una solución depende de dos elementos:

- **cantidad de conflictos**: dos nodos conectados con el mismo color,
- **cantidad de colores usados**.

Se utiliza una penalización fuerte por conflictos para priorizar soluciones válidas.

### 5.3 Selección
Se usa selección por torneo.

### 5.4 Cruce
Se aplica cruce de un punto, combinando segmentos de dos soluciones parentales.

### 5.5 Mutación
Cada nodo puede cambiar a otro color posible con una cierta probabilidad.

---

## 6. Gráficos para visualizar la solución

### 6.1 Curva de convergencia del fitness
Muestra la evolución del fitness del mejor individuo y del promedio poblacional.

### 6.2 Evolución de conflictos
Permite verificar si el algoritmo reduce progresivamente la cantidad de conflictos.

### 6.3 Grafo coloreado
Muestra visualmente el coloreo final hallado por el AG.

### 6.4 Colores asignados por nodo
Permite observar la codificación final por nodo.

---

## 7. Librerías utilizadas

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
```

### Opcionales para versiones más avanzadas
También podrían utilizarse:

- `deap`
- `pygad`
- `plotly`
- `pandas`

---

## 8. Instalación

Si hiciera falta:

```python
!pip install numpy matplotlib networkx
```

Y para probar frameworks genéticos:

```python
!pip install deap pygad
```

---

## 9. Posibles mejoras

- intentar minimizar explícitamente el número cromático,
- usar operadores de reparación,
- usar mutación dirigida hacia nodos conflictivos,
- trabajar con grafos más grandes,
- incorporar elitismo,
- comparar con heurísticas greedy de coloreo.

---

## 10. Conclusión

El problema de coloreo de grafos es un caso clásico de optimización combinatoria donde los algoritmos genéticos resultan útiles para explorar muchas asignaciones de colores posibles.

Con una codificación simple por nodo y una función de fitness que penalice conflictos, es posible obtener soluciones válidas y visualmente interpretables, además de estudiar el comportamiento evolutivo del algoritmo sobre un problema NP-completo.
