# Implementación del problema de Asignación de Frecuencias con Algoritmos Genéticos en Python

## 1. Introducción

El **problema de Asignación de Frecuencias** consiste en asignar frecuencias a un conjunto de transmisores, antenas o estaciones, de modo que se minimicen las interferencias y se respeten restricciones técnicas.

Es un problema clásico de optimización combinatoria con aplicaciones en:

- redes celulares,
- radioenlaces,
- televisión y radiodifusión,
- sistemas satelitales,
- planificación del espectro.

En esta práctica se trabajará con una versión simplificada donde:

- cada transmisor debe recibir una frecuencia,
- algunos pares de transmisores interfieren entre sí,
- transmisores cercanos no deberían usar frecuencias iguales o demasiado próximas.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque permiten explorar muchas configuraciones posibles de asignación y reducir conflictos sin recorrer exhaustivamente todas las combinaciones.

---

## 2. Idea de modelado con Algoritmos Genéticos

En esta implementación:

- cada **gen** representa la frecuencia asignada a un transmisor,
- el **cromosoma** completo representa una solución de asignación.

### Ejemplo de cromosoma

```python
[0, 2, 1, 3, 0, 2]
```

### Objetivo

Se busca una solución que:

- minimice interferencias,
- minimice conflictos por cercanía espectral,
- y use frecuencias de manera adecuada.

---

## 3. Instancia de ejemplo

Se trabajará con:

- 8 transmisores,
- 5 frecuencias posibles,
- una lista de pares conflictivos,
- y restricciones de separación mínima entre frecuencias.

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
n_transmisores = 8
n_frecuencias = 5

conflictos = [
    (0, 1, 1),
    (0, 2, 2),
    (1, 2, 1),
    (1, 3, 2),
    (2, 4, 1),
    (3, 4, 1),
    (3, 5, 2),
    (4, 6, 1),
    (5, 6, 1),
    (5, 7, 2),
    (6, 7, 1),
]

# Cada tupla es (transmisor_a, transmisor_b, separacion_minima)

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
# GRAFO DE INTERFERENCIAS
# =========================
G = nx.Graph()
G.add_nodes_from(range(n_transmisores))
for u, v, sep in conflictos:
    G.add_edge(u, v, sep=sep)

# =========================
# REPRESENTACIÓN
# =========================
def crear_individuo():
    return np.random.randint(0, n_frecuencias, size=n_transmisores)

# =========================
# FITNESS
# =========================
def penalizacion_total(individuo):
    penalizacion = 0
    conflictos_duros = 0
    conflictos_suaves = 0

    for u, v, sep_min in conflictos:
        diff = abs(int(individuo[u]) - int(individuo[v]))
        if diff == 0:
            penalizacion += 100
            conflictos_duros += 1
        elif diff < sep_min:
            penalizacion += 20 * (sep_min - diff)
            conflictos_suaves += 1

    return penalizacion, conflictos_duros, conflictos_suaves

def fitness(individuo):
    penalizacion, _, _ = penalizacion_total(individuo)
    return -penalizacion

# =========================
# OPERADORES GENÉTICOS
# =========================
def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_un_punto(padre1, padre2):
    if random.random() < prob_cruce:
        punto = random.randint(1, n_transmisores - 1)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    return padre1.copy(), padre2.copy()

def mutar(individuo):
    mutado = individuo.copy()
    for i in range(n_transmisores):
        if random.random() < prob_mutacion:
            mutado[i] = random.randint(0, n_frecuencias - 1)
    return mutado

# =========================
# INICIALIZACIÓN
# =========================
poblacion = [crear_individuo() for _ in range(tam_poblacion)]

mejores_fitness = []
promedios_fitness = []
mejores_penalizaciones = []
mejores_duros = []
mejores_suaves = []

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

    penalizacion, duros, suaves = penalizacion_total(mejor_generacion)

    mejores_fitness.append(mejor_fit)
    promedios_fitness.append(promedio_fit)
    mejores_penalizaciones.append(penalizacion)
    mejores_duros.append(duros)
    mejores_suaves.append(suaves)

    if mejor_fit > mejor_fit_global:
        mejor_fit_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
pen_final, duros_final, suaves_final = penalizacion_total(mejor_individuo_global)

print("Mejor cromosoma encontrado:", mejor_individuo_global.tolist())
print("Penalización total:", pen_final)
print("Conflictos duros:", duros_final)
print("Conflictos suaves:", suaves_final)

for t, f in enumerate(mejor_individuo_global):
    print(f"Transmisor {t} -> frecuencia {int(f)}")

# =========================
# GRÁFICO 1: CONVERGENCIA
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_penalizaciones, label="Mejor penalización")
plt.xlabel("Generación")
plt.ylabel("Penalización")
plt.title("Evolución del algoritmo genético")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 2: CONFLICTOS
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_duros, label="Conflictos duros")
plt.plot(mejores_suaves, label="Conflictos suaves")
plt.xlabel("Generación")
plt.ylabel("Cantidad")
plt.title("Evolución de conflictos")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 3: GRAFO DE INTERFERENCIAS
# =========================
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
palette = ["red", "blue", "green", "orange", "purple", "cyan", "yellow"]
node_colors = [palette[int(f) % len(palette)] for f in mejor_individuo_global]

nx.draw(
    G, pos,
    with_labels=True,
    node_color=node_colors,
    node_size=900,
    font_size=10,
    edgecolors="black"
)

edge_labels = {(u, v): f"sep>={sep}" for u, v, sep in conflictos}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Asignación final de frecuencias")
plt.show()

# =========================
# GRÁFICO 4: FRECUENCIAS POR TRANSMISOR
# =========================
plt.figure(figsize=(8, 5))
plt.bar([f"T{i}" for i in range(n_transmisores)], mejor_individuo_global)
plt.xlabel("Transmisor")
plt.ylabel("Frecuencia asignada")
plt.title("Frecuencia por transmisor")
plt.grid(axis="y")
plt.show()
```

---

## 5. Explicación paso a paso

### 5.1 Representación del cromosoma
Cada posición representa un transmisor y el valor almacenado indica la frecuencia asignada.

### 5.2 Función de fitness
La calidad de una solución depende de:

- **conflictos duros**: dos transmisores interferentes con la misma frecuencia,
- **conflictos suaves**: frecuencias demasiado cercanas respecto de la separación mínima requerida.

### 5.3 Selección
Se usa selección por torneo.

### 5.4 Cruce
Se aplica cruce de un punto.

### 5.5 Mutación
Cada transmisor puede cambiar a otra frecuencia posible.

---

## 6. Gráficos para visualizar la solución

- Curva de convergencia de la penalización.
- Evolución de conflictos duros y suaves.
- Grafo de interferencias coloreado según la frecuencia asignada.
- Frecuencia asignada por transmisor.

---

## 7. Librerías utilizadas

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
```

---

## 8. Instalación

```python
!pip install numpy matplotlib networkx
```

---

## 9. Posibles mejoras

- incorporar costos diferentes por frecuencia,
- usar más niveles de interferencia,
- agregar reparación de soluciones,
- usar mutación dirigida a transmisores conflictivos.

---

## 10. Conclusión

La asignación de frecuencias es un problema combinatorio con fuertes restricciones. Los algoritmos genéticos permiten explorar asignaciones alternativas y reducir progresivamente los conflictos de interferencia.
