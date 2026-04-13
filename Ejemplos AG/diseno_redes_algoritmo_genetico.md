# Implementación del problema de Diseño de Redes con Algoritmos Genéticos en Python

## 1. Introducción

El **problema de Diseño de Redes** consiste en definir una topología de conexión entre un conjunto de nodos de manera que la red resultante cumpla ciertos objetivos de desempeño y costo.

Según la formulación, pueden buscarse soluciones que:

- minimicen el costo total de instalación,
- aseguren conectividad completa,
- mejoren la tolerancia a fallas,
- reduzcan latencia,
- mantengan redundancia o robustez estructural.

Es un problema combinatorio complejo porque cada enlace posible puede estar presente o ausente, lo que hace que el espacio de búsqueda crezca de forma **exponencial**. Además, suele involucrar múltiples restricciones y objetivos contrapuestos.

Los **algoritmos genéticos (AG)** son apropiados para este problema porque permiten explorar distintas topologías de red y optimizar simultáneamente criterios de costo y conectividad.

---

## 2. Idea de modelado con Algoritmos Genéticos

En esta práctica se trabajará con una versión simplificada del problema:

- se tiene un conjunto fijo de nodos,
- entre algunos pares de nodos puede instalarse un enlace,
- cada enlace tiene un costo,
- se desea encontrar una red que:
  - conecte todos los nodos,
  - y minimice el costo total.

### Representación del cromosoma

Cada cromosoma será un vector binario:

- `1`: el enlace está presente,
- `0`: el enlace no está presente.

Ejemplo:

```python
[1, 0, 1, 1, 0, 0, 1]
```

Cada posición representa un enlace posible de la red.

---

## 3. Instancia de ejemplo

Se trabajará con una red de 6 nodos y un conjunto de enlaces posibles con costos asociados.

Ejemplo:

```python
enlaces = [
    (0, 1, 4),
    (0, 2, 3),
    (0, 3, 7),
    (1, 2, 6),
    (1, 4, 5),
    (2, 3, 4),
    (2, 4, 2),
    (2, 5, 8),
    (3, 5, 3),
    (4, 5, 6)
]
```

Cada tupla contiene:

```python
(origen, destino, costo)
```

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
n_nodos = 6
enlaces = [
    (0, 1, 4),
    (0, 2, 3),
    (0, 3, 7),
    (1, 2, 6),
    (1, 4, 5),
    (2, 3, 4),
    (2, 4, 2),
    (2, 5, 8),
    (3, 5, 3),
    (4, 5, 6)
]

n_enlaces = len(enlaces)

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 100
n_generaciones = 120
prob_cruce = 0.8
prob_mutacion = 0.05
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# FUNCIONES AUXILIARES
# =========================
def crear_individuo():
    return np.random.randint(0, 2, size=n_enlaces)

def construir_grafo(individuo):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodos))
    for gen, (u, v, costo) in zip(individuo, enlaces):
        if gen == 1:
            G.add_edge(u, v, weight=costo)
    return G

def costo_total(individuo):
    return sum(costo for gen, (_, _, costo) in zip(individuo, enlaces) if gen == 1)

def es_conexa(individuo):
    G = construir_grafo(individuo)
    return nx.is_connected(G)

def cantidad_componentes(individuo):
    G = construir_grafo(individuo)
    return nx.number_connected_components(G)

def fitness(individuo):
    costo = costo_total(individuo)
    componentes = cantidad_componentes(individuo)

    if componentes == 1:
        return -costo
    else:
        penalizacion = 100 * (componentes - 1)
        return -(costo + penalizacion)

def reparar(individuo):
    individuo = individuo.copy()
    G = construir_grafo(individuo)

    while not nx.is_connected(G):
        componentes = list(nx.connected_components(G))
        comp_a = list(componentes[0])
        comp_b = list(componentes[1])

        posibles = []
        for i, (u, v, costo) in enumerate(enlaces):
            if individuo[i] == 0:
                if (u in comp_a and v in comp_b) or (u in comp_b and v in comp_a):
                    posibles.append((i, costo))

        if posibles:
            mejor_idx = min(posibles, key=lambda x: x[1])[0]
            individuo[mejor_idx] = 1
        else:
            idx = random.randint(0, n_enlaces - 1)
            individuo[idx] = 1

        G = construir_grafo(individuo)

    return individuo

def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_un_punto(padre1, padre2):
    if random.random() < prob_cruce:
        punto = random.randint(1, n_enlaces - 1)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    return padre1.copy(), padre2.copy()

def mutar(individuo):
    mutado = individuo.copy()
    for i in range(len(mutado)):
        if random.random() < prob_mutacion:
            mutado[i] = 1 - mutado[i]
    return mutado

# =========================
# INICIALIZACIÓN
# =========================
poblacion = [crear_individuo() for _ in range(tam_poblacion)]
poblacion = [reparar(ind) for ind in poblacion]

mejores_costos = []
promedios_costos = []
mejor_individuo_global = None
mejor_fitness_global = -float("inf")

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

        hijo1 = reparar(hijo1)
        hijo2 = reparar(hijo2)

        nueva_poblacion.append(hijo1)
        if len(nueva_poblacion) < tam_poblacion:
            nueva_poblacion.append(hijo2)

    poblacion = nueva_poblacion

    fitness_poblacion = [fitness(ind) for ind in poblacion]
    mejor_generacion = max(poblacion, key=fitness)
    mejor_fit = fitness(mejor_generacion)
    promedio_fit = np.mean(fitness_poblacion)

    mejores_costos.append(-mejor_fit)
    promedios_costos.append(-promedio_fit)

    if mejor_fit > mejor_fitness_global:
        mejor_fitness_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
mejor_individuo_global = reparar(mejor_individuo_global)
costo_final = costo_total(mejor_individuo_global)
G_final = construir_grafo(mejor_individuo_global)

print("Mejor solución encontrada:", mejor_individuo_global.tolist())
print("Costo total:", costo_final)
print("¿La red es conexa?:", nx.is_connected(G_final))
print("Enlaces seleccionados:")

for gen, (u, v, costo) in zip(mejor_individuo_global, enlaces):
    if gen == 1:
        print(f"{u} - {v}  costo={costo}")

# =========================
# GRÁFICO 1: CONVERGENCIA
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_costos, label="Mejor costo")
plt.plot(promedios_costos, label="Costo promedio")
plt.xlabel("Generación")
plt.ylabel("Costo")
plt.title("Evolución del algoritmo genético")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 2: RED FINAL
# =========================
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G_final, seed=42)
nx.draw(G_final, pos, with_labels=True, node_size=800, font_size=10)
labels = nx.get_edge_attributes(G_final, 'weight')
nx.draw_networkx_edge_labels(G_final, pos, edge_labels=labels)
plt.title("Topología de la mejor red encontrada")
plt.show()

# =========================
# GRÁFICO 3: COSTO DE ENLACES SELECCIONADOS
# =========================
etiquetas = []
costos = []
for gen, (u, v, costo) in zip(mejor_individuo_global, enlaces):
    if gen == 1:
        etiquetas.append(f"{u}-{v}")
        costos.append(costo)

plt.figure(figsize=(8, 5))
plt.bar(etiquetas, costos)
plt.xlabel("Enlaces")
plt.ylabel("Costo")
plt.title("Costos de los enlaces seleccionados")
plt.grid(axis="y")
plt.show()
```

---

## 5. Explicación paso a paso

### 5.1 Representación
Cada individuo es un vector binario que indica qué enlaces forman parte de la topología propuesta.

### 5.2 Construcción del grafo
A partir del cromosoma se construye un grafo usando `networkx`.

### 5.3 Fitness
La función de fitness:

- minimiza el costo total,
- penaliza las redes no conexas.

En este ejemplo:

```python
fitness = -(costo_total + penalización)
```

De esta forma, el AG sigue maximizando fitness mientras minimiza el costo y favorece la conectividad.

### 5.4 Reparación
Si una solución no es conexa, se agregan enlaces hasta conectar todos los componentes.

### 5.5 Selección
Se utiliza **selección por torneo**.

### 5.6 Cruce
Se aplica **cruce de un punto**.

### 5.7 Mutación
Cada bit puede cambiar con una probabilidad baja, lo que implica agregar o quitar enlaces.

---

## 6. Gráficos para visualizar la solución

### 6.1 Curva de convergencia
Permite observar cómo evoluciona el costo a lo largo de las generaciones.

### 6.2 Topología final
Muestra el grafo final encontrado por el algoritmo genético.

### 6.3 Costos de enlaces seleccionados
Permite ver cuáles fueron los enlaces elegidos y qué peso económico aporta cada uno.

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

Y para probar bibliotecas especializadas:

```python
!pip install deap pygad
```

---

## 9. Posibles mejoras

- agregar criterios de redundancia,
- optimizar simultáneamente costo y resiliencia,
- imponer grado mínimo por nodo,
- modelar anchos de banda o latencias,
- incorporar objetivos múltiples,
- comparar con árbol generador mínimo.

---

## 10. Conclusión

El problema de Diseño de Redes es un caso claro de optimización combinatoria con espacio de búsqueda exponencial.  
Los algoritmos genéticos permiten representar topologías de forma simple, incorporar restricciones de conectividad y encontrar soluciones de buena calidad.

Además, el uso de grafos y visualización facilita el análisis de la estructura final de la red obtenida.
