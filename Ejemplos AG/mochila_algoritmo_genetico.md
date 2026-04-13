# Implementación del Problema de la Mochila con Algoritmos Genéticos en Python

## 1. Introducción

El **Problema de la Mochila (Knapsack Problem)** es un problema clásico de optimización combinatoria.  
Dado un conjunto de objetos, cada uno con un **peso** y un **valor**, se busca seleccionar un subconjunto que maximice el valor total sin superar la capacidad máxima de la mochila.

En su versión **0/1**, cada objeto puede tomarse una sola vez o no tomarse. El espacio de búsqueda tiene tamaño `2^n`, por lo que crece exponencialmente con la cantidad de objetos.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque exploran múltiples combinaciones de manera simultánea y permiten obtener buenas soluciones aproximadas en tiempos razonables.

---

## 2. Modelado con Algoritmos Genéticos

Para resolver la mochila con AG:

- **Cromosoma**: vector binario.
  - `1`: el objeto se incluye.
  - `0`: el objeto no se incluye.
- **Población**: conjunto de soluciones candidatas.
- **Fitness**: valor total con penalización si se excede la capacidad.
- **Selección**: se eligen los mejores individuos para reproducirse.
- **Cruce**: se combinan dos soluciones.
- **Mutación**: se alteran genes para introducir diversidad.

Ejemplo:

```python
[1, 0, 1, 1, 0, 0, 1]
```

---

## 3. Código Python completo

```python
import random
import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATOS DEL PROBLEMA
# =========================
pesos = np.array([12, 7, 11, 8, 9, 6, 7, 3, 5, 4])
valores = np.array([24, 13, 23, 15, 16, 12, 13, 8, 10, 8])
capacidad = 35

n_items = len(pesos)

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 80
n_generaciones = 100
prob_cruce = 0.8
prob_mutacion = 0.05
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# FUNCIONES AUXILIARES
# =========================
def crear_individuo():
    return np.random.randint(0, 2, size=n_items)

def peso_total(individuo):
    return np.sum(individuo * pesos)

def valor_total(individuo):
    return np.sum(individuo * valores)

def fitness(individuo):
    peso = peso_total(individuo)
    valor = valor_total(individuo)
    if peso <= capacidad:
        return valor
    exceso = peso - capacidad
    return valor - 10 * exceso

def reparar(individuo):
    individuo = individuo.copy()
    while peso_total(individuo) > capacidad:
        indices_activos = np.where(individuo == 1)[0]
        if len(indices_activos) == 0:
            break
        idx = np.random.choice(indices_activos)
        individuo[idx] = 0
    return individuo

def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_un_punto(padre1, padre2):
    if random.random() < prob_cruce:
        punto = random.randint(1, n_items - 1)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    return padre1.copy(), padre2.copy()

def mutar(individuo):
    mutado = individuo.copy()
    for i in range(n_items):
        if random.random() < prob_mutacion:
            mutado[i] = 1 - mutado[i]
    return mutado

# =========================
# INICIALIZACIÓN
# =========================
poblacion = [crear_individuo() for _ in range(tam_poblacion)]

mejores_fitness = []
promedios_fitness = []
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

    mejores_fitness.append(mejor_fit)
    promedios_fitness.append(promedio_fit)

    if mejor_fit > mejor_fitness_global:
        mejor_fitness_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
mejor_individuo_global = reparar(mejor_individuo_global)
peso_final = peso_total(mejor_individuo_global)
valor_final = valor_total(mejor_individuo_global)

print("Mejor solución encontrada:", mejor_individuo_global.tolist())
print("Peso total:", peso_final)
print("Valor total:", valor_final)

objetos_seleccionados = np.where(mejor_individuo_global == 1)[0]
print("Objetos seleccionados (índices):", objetos_seleccionados.tolist())

# =========================
# GRÁFICO 1: CONVERGENCIA
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
# GRÁFICO 2: OBJETOS SELECCIONADOS
# =========================
etiquetas = [f"Obj {i}" for i in range(n_items)]
colores = ["green" if gen == 1 else "lightgray" for gen in mejor_individuo_global]

plt.figure(figsize=(10, 5))
plt.bar(etiquetas, valores, color=colores)
plt.xlabel("Objetos")
plt.ylabel("Valor")
plt.title("Objetos seleccionados en la mejor solución")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

# =========================
# GRÁFICO 3: PESOS VS VALORES
# =========================
plt.figure(figsize=(8, 6))
for i in range(n_items):
    color = "green" if mejor_individuo_global[i] == 1 else "red"
    plt.scatter(pesos[i], valores[i], s=120, c=color)
    plt.text(pesos[i] + 0.1, valores[i] + 0.1, f"{i}")

plt.xlabel("Peso")
plt.ylabel("Valor")
plt.title("Distribución peso-valor de los objetos")
plt.grid(True)
plt.show()
```

---

## 4. Explicación paso a paso

### 4.1 Datos del problema
Se definen:

- `pesos`: peso de cada objeto.
- `valores`: beneficio de cada objeto.
- `capacidad`: peso máximo permitido.

### 4.2 Representación
Cada solución es un vector binario de longitud `n_items`.

Por ejemplo:

```python
[1, 0, 1, 0, 1]
```

### 4.3 Fitness
La función de fitness devuelve:

- el valor total si la solución es válida,
- una penalización si el peso supera la capacidad.

### 4.4 Reparación
Después del cruce y la mutación, una solución puede quedar inválida.  
La función `reparar()` elimina objetos hasta cumplir la restricción de capacidad.

### 4.5 Selección
Se usa **selección por torneo**, que elige varios individuos al azar y selecciona el mejor.

### 4.6 Cruce
Se usa **cruce de un punto**, intercambiando fragmentos entre dos padres.

### 4.7 Mutación
Se invierten algunos bits con baja probabilidad para mantener diversidad.

---

## 5. Visualización

### 5.1 Curva de convergencia
Muestra:

- mejor fitness por generación,
- fitness promedio por generación.

### 5.2 Objetos seleccionados
Usa barras para resaltar qué objetos integran la mejor solución.

### 5.3 Relación peso-valor
Ubica cada objeto en el plano `(peso, valor)` y diferencia los seleccionados de los no seleccionados.

---

## 6. Librerías utilizadas

```python
import random
import numpy as np
import matplotlib.pyplot as plt
```

### Librerías opcionales
- `deap`
- `pygad`
- `pandas`
- `plotly`

---

## 7. Instalación

Si hiciera falta:

```python
!pip install numpy matplotlib
```

Y para versiones más avanzadas:

```python
!pip install deap pygad
```

---

## 8. Posibles mejoras

- agregar **elitismo**,
- cambiar operadores de cruce,
- ajustar la penalización,
- usar mutación adaptativa,
- extender a mochila multidimensional o múltiples mochilas.

---

## 9. Conclusión

El problema de la mochila es ideal para introducir algoritmos genéticos, porque combina representación simple, restricciones claras y resultados fáciles de visualizar. Además, sirve como base para luego avanzar a problemas más complejos de optimización combinatoria.
