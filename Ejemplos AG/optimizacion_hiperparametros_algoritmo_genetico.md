# Implementación del problema de Optimización de Hiperparámetros con Algoritmos Genéticos en Python

## 1. Introducción

La **optimización de hiperparámetros** consiste en encontrar la mejor combinación de parámetros de configuración de un modelo de machine learning.  
A diferencia de los parámetros internos del modelo, los **hiperparámetros** se definen antes del entrenamiento y afectan directamente el desempeño final.

Algunos ejemplos típicos de hiperparámetros son:

- cantidad de vecinos en KNN,
- profundidad máxima en árboles de decisión,
- tasa de aprendizaje,
- cantidad de capas o neuronas,
- regularización,
- tamaño de lote.

Este problema es complejo porque:

- el espacio de búsqueda puede ser grande,
- puede incluir variables enteras, continuas y categóricas,
- existen múltiples óptimos locales,
- cada evaluación requiere entrenar y validar un modelo.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque permiten explorar combinaciones diversas de hiperparámetros y buscar configuraciones de alto desempeño sin evaluar exhaustivamente todas las posibilidades.

---

## 2. Idea de modelado con Algoritmos Genéticos

En esta práctica se usará una versión sencilla del problema, optimizando hiperparámetros de un modelo **K-Nearest Neighbors (KNN)** sobre el dataset clásico **Iris**.

### Hiperparámetros a optimizar

Se buscará optimizar:

- `n_neighbors`: cantidad de vecinos,
- `weights`: esquema de ponderación (`uniform` o `distance`),
- `metric`: métrica de distancia (`euclidean` o `manhattan`).

### Representación del cromosoma

Cada individuo será un vector con tres genes:

```python
[k, weights_code, metric_code]
```

donde:

- `k` es un entero entre 1 y 20,
- `weights_code`:
  - `0` → `"uniform"`
  - `1` → `"distance"`
- `metric_code`:
  - `0` → `"euclidean"`
  - `1` → `"manhattan"`

### Objetivo

Se busca maximizar el desempeño del modelo, medido mediante **accuracy promedio por validación cruzada**.

---

## 3. Código Python completo

```python
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# DATOS DEL PROBLEMA
# =========================
X, y = load_iris(return_X_y=True)

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 60
n_generaciones = 60
prob_cruce = 0.8
prob_mutacion = 0.15
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# DICCIONARIOS DE DECODIFICACIÓN
# =========================
weights_map = {0: "uniform", 1: "distance"}
metric_map = {0: "euclidean", 1: "manhattan"}

# =========================
# REPRESENTACIÓN
# =========================
def crear_individuo():
    return [
        random.randint(1, 20),   # n_neighbors
        random.randint(0, 1),    # weights
        random.randint(0, 1)     # metric
    ]

def decodificar(individuo):
    return {
        "n_neighbors": int(individuo[0]),
        "weights": weights_map[int(individuo[1])],
        "metric": metric_map[int(individuo[2])]
    }

# =========================
# FITNESS
# =========================
def fitness(individuo):
    params = decodificar(individuo)

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric=params["metric"]
        ))
    ])

    scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
    return scores.mean()

# =========================
# OPERADORES GENÉTICOS
# =========================
def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_un_punto(padre1, padre2):
    if random.random() < prob_cruce:
        punto = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1.copy(), padre2.copy()

def mutar(individuo):
    mutado = individuo.copy()

    if random.random() < prob_mutacion:
        mutado[0] = random.randint(1, 20)

    if random.random() < prob_mutacion:
        mutado[1] = random.randint(0, 1)

    if random.random() < prob_mutacion:
        mutado[2] = random.randint(0, 1)

    return mutado

# =========================
# INICIALIZACIÓN
# =========================
poblacion = [crear_individuo() for _ in range(tam_poblacion)]

mejores_fitness = []
promedios_fitness = []
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

    if mejor_fit > mejor_fit_global:
        mejor_fit_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
mejores_params = decodificar(mejor_individuo_global)

print("Mejor cromosoma encontrado:", mejor_individuo_global)
print("Mejores hiperparámetros:", mejores_params)
print("Accuracy promedio CV:", round(mejor_fit_global, 4))

# =========================
# GRÁFICO 1: CONVERGENCIA
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_fitness, label="Mejor fitness")
plt.plot(promedios_fitness, label="Fitness promedio")
plt.xlabel("Generación")
plt.ylabel("Accuracy")
plt.title("Evolución del algoritmo genético")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 2: DISTRIBUCIÓN DE FITNESS FINAL
# =========================
fitness_final = [fitness(ind) for ind in poblacion]

plt.figure(figsize=(8, 5))
plt.hist(fitness_final, bins=10, edgecolor="black")
plt.xlabel("Accuracy")
plt.ylabel("Frecuencia")
plt.title("Distribución de fitness en la población final")
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 3: MEJORES HIPERPARÁMETROS
# =========================
labels = ["n_neighbors", "weights_code", "metric_code"]
values = mejor_individuo_global

plt.figure(figsize=(8, 5))
plt.bar(labels, values)
plt.xlabel("Hiperparámetros")
plt.ylabel("Valor codificado")
plt.title("Codificación del mejor individuo encontrado")
plt.grid(axis="y")
plt.show()
```

---

## 4. Explicación paso a paso

### 4.1 Representación del cromosoma
Cada cromosoma representa una combinación de hiperparámetros.  
Por ejemplo:

```python
[7, 1, 0]
```

significa:

- `n_neighbors = 7`
- `weights = "distance"`
- `metric = "euclidean"`

### 4.2 Decodificación
La función `decodificar()` transforma la representación genética en parámetros reales que puede usar el modelo.

### 4.3 Función de fitness
La función de fitness entrena y valida el modelo KNN usando **validación cruzada de 5 folds** y devuelve el accuracy promedio.

### 4.4 Selección
Se utiliza selección por torneo.

### 4.5 Cruce
Se usa cruce de un punto.

### 4.6 Mutación
Cada gen puede cambiar aleatoriamente a otro valor válido.

---

## 5. Gráficos para visualizar la solución

### 5.1 Curva de convergencia
Permite observar cómo mejora el accuracy a lo largo de las generaciones.

### 5.2 Distribución de fitness final
Muestra qué tan diversa sigue siendo la población al final de la evolución.

### 5.3 Mejor individuo encontrado
Representa la codificación genética del mejor conjunto de hiperparámetros.

---

## 6. Librerías utilizadas

```python
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

### Opcionales para versiones más avanzadas
También podrían utilizarse:

- `deap`
- `pygad`
- `pandas`
- `plotly`
- `optuna` (para comparación, aunque no es AG puro)

---

## 7. Instalación

Si hiciera falta:

```python
!pip install numpy matplotlib scikit-learn
```

Y para variantes más avanzadas:

```python
!pip install deap pygad
```

---

## 8. Posibles mejoras

- optimizar más hiperparámetros,
- usar otro modelo, como SVM o Random Forest,
- incorporar validación cruzada estratificada,
- penalizar modelos más costosos,
- usar codificación mixta con parámetros continuos,
- agregar elitismo,
- comparar con búsqueda aleatoria y grid search.

---

## 9. Conclusión

La optimización de hiperparámetros es un caso muy representativo de búsqueda en espacios complejos y mixtos.  
Los algoritmos genéticos permiten construir estrategias flexibles para explorar combinaciones de configuración y encontrar modelos de buen desempeño sin evaluar exhaustivamente todas las opciones.

En esta práctica se mostró una implementación simple y completamente funcional sobre KNN, útil como base para extender luego a modelos más complejos y escenarios reales.
