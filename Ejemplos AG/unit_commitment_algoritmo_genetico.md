# Implementación del problema de Unit Commitment en Sistemas Eléctricos con Algoritmos Genéticos en Python

## 1. Introducción

El **Unit Commitment (UC)** consiste en decidir qué unidades generadoras deben estar encendidas o apagadas a lo largo de un horizonte temporal para satisfacer la demanda eléctrica al menor costo posible.

Es un problema clásico de optimización en sistemas eléctricos y operación de potencia. Involucra decisiones binarias y restricciones técnicas, por ejemplo:

- demanda mínima a cubrir,
- capacidad máxima de cada unidad,
- costos fijos de operación,
- costos de arranque,
- restricciones temporales.

En esta práctica se trabajará con una versión simplificada donde:

- hay varias unidades generadoras,
- existe una demanda para cada período,
- cada unidad tiene potencia máxima, costo fijo y costo de arranque,
- se busca minimizar el costo total y cubrir la demanda.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque permiten explorar muchas combinaciones posibles de encendido/apagado y encontrar planes operativos de buena calidad.

---

## 2. Idea de modelado con Algoritmos Genéticos

En esta implementación:

- cada **gen** representa el estado de una unidad en un período,
- el cromosoma completo codifica toda la matriz de encendido/apagado.

### Ejemplo de cromosoma

Para 3 unidades y 4 períodos:

```python
[1,0,1,  1,1,0,  0,1,1,  1,1,1]
```

Se interpreta como:

- período 1 → unidades [1,0,1]
- período 2 → unidades [1,1,0]
- período 3 → unidades [0,1,1]
- período 4 → unidades [1,1,1]

### Objetivo

Se busca una solución que:

- cubra la demanda en todos los períodos,
- minimice el costo fijo de operación,
- minimice costos de arranque,
- evite déficit de generación.

---

## 3. Instancia de ejemplo

Se trabajará con:

- 4 unidades generadoras,
- 6 períodos,
- una demanda por período,
- potencia máxima, costo fijo y costo de arranque para cada unidad.

---

## 4. Código Python completo

```python
import random
import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATOS DEL PROBLEMA
# =========================
unidades = [
    {"pmax": 50, "costo_fijo": 20, "costo_arranque": 30},
    {"pmax": 60, "costo_fijo": 24, "costo_arranque": 35},
    {"pmax": 40, "costo_fijo": 16, "costo_arranque": 20},
    {"pmax": 30, "costo_fijo": 12, "costo_arranque": 15},
]

demanda = [70, 90, 110, 95, 80, 60]

n_unidades = len(unidades)
n_periodos = len(demanda)
longitud_cromosoma = n_unidades * n_periodos

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 120
n_generaciones = 150
prob_cruce = 0.8
prob_mutacion = 0.05
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# REPRESENTACIÓN
# =========================
def crear_individuo():
    return np.random.randint(0, 2, size=longitud_cromosoma)

def decodificar(individuo):
    return np.array(individuo).reshape((n_periodos, n_unidades))

# =========================
# FITNESS
# =========================
def evaluar(individuo):
    matriz = decodificar(individuo)

    costo_total = 0
    deficits = []
    generacion_periodo = []

    estado_anterior = np.zeros(n_unidades, dtype=int)

    for t in range(n_periodos):
        estado = matriz[t]
        generacion = sum(estado[i] * unidades[i]["pmax"] for i in range(n_unidades))
        generacion_periodo.append(generacion)

        deficit = max(0, demanda[t] - generacion)
        deficits.append(deficit)

        # costos fijos
        costo_total += sum(estado[i] * unidades[i]["costo_fijo"] for i in range(n_unidades))

        # costos de arranque
        for i in range(n_unidades):
            if estado[i] == 1 and estado_anterior[i] == 0:
                costo_total += unidades[i]["costo_arranque"]

        # penalización por déficit
        costo_total += 200 * deficit

        estado_anterior = estado.copy()

    return costo_total, deficits, generacion_periodo

def fitness(individuo):
    costo_total, _, _ = evaluar(individuo)
    return -costo_total

# =========================
# OPERADORES GENÉTICOS
# =========================
def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_un_punto(padre1, padre2):
    if random.random() < prob_cruce:
        punto = random.randint(1, longitud_cromosoma - 1)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    return padre1.copy(), padre2.copy()

def mutar(individuo):
    mutado = individuo.copy()
    for i in range(longitud_cromosoma):
        if random.random() < prob_mutacion:
            mutado[i] = 1 - mutado[i]
    return mutado

# =========================
# INICIALIZACIÓN
# =========================
poblacion = [crear_individuo() for _ in range(tam_poblacion)]

mejores_costos = []
promedios_costos = []
mejores_deficits = []

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

    costo, deficits, _ = evaluar(mejor_generacion)

    mejores_costos.append(costo)
    promedios_costos.append(-promedio_fit)
    mejores_deficits.append(sum(deficits))

    if mejor_fit > mejor_fit_global:
        mejor_fit_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
costo_final, deficits_final, generacion_final = evaluar(mejor_individuo_global)
matriz_final = decodificar(mejor_individuo_global)

print("Costo total final:", costo_final)
print("Déficit total:", sum(deficits_final))
print("Matriz de compromiso (periodos x unidades):")
print(matriz_final)

# =========================
# GRÁFICO 1: CONVERGENCIA
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_costos, label="Mejor costo")
plt.plot(promedios_costos, label="Costo promedio")
plt.xlabel("Generación")
plt.ylabel("Costo")
plt.title("Evolución del algoritmo genético para Unit Commitment")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 2: DEMANDA VS GENERACIÓN
# =========================
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_periodos + 1), demanda, marker="o", label="Demanda")
plt.plot(range(1, n_periodos + 1), generacion_final, marker="s", label="Generación disponible")
plt.xlabel("Período")
plt.ylabel("Potencia")
plt.title("Demanda vs generación comprometida")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 3: ESTADO DE UNIDADES
# =========================
plt.figure(figsize=(8, 5))
plt.imshow(matriz_final.T, aspect="auto")
plt.xlabel("Período")
plt.ylabel("Unidad")
plt.title("Mapa de encendido/apagado de unidades")
plt.colorbar(label="Estado (0=OFF, 1=ON)")
plt.show()

# =========================
# GRÁFICO 4: DÉFICIT POR PERÍODO
# =========================
plt.figure(figsize=(8, 5))
plt.bar([f"P{t+1}" for t in range(n_periodos)], deficits_final)
plt.xlabel("Período")
plt.ylabel("Déficit")
plt.title("Déficit por período")
plt.grid(axis="y")
plt.show()
```

---

## 5. Explicación paso a paso

### 5.1 Representación del cromosoma
Cada gen indica si una unidad está encendida (`1`) o apagada (`0`) en un período dado.

### 5.2 Decodificación
El vector lineal se convierte en una matriz de tamaño:

```python
(periodos x unidades)
```

### 5.3 Función de fitness
La evaluación incluye:

- costo fijo por unidades encendidas,
- costo de arranque,
- fuerte penalización por déficit de generación.

### 5.4 Selección
Se usa selección por torneo.

### 5.5 Cruce
Se aplica cruce de un punto.

### 5.6 Mutación
Cada gen puede invertirse con cierta probabilidad.

---

## 6. Gráficos para visualizar la solución

- Curva de convergencia del costo.
- Demanda versus generación comprometida.
- Mapa de estados ON/OFF por unidad y período.
- Déficit por período.

---

## 7. Librerías utilizadas

```python
import random
import numpy as np
import matplotlib.pyplot as plt
```

---

## 8. Instalación

```python
!pip install numpy matplotlib
```

---

## 9. Posibles mejoras

- agregar potencia mínima,
- tiempos mínimos de encendido/apagado,
- costos variables por producción,
- despacho económico combinado,
- reparación de soluciones inviables.

---

## 10. Conclusión

El problema de Unit Commitment combina decisiones binarias, costos operativos y restricciones de cobertura de demanda. Los algoritmos genéticos permiten construir planes de encendido/apagado razonables y estudiar cómo evoluciona el costo total bajo distintas configuraciones.
