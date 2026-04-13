# Implementación del problema Vehicle Routing Problem (VRP) con Algoritmos Genéticos en Python

## 1. Introducción

El **Vehicle Routing Problem (VRP)** es un problema clásico de optimización combinatoria y logística.  
Consiste en diseñar rutas para una flota de vehículos que debe atender un conjunto de clientes, partiendo y regresando a un depósito, respetando restricciones como capacidad, cantidad de vehículos o ventanas de tiempo.

En esta práctica se trabajará con una versión simplificada del problema:

- un único depósito,
- varios clientes con demanda,
- varios vehículos con capacidad máxima,
- objetivo de minimizar la distancia total recorrida.

El VRP es un problema difícil porque combina:

- asignación de clientes a vehículos,
- orden de visita de cada ruta,
- restricciones de capacidad,
- enorme espacio de búsqueda.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque permiten explorar simultáneamente muchas configuraciones de rutas y encontrar soluciones de buena calidad sin recurrir a métodos exactos costosos.

---

## 2. Idea de modelado con Algoritmos Genéticos

En esta implementación se usará una representación sencilla:

- el cromosoma es una **permutación de clientes**,
- un **decodificador** recorre el cromosoma y va construyendo rutas,
- cuando agregar un cliente excede la capacidad del vehículo actual, se cierra la ruta y se comienza una nueva.

### Ejemplo de cromosoma

```python
[3, 1, 5, 2, 4, 6]
```

El orden del cromosoma no representa directamente las rutas finales, sino el orden en que el decodificador intenta asignar clientes a vehículos.

### Objetivo

Se busca una solución que:

- visite todos los clientes exactamente una vez,
- respete la capacidad de los vehículos,
- minimice la distancia total recorrida.

---

## 3. Instancia de ejemplo

Se utilizará un ejemplo pequeño con:

- 1 depósito,
- 8 clientes,
- demandas por cliente,
- capacidad fija por vehículo,
- coordenadas 2D para calcular distancias euclidianas.

---

## 4. Código Python completo

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import math

# =========================
# DATOS DEL PROBLEMA
# =========================
deposito = (50, 50)

clientes = {
    1: {"coord": (20, 60), "demanda": 4},
    2: {"coord": (18, 54), "demanda": 6},
    3: {"coord": (30, 40), "demanda": 5},
    4: {"coord": (60, 20), "demanda": 7},
    5: {"coord": (65, 35), "demanda": 3},
    6: {"coord": (80, 55), "demanda": 4},
    7: {"coord": (72, 70), "demanda": 6},
    8: {"coord": (45, 75), "demanda": 5},
}

lista_clientes = list(clientes.keys())
capacidad_vehiculo = 15
max_vehiculos = 4

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 100
n_generaciones = 150
prob_cruce = 0.85
prob_mutacion = 0.10
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# FUNCIONES AUXILIARES
# =========================
def distancia(p1, p2):
    return math.dist(p1, p2)

def crear_individuo():
    individuo = lista_clientes.copy()
    random.shuffle(individuo)
    return individuo

def decodificar(individuo):
    rutas = []
    ruta_actual = []
    carga_actual = 0

    for cliente in individuo:
        demanda = clientes[cliente]["demanda"]
        if carga_actual + demanda <= capacidad_vehiculo:
            ruta_actual.append(cliente)
            carga_actual += demanda
        else:
            rutas.append(ruta_actual)
            ruta_actual = [cliente]
            carga_actual = demanda

    if ruta_actual:
        rutas.append(ruta_actual)

    return rutas

def distancia_ruta(ruta):
    if not ruta:
        return 0

    total = 0
    punto_actual = deposito

    for cliente in ruta:
        coord_cliente = clientes[cliente]["coord"]
        total += distancia(punto_actual, coord_cliente)
        punto_actual = coord_cliente

    total += distancia(punto_actual, deposito)
    return total

def distancia_total(rutas):
    return sum(distancia_ruta(ruta) for ruta in rutas)

def fitness(individuo):
    rutas = decodificar(individuo)
    total = distancia_total(rutas)

    penalizacion = 0
    if len(rutas) > max_vehiculos:
        penalizacion += 1000 * (len(rutas) - max_vehiculos)

    return -(total + penalizacion)

# =========================
# OPERADORES GENÉTICOS
# =========================
def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_orden(padre1, padre2):
    if random.random() >= prob_cruce:
        return padre1.copy(), padre2.copy()

    n = len(padre1)
    a, b = sorted(random.sample(range(n), 2))

    hijo1 = [-1] * n
    hijo2 = [-1] * n

    hijo1[a:b] = padre1[a:b]
    hijo2[a:b] = padre2[a:b]

    def completar(hijo, otro_padre):
        pos = 0
        for gen in otro_padre:
            if gen not in hijo:
                while hijo[pos] != -1:
                    pos += 1
                hijo[pos] = gen
        return hijo

    hijo1 = completar(hijo1, padre2)
    hijo2 = completar(hijo2, padre1)

    return hijo1, hijo2

def mutar_intercambio(individuo):
    mutado = individuo.copy()
    if random.random() < prob_mutacion:
        i, j = random.sample(range(len(mutado)), 2)
        mutado[i], mutado[j] = mutado[j], mutado[i]
    return mutado

# =========================
# INICIALIZACIÓN
# =========================
poblacion = [crear_individuo() for _ in range(tam_poblacion)]

mejores_costos = []
promedios_costos = []
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

        hijo1, hijo2 = cruce_orden(padre1, padre2)
        hijo1 = mutar_intercambio(hijo1)
        hijo2 = mutar_intercambio(hijo2)

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

    if mejor_fit > mejor_fit_global:
        mejor_fit_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
rutas_finales = decodificar(mejor_individuo_global)
costo_final = distancia_total(rutas_finales)

print("Mejor cromosoma encontrado:", mejor_individuo_global)
print("Cantidad de rutas:", len(rutas_finales))
print("Distancia total:", round(costo_final, 2))

for i, ruta in enumerate(rutas_finales, start=1):
    carga = sum(clientes[c]["demanda"] for c in ruta)
    print(f"Ruta {i}: {ruta} | carga={carga} | distancia={round(distancia_ruta(ruta), 2)}")

# =========================
# GRÁFICO 1: CONVERGENCIA
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_costos, label="Mejor costo")
plt.plot(promedios_costos, label="Costo promedio")
plt.xlabel("Generación")
plt.ylabel("Distancia total")
plt.title("Evolución del algoritmo genético para VRP")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 2: RUTAS FINALES
# =========================
plt.figure(figsize=(8, 8))

colores = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

# depósito
plt.scatter(deposito[0], deposito[1], s=300, marker="s")
plt.text(deposito[0], deposito[1], "Depósito", ha="left", va="bottom")

for cliente, data in clientes.items():
    x, y = data["coord"]
    plt.scatter(x, y, s=150)
    plt.text(x, y, f"C{cliente}", ha="right", va="bottom")

for idx, ruta in enumerate(rutas_finales):
    color = colores[idx % len(colores)]
    puntos = [deposito] + [clientes[c]["coord"] for c in ruta] + [deposito]
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    plt.plot(xs, ys, linewidth=2, label=f"Ruta {idx+1}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Rutas finales encontradas")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 3: CARGA POR RUTA
# =========================
cargas = [sum(clientes[c]["demanda"] for c in ruta) for ruta in rutas_finales]
etiquetas = [f"Ruta {i+1}" for i in range(len(rutas_finales))]

plt.figure(figsize=(8, 5))
plt.bar(etiquetas, cargas)
plt.axhline(capacidad_vehiculo, linestyle="--", label="Capacidad")
plt.xlabel("Rutas")
plt.ylabel("Carga")
plt.title("Carga por ruta")
plt.legend()
plt.grid(axis="y")
plt.show()
```

---

## 5. Explicación paso a paso

### 5.1 Representación del cromosoma
Cada individuo es una permutación de clientes.  
Por ejemplo:

```python
[4, 2, 7, 1, 3, 8, 5, 6]
```

El cromosoma no contiene separadores explícitos de rutas. La separación se realiza durante la decodificación según la capacidad del vehículo.

### 5.2 Decodificación
El decodificador recorre la permutación en orden:

- va agregando clientes a la ruta actual,
- si al agregar uno se excede la capacidad, cierra la ruta,
- comienza una nueva.

### 5.3 Función de fitness
La función de fitness busca minimizar la distancia total recorrida y penaliza si se usan más vehículos que el máximo permitido.

En esta implementación:

```python
fitness = -(distancia_total + penalizacion)
```

### 5.4 Selección
Se utiliza selección por torneo.

### 5.5 Cruce
Se utiliza **Order Crossover (OX)**, muy adecuado para cromosomas permutacionales.

### 5.6 Mutación
La mutación se realiza intercambiando dos clientes del cromosoma.

---

## 6. Gráficos para visualizar la solución

### 6.1 Curva de convergencia
Permite analizar cómo disminuye la distancia total a lo largo de las generaciones.

### 6.2 Rutas finales
Muestra el depósito, los clientes y las rutas encontradas para cada vehículo.

### 6.3 Carga por ruta
Permite verificar visualmente la demanda asignada a cada ruta y compararla con la capacidad del vehículo.

---

## 7. Librerías utilizadas

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import math
```

### Opcionales para versiones más avanzadas
También podrían utilizarse:

- `deap`
- `pygad`
- `networkx`
- `plotly`
- `ortools` (para comparación, aunque no es AG)

---

## 8. Instalación

Si hiciera falta:

```python
!pip install numpy matplotlib
```

Y para pruebas más avanzadas:

```python
!pip install deap pygad
```

---

## 9. Posibles mejoras

- incorporar ventanas de tiempo,
- agregar múltiples depósitos,
- usar capacidad heterogénea por vehículo,
- usar operadores de cruce más avanzados,
- agregar elitismo,
- incluir búsqueda local sobre las rutas.

---

## 10. Conclusión

El Vehicle Routing Problem es uno de los problemas más representativos de optimización logística.  
Mediante una representación permutacional y un decodificador simple, los algoritmos genéticos permiten construir rutas factibles y reducir progresivamente la distancia total recorrida.

Además, la visualización de las rutas facilita la interpretación de la solución obtenida y el análisis del reparto de carga entre vehículos.
