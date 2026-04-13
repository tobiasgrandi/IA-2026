# Implementación del problema de Plegamiento de Proteínas con Algoritmos Genéticos en Python

## 1. Introducción

El **problema de Plegamiento de Proteínas (Protein Folding Problem)** consiste en determinar una conformación espacial estable para una cadena de aminoácidos.  
En su forma realista es uno de los problemas más complejos de la bioinformática, ya que la cantidad de estructuras posibles crece de manera extremadamente rápida con la longitud de la secuencia.

En esta práctica se trabajará con una **versión simplificada en 2D**, basada en el modelo **HP (Hydrophobic-Polar)**:

- `H`: aminoácido hidrofóbico,
- `P`: aminoácido polar.

La idea del modelo es que los aminoácidos hidrofóbicos tienden a agruparse, por lo que una buena conformación será aquella que genere más contactos `H-H` no consecutivos dentro de la cadena.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque permiten explorar muchas configuraciones distintas del plegamiento y buscar estructuras de menor energía sin tener que recorrer exhaustivamente todo el espacio de búsqueda.

---

## 2. Idea de modelado con Algoritmos Genéticos

En esta versión simplificada:

- la proteína es una secuencia de caracteres `H` y `P`,
- la cadena se pliega sobre una grilla bidimensional,
- cada gen del cromosoma representa un movimiento relativo.

### Movimientos posibles

Se usan tres movimientos relativos:

- `0`: seguir recto,
- `1`: girar a la izquierda,
- `2`: girar a la derecha.

Si la secuencia tiene `n` aminoácidos, el cromosoma tendrá longitud `n - 2`, porque:

- el primer aminoácido se fija en `(0, 0)`,
- el segundo en `(1, 0)`,
- y a partir de allí cada gen define cómo continúa la cadena.

### Objetivo

Se busca una conformación que:

- no tenga solapamientos,
- genere la mayor cantidad posible de contactos `H-H` no consecutivos.

---

## 3. Instancia de ejemplo

Se utilizará una secuencia HP corta para hacer visible el comportamiento del algoritmo:

```python
secuencia = "HPPHHPHPH"
```

---

## 4. Código Python completo

```python
import random
import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATOS DEL PROBLEMA
# =========================
secuencia = "HPPHHPHPH"
n = len(secuencia)

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 100
n_generaciones = 150
prob_cruce = 0.8
prob_mutacion = 0.08
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# DIRECCIONES
# =========================
# 0=este, 1=norte, 2=oeste, 3=sur
direcciones = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1)
}

def girar(direccion_actual, movimiento):
    # movimiento: 0 recto, 1 izquierda, 2 derecha
    if movimiento == 0:
        return direccion_actual
    elif movimiento == 1:
        return (direccion_actual + 1) % 4
    else:
        return (direccion_actual - 1) % 4

# =========================
# REPRESENTACIÓN
# =========================
def crear_individuo():
    return np.random.randint(0, 3, size=n - 2)

# =========================
# DECODIFICACIÓN
# =========================
def decodificar(individuo):
    coords = [(0, 0), (1, 0)]
    direccion_actual = 0  # este
    ocupadas = {(0, 0), (1, 0)}
    solapamiento = False

    x, y = 1, 0

    for mov in individuo:
        direccion_actual = girar(direccion_actual, mov)
        dx, dy = direcciones[direccion_actual]
        x, y = x + dx, y + dy

        if (x, y) in ocupadas:
            solapamiento = True

        coords.append((x, y))
        ocupadas.add((x, y))

    return coords, solapamiento

# =========================
# ENERGÍA / FITNESS
# =========================
def contactos_hh(coords, secuencia):
    contactos = 0
    for i in range(len(coords)):
        for j in range(i + 2, len(coords)):  # evitar consecutivos
            if secuencia[i] == 'H' and secuencia[j] == 'H':
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                if dist == 1:
                    contactos += 1
    return contactos

def fitness(individuo):
    coords, solapamiento = decodificar(individuo)
    contactos = contactos_hh(coords, secuencia)

    if solapamiento:
        return -100 + contactos
    return contactos

# =========================
# OPERADORES GENÉTICOS
# =========================
def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def cruce_un_punto(padre1, padre2):
    if random.random() < prob_cruce:
        punto = random.randint(1, len(padre1) - 1)
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        return hijo1, hijo2
    return padre1.copy(), padre2.copy()

def mutar(individuo):
    mutado = individuo.copy()
    for i in range(len(mutado)):
        if random.random() < prob_mutacion:
            opciones = [0, 1, 2]
            opciones.remove(mutado[i])
            mutado[i] = random.choice(opciones)
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
coords_final, solapamiento_final = decodificar(mejor_individuo_global)
contactos_final = contactos_hh(coords_final, secuencia)

print("Mejor cromosoma encontrado:", mejor_individuo_global.tolist())
print("Fitness final:", mejor_fit_global)
print("Contactos H-H:", contactos_final)
print("¿Hay solapamiento?:", solapamiento_final)
print("Coordenadas:", coords_final)

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
# GRÁFICO 2: PLEGAMIENTO FINAL
# =========================
plt.figure(figsize=(7, 7))

for i in range(len(coords_final) - 1):
    x1, y1 = coords_final[i]
    x2, y2 = coords_final[i + 1]
    plt.plot([x1, x2], [y1, y2], linewidth=2)

for i, (x, y) in enumerate(coords_final):
    color = "red" if secuencia[i] == "H" else "lightblue"
    plt.scatter(x, y, s=500, c=color, edgecolors="black")
    plt.text(x, y, f"{secuencia[i]}{i}", ha="center", va="center", fontsize=9)

plt.title("Mejor plegamiento encontrado")
plt.axis("equal")
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 3: CONTACTOS H-H
# =========================
plt.figure(figsize=(7, 7))

for i in range(len(coords_final) - 1):
    x1, y1 = coords_final[i]
    x2, y2 = coords_final[i + 1]
    plt.plot([x1, x2], [y1, y2], linewidth=2)

for i, (x, y) in enumerate(coords_final):
    color = "red" if secuencia[i] == "H" else "lightblue"
    plt.scatter(x, y, s=500, c=color, edgecolors="black")
    plt.text(x, y, f"{secuencia[i]}{i}", ha="center", va="center", fontsize=9)

for i in range(len(coords_final)):
    for j in range(i + 2, len(coords_final)):
        if secuencia[i] == 'H' and secuencia[j] == 'H':
            x1, y1 = coords_final[i]
            x2, y2 = coords_final[j]
            if abs(x1 - x2) + abs(y1 - y2) == 1:
                plt.plot([x1, x2], [y1, y2], linestyle="--", linewidth=1)

plt.title("Contactos H-H no consecutivos")
plt.axis("equal")
plt.grid(True)
plt.show()
```

---

## 5. Explicación paso a paso

### 5.1 Representación del cromosoma
Cada gen representa una decisión local de plegado:

- seguir recto,
- girar a la izquierda,
- girar a la derecha.

### 5.2 Decodificación
El cromosoma se transforma en una secuencia de coordenadas en la grilla 2D.  
Durante este proceso se detecta si hay **solapamientos**, es decir, si dos aminoácidos ocupan la misma celda.

### 5.3 Función de fitness
La calidad de una conformación se evalúa según:

- la cantidad de contactos `H-H` no consecutivos,
- una fuerte penalización si hay solapamiento.

En esta versión:

```python
fitness = contactos_hh
```

y si hay superposición:

```python
fitness = -100 + contactos_hh
```

### 5.4 Selección
Se usa selección por torneo.

### 5.5 Cruce
Se aplica cruce de un punto.

### 5.6 Mutación
Cada gen puede cambiar a otro movimiento posible.

---

## 6. Gráficos para visualizar la solución

### 6.1 Curva de convergencia
Permite ver cómo mejora el fitness a lo largo de las generaciones.

### 6.2 Plegamiento final
Muestra la conformación espacial de la proteína en la grilla.

### 6.3 Contactos H-H
Resalta visualmente los contactos favorables entre aminoácidos hidrofóbicos no consecutivos.

---

## 7. Librerías utilizadas

```python
import random
import numpy as np
import matplotlib.pyplot as plt
```

### Opcionales para versiones más avanzadas
También podrían utilizarse:

- `deap`
- `pygad`
- `plotly`
- `biopython`

---

## 8. Instalación

Si hiciera falta:

```python
!pip install numpy matplotlib
```

Y para una versión más avanzada:

```python
!pip install deap pygad biopython
```

---

## 9. Posibles mejoras

- usar secuencias HP más largas,
- extender el modelo a 3D,
- utilizar función de energía más realista,
- agregar elitismo,
- evitar explícitamente individuos inválidos,
- comparar distintos operadores de cruce y mutación.

---

## 10. Conclusión

El plegamiento de proteínas es un problema extremadamente complejo, pero incluso una versión simplificada como el modelo HP permite estudiar ideas fundamentales de optimización estructural.

Los algoritmos genéticos resultan útiles para explorar conformaciones, mejorar progresivamente la calidad de las soluciones y visualizar de forma concreta cómo una secuencia puede plegarse buscando configuraciones de menor energía.
