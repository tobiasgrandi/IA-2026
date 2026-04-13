# Implementación del problema Job Shop Scheduling con Algoritmos Genéticos en Python

## 1. Introducción

El **Job Shop Scheduling Problem (JSSP)** es uno de los problemas clásicos de planificación y optimización combinatoria.  
Consiste en programar un conjunto de trabajos, donde cada trabajo está formado por una secuencia ordenada de operaciones, y cada operación debe ejecutarse en una máquina específica durante un tiempo determinado.

El objetivo más habitual es **minimizar el makespan**, es decir, el tiempo total necesario para completar todos los trabajos.

Este problema es altamente complejo porque:

- las operaciones deben respetar un orden interno dentro de cada trabajo,
- cada máquina sólo puede procesar una operación a la vez,
- el espacio de búsqueda crece de forma explosiva,
- existen muchísimos óptimos locales.

Los **algoritmos genéticos (AG)** son adecuados para este problema porque permiten explorar distintas secuencias de planificación y encontrar buenas soluciones aproximadas en tiempos razonables.

---

## 2. Idea de modelado con Algoritmos Genéticos

Para resolver JSSP con AG se puede usar una representación basada en **secuencias de trabajos**.

En este enfoque:

- cada trabajo aparece repetido tantas veces como operaciones tenga,
- el orden de aparición de cada trabajo en el cromosoma define la prioridad con la que se programan sus operaciones,
- un **decodificador** transforma el cromosoma en un cronograma válido.

### Ejemplo de cromosoma

```python
[0, 1, 2, 0, 1, 2, 0, 1, 2]
```

Si hay 3 trabajos y cada uno tiene 3 operaciones, este cromosoma indica el orden relativo de despacho de las operaciones.

---

## 3. Instancia de ejemplo

Se utilizará una instancia pequeña con 3 trabajos y 3 máquinas.

Cada trabajo está compuesto por una lista de operaciones del tipo:

```python
(maquina, duracion)
```

Ejemplo:

```python
trabajos = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 1)]
]
```

Esto significa que:

- el trabajo 0 necesita:
  - máquina 0 durante 3 unidades,
  - luego máquina 1 durante 2,
  - luego máquina 2 durante 2.

---

## 4. Código Python completo

```python
import random
import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATOS DEL PROBLEMA
# =========================
trabajos = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 1)]
]

n_trabajos = len(trabajos)
n_maquinas = 3
ops_por_trabajo = [len(t) for t in trabajos]
longitud_cromosoma = sum(ops_por_trabajo)

# =========================
# PARÁMETROS DEL AG
# =========================
tam_poblacion = 80
n_generaciones = 120
prob_cruce = 0.85
prob_mutacion = 0.10
tam_torneo = 3

random.seed(42)
np.random.seed(42)

# =========================
# REPRESENTACIÓN
# =========================
base_cromosoma = []
for job_id, cant_ops in enumerate(ops_por_trabajo):
    base_cromosoma.extend([job_id] * cant_ops)

def crear_individuo():
    individuo = base_cromosoma.copy()
    random.shuffle(individuo)
    return individuo

# =========================
# DECODIFICADOR
# =========================
def decodificar(individuo):
    siguiente_op = [0] * n_trabajos
    disponible_maquina = [0] * n_maquinas
    fin_trabajo = [0] * n_trabajos
    agenda = []

    for job_id in individuo:
        op_idx = siguiente_op[job_id]
        maquina, duracion = trabajos[job_id][op_idx]

        inicio = max(fin_trabajo[job_id], disponible_maquina[maquina])
        fin = inicio + duracion

        agenda.append({
            "job": job_id,
            "op": op_idx,
            "machine": maquina,
            "start": inicio,
            "end": fin,
            "duration": duracion
        })

        fin_trabajo[job_id] = fin
        disponible_maquina[maquina] = fin
        siguiente_op[job_id] += 1

    makespan = max(item["end"] for item in agenda)
    return agenda, makespan

# =========================
# FITNESS
# =========================
def fitness(individuo):
    _, makespan = decodificar(individuo)
    return -makespan  # maximización del fitness = minimizar makespan

# =========================
# OPERADORES GENÉTICOS
# =========================
def seleccion_torneo(poblacion):
    candidatos = random.sample(poblacion, tam_torneo)
    return max(candidatos, key=fitness).copy()

def reparar(individuo):
    esperado = {j: ops_por_trabajo[j] for j in range(n_trabajos)}
    actual = {j: individuo.count(j) for j in range(n_trabajos)}

    faltantes = []
    sobrantes_idx = []

    for j in range(n_trabajos):
        if actual[j] < esperado[j]:
            faltantes.extend([j] * (esperado[j] - actual[j]))
        elif actual[j] > esperado[j]:
            exceso = actual[j] - esperado[j]
            encontrados = 0
            for idx, val in enumerate(individuo):
                if val == j and encontrados < exceso:
                    sobrantes_idx.append(idx)
                    encontrados += 1

    random.shuffle(faltantes)
    for idx, nuevo_val in zip(sobrantes_idx, faltantes):
        individuo[idx] = nuevo_val

    return individuo

def cruce_orden_multiconjunto(padre1, padre2):
    if random.random() >= prob_cruce:
        return padre1.copy(), padre2.copy()

    a, b = sorted(random.sample(range(len(padre1)), 2))

    hijo1 = [-1] * len(padre1)
    hijo2 = [-1] * len(padre2)

    hijo1[a:b] = padre1[a:b]
    hijo2[a:b] = padre2[a:b]

    def completar(hijo, otro_padre, segmento):
        usados = {j: segmento.count(j) for j in range(n_trabajos)}
        requerido = {j: ops_por_trabajo[j] for j in range(n_trabajos)}
        pos_libres = [i for i, x in enumerate(hijo) if x == -1]

        carga = []
        for gen in otro_padre:
            if usados.get(gen, 0) < requerido[gen]:
                carga.append(gen)
                usados[gen] = usados.get(gen, 0) + 1

        for pos, gen in zip(pos_libres, carga):
            hijo[pos] = gen
        return hijo

    hijo1 = completar(hijo1, padre2, padre1[a:b])
    hijo2 = completar(hijo2, padre1, padre2[a:b])

    hijo1 = reparar(hijo1)
    hijo2 = reparar(hijo2)
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

        hijo1, hijo2 = cruce_orden_multiconjunto(padre1, padre2)
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

    mejores_fitness.append(-mejor_fit)
    promedios_fitness.append(-promedio_fit)

    if mejor_fit > mejor_fit_global:
        mejor_fit_global = mejor_fit
        mejor_individuo_global = mejor_generacion.copy()

# =========================
# RESULTADOS
# =========================
agenda_final, makespan_final = decodificar(mejor_individuo_global)

print("Mejor cromosoma encontrado:", mejor_individuo_global)
print("Makespan final:", makespan_final)

for op in agenda_final:
    print(op)

# =========================
# GRÁFICO 1: CONVERGENCIA
# =========================
plt.figure(figsize=(10, 5))
plt.plot(mejores_fitness, label="Mejor makespan")
plt.plot(promedios_fitness, label="Makespan promedio")
plt.xlabel("Generación")
plt.ylabel("Makespan")
plt.title("Evolución del algoritmo genético para JSSP")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# GRÁFICO 2: DIAGRAMA DE GANTT
# =========================
plt.figure(figsize=(10, 5))

colores = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

for item in agenda_final:
    maquina = item["machine"]
    inicio = item["start"]
    duracion = item["duration"]
    job = item["job"]
    op = item["op"]

    plt.barh(maquina, duracion, left=inicio, height=0.5, color=colores[job], edgecolor="black")
    plt.text(inicio + duracion / 2, maquina, f"J{job}-O{op}", ha="center", va="center", color="white", fontsize=9)

plt.yticks(range(n_maquinas), [f"M{m}" for m in range(n_maquinas)])
plt.xlabel("Tiempo")
plt.ylabel("Máquina")
plt.title("Diagrama de Gantt de la mejor solución")
plt.grid(axis="x")
plt.show()
```

---

## 5. Explicación paso a paso

### 5.1 Representación del cromosoma
Cada cromosoma es una lista donde cada identificador de trabajo aparece tantas veces como operaciones tenga ese trabajo.

Ejemplo:

```python
[2, 0, 1, 2, 0, 1, 0, 2, 1]
```

Este cromosoma no indica directamente máquinas ni tiempos, sino el orden relativo en que se despachan las operaciones.

### 5.2 Decodificación
El cromosoma se transforma en un cronograma válido mediante un decodificador que:

- lleva control de la siguiente operación pendiente de cada trabajo,
- controla cuándo queda libre cada máquina,
- respeta la precedencia entre operaciones del mismo trabajo.

### 5.3 Función de fitness
Como el objetivo es minimizar el makespan, puede definirse:

```python
fitness = -makespan
```

De esta forma el AG sigue maximizando fitness, pero en realidad minimiza el tiempo total.

### 5.4 Selección
Se utiliza selección por torneo.

### 5.5 Cruce
Se usa un cruce de orden adaptado a multiconjuntos, ya que cada trabajo aparece varias veces en el cromosoma.

### 5.6 Mutación
La mutación intercambia dos posiciones del cromosoma.  
Esto modifica el orden de despacho y puede producir una secuencia mejor.

---

## 6. Gráficos para visualizar la solución

### 6.1 Curva de convergencia
Muestra cómo evoluciona el makespan a lo largo de las generaciones.

### 6.2 Diagrama de Gantt
Permite visualizar:

- qué operación se ejecuta en cada máquina,
- en qué instante comienza,
- cuánto dura,
- y el orden final del cronograma.

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
- `pandas`
- `plotly`

---

## 8. Instalación

Si hiciera falta:

```python
!pip install numpy matplotlib
```

Y para variantes más avanzadas:

```python
!pip install deap pygad
```

---

## 9. Posibles mejoras

- agregar **elitismo**,
- usar operadores de cruce específicos para scheduling,
- incorporar múltiples objetivos,
- considerar tiempos de setup,
- agregar restricciones adicionales,
- comparar con heurísticas clásicas como dispatching rules.

---

## 10. Conclusión

El problema Job Shop Scheduling es un excelente ejemplo de aplicación de algoritmos genéticos a problemas de planificación compleja.  
Su dificultad radica en la combinación de restricciones de precedencia y recursos compartidos, pero mediante una buena representación y un decodificador adecuado es posible obtener soluciones factibles y de buena calidad.

Además, el uso del diagrama de Gantt permite interpretar visualmente la solución encontrada y analizar el comportamiento del algoritmo.
