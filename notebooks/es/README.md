# Serie de tutoriales (Español)

Una introducción ejecutable y centrada en la teoría a la idea detrás de este
repositorio, para lectores sin experiencia en redes neuronales de grafos ni
álgebra de quivers. **Cada cuaderno tiene al menos cuatro figuras teóricas
hechas a mano** que explican los conceptos y los ejemplos, más celdas de código
que usan `aiq-quivers` y `networkx` para *construir los grafos y comprobar las
afirmaciones*. Léelos en orden.

| # | Cuaderno | Lo que explica |
|---|----------|----------------|
| **P0** | `P0_playground.ipynb` | Quivers, caminos y multiplicidad (la anatomía) |
| **P1** | `P1_oversquashing.ipynb` | Qué es el over-squashing (el cuello de botella, K·M^d, el vector que se desborda) |
| **P2** | `P2_quivers_y_caminos.ipynb` | El álgebra de caminos `kQ`: contar caminos con `(A^g)[i,j]` |
| **P3** | `P3_caminos_son_mensajes.ipynb` | Cada camino es un mensaje; el cociente `kQ/I`; señal vs ruido |
| **P4** | `P4_walk_es_atencion.ipynb` | El operador de walk **es** atención (soporte sparse, multi-salto) |
| **P5** | `P5_juntandolo_todo.ipynb` | GAT vs operador de walk vs Walk Attention — la prueba |

Las figuras teóricas están en `../figs-theory/es/` (SVGs hechos a mano); la serie
en inglés en `../en/` usa las mismas figuras en `../figs-theory/en/`. Las celdas
de código usan `aiq-quivers` (álgebra de caminos) y `networkx` (para dibujar las
redes) — sin gráficos de datos; las figuras llevan las explicaciones.

**La idea en una frase:** el álgebra de caminos de un grafo te dice *qué* nodos
pueden atenderse a través de muchos saltos; la atención aprende luego *cuánto*
importa cada uno — superando tanto al paso de mensajes ordinario como al conteo
fijo de caminos.
