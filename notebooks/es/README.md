# Tutorial (Español)

Un único cuaderno ejecutable y centrado en la teoría que recorre toda la idea
detrás de este repositorio desde cero — para lectores sin experiencia en redes
neuronales de grafos ni álgebra de quivers.

**[`walk_attention_tutorial.ipynb`](walk_attention_tutorial.ipynb)** — seis
partes, **24 figuras teóricas hechas a mano** que explican los conceptos y los
ejemplos, más celdas de código que usan `aiq-quivers` (álgebra de caminos) y
`networkx` (para ver las redes) para definir problemas y comprobar las
afirmaciones. Sin gráficos de datos — las figuras llevan las explicaciones.

| Parte | Tema |
|-------|------|
| 0 | Quivers, caminos y multiplicidad (la anatomía) |
| 1 | Qué es el over-squashing (el cuello de botella, K·M^d, el vector que se desborda) |
| 2 | El álgebra de caminos `kQ`: contar caminos con `(A^g)[i,j]` |
| 3 | Los caminos son mensajes; el cociente `kQ/I`; señal vs ruido |
| 4 | El operador de walk **es** atención (soporte sparse, multi-salto) |
| 5 | GAT vs operador de walk vs Walk Attention — la prueba |

Las figuras están en `../figs-theory/es/`. La versión en inglés está en `../en/`.

**La idea en una frase:** el álgebra de caminos de un grafo te dice *qué* nodos
pueden atenderse a través de muchos saltos; la atención aprende luego *cuánto*
importa cada uno — superando tanto al paso de mensajes ordinario como al conteo
fijo de caminos.
