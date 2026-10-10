# Serie de tutoriales (Español)

Una introducción práctica y ejecutable a la idea detrás de este repositorio, para
lectores sin experiencia en redes neuronales de grafos ni álgebra de quivers.
**Cada cuaderno tiene al menos tres figuras.** Léelos en orden.

| # | Cuaderno | Lo que construyes |
|---|----------|-------------------|
| **P0** | `P0_playground.ipynb` | Dibujar grafos pequeños y contar sus caminos (4 grafos) |
| **P1** | `P1_oversquashing.ipynb` | Qué es el over-squashing — mensajes apretados en un vector |
| **P2** | `P2_quivers_y_caminos.ipynb` | Quivers y conteo de caminos: `(A^g)[i,j]`, el álgebra `kQ` |
| **P3** | `P3_caminos_son_mensajes.ipynb` | Cada camino es un mensaje; redundancia = compresión |
| **P4** | `P4_walk_es_atencion.ipynb` | El operador de walk **es** atención (sparse, multi-salto) |
| **P5** | `P5_juntandolo_todo.ipynb` | Entrenar GAT vs walkraw vs WalkAttention — la prueba |

**La idea en una frase:** el álgebra de caminos de un grafo te dice *qué* nodos
pueden atenderse a través de muchos saltos; la atención aprende luego *cuánto*
importa cada uno — superando tanto al paso de mensajes ordinario como al conteo
fijo de caminos.

Todo el código de gráficos está en `oversquash.viz` (compartido con la serie en
inglés en `../en/`), así que las figuras son idénticas en ambos idiomas.
Preparación: activa el entorno conda `oversquash` (ver el `README.md` del repo),
luego abre `P0_playground.ipynb`.
