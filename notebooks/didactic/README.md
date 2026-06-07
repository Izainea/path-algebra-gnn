# Didactic series — from over-squashing to attention via path algebra
# Serie didáctica — del over-squashing a la atención vía álgebra de caminos

🇬🇧 A gentle, hands-on walkthrough of the idea behind this repo, built for someone
with no background in graph neural networks or quiver algebra. Every cell runs;
every claim is something you can re-derive yourself. Read in order.

🇪🇸 Un recorrido suave y práctico de la idea detrás de este repositorio, pensado
para alguien sin experiencia en redes neuronales de grafos ni álgebra de quivers.
Cada celda se ejecuta; cada afirmación es algo que puedes re-derivar tú mismo.
Léelos en orden.

---

## The five notebooks / Los cinco cuadernos

| # | 🇬🇧 English | 🇪🇸 Español |
|---|-----------|-----------|
| **D1** | What is over-squashing, and why does it hurt? | ¿Qué es el over-squashing y por qué duele? |
| **D2** | Quivers and counting paths (the algebra `kQ`) | Quivers y conteo de caminos (el álgebra `kQ`) |
| **D3** | Paths are messages: redundancy = compression | Los caminos son mensajes: redundancia = compresión |
| **D4** | The walk operator **is** attention | El operador de walk **es** atención |
| **D5** | Putting it together: the network that solves it | Juntándolo todo: la red que lo resuelve |

## The one-sentence idea / La idea en una frase

🇬🇧 **The path algebra of a graph tells you which nodes can "see" each other across
many hops; an attention mechanism then learns how much each one matters — and this
beats both ordinary message passing and fixed path-counting.**

🇪🇸 **El álgebra de caminos de un grafo te dice qué nodos pueden "verse" a través de
muchos saltos; un mecanismo de atención aprende luego cuánto importa cada uno — y
esto supera tanto al paso de mensajes ordinario como al conteo fijo de caminos.**

## Setup / Preparación

🇬🇧 These notebooks use the `oversquash` conda environment (see the top-level
`README.md`). Activate it and register the kernel:

🇪🇸 Estos cuadernos usan el entorno conda `oversquash` (ver el `README.md` raíz).
Actívalo y registra el kernel:

```bash
conda activate oversquash
pip install -e ../../quivers_analysis   # aiq-quivers (local)
pip install -e ../..                    # oversquash
python -m ipykernel install --user --name oversquash
```

Then open `D1_what_is_oversquashing.ipynb`. / Luego abre `D1_what_is_oversquashing.ipynb`.
