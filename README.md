Here's a GitHub README:

---

# 🗺️ TSP Explorer

An interactive Traveling Salesman Problem visualizer built as a final project for CS3C — Data Structures & Algorithms at Foothill College. Drop cities on a real map, pick an algorithm, and watch it solve the tour in real time.

---

## 🧠 What It Does

Lets you place cities on a live OpenStreetMap, select one of four TSP algorithms, and either run it instantly or watch it work step by step through an animation. A "Run All" mode runs every algorithm on the same city set and shows a side-by-side comparison table of cost and time.

---

## ⚙️ Algorithms Implemented

- **Brute Force** — tries every possible ordering, guaranteed optimal, practical up to ~10 cities (`O(n!)`)
- **Held-Karp** — dynamic programming, builds the optimal solution from reused sub-problems, practical up to ~18 cities (`O(n² · 2ⁿ)`)
- **Nearest Neighbor** — greedy heuristic, always walks to the closest unvisited city, scales to thousands (`O(n²)`)
- **Nearest Neighbor + 2-opt** — improves the NN tour by repeatedly uncrossing edges until no swap helps, averages ~1% above optimal (`O(n²)` per pass)

---

## 🧪 Tested

39 unit tests across all four algorithms covering known-answer cases, cross-validation between exact algorithms, asymmetric distance matrices, edge cases (single city, degenerate inputs), adversarial examples where greedy demonstrably fails, and direct tests of internal helper functions.

---

## 📊 Benchmarked

Algorithms were pushed until they crossed a 10-second wall to confirm their theoretical Big-O in practice. Brute Force hit its limit at 12 cities, Held-Karp at 20. The heuristics ran to 300 cities without breaking a sweat.

---

## 🛠️ Built With

- Python + Tkinter
- tkintermapview (OpenStreetMap, no API key required — tiles cached locally for offline use)
- Fully hand-written algorithm implementations — no TSP libraries

---

## 🚀 Getting Started

```bash
pip install tkintermapview
python TSP_explorer.py
```

---

## 📁 Project Structure

```
TSP_brute_force.py        # Brute force with hand-written permutation generator
TSP_held_karp.py          # Held-Karp with hand-written combinations generator
TSP_nearest_neighbor.py   # Nearest neighbor heuristic
TSP_two_opt.py            # 2-opt local search
TSP_test.py               # 39-test unit test suite
TSP_benchmark.py          # Benchmark script, outputs CSV
TSP_explorer.py           # Interactive map application
preset_cities.json        # 15 real US cities for quick demos
```

---

## 🎓 Course

CS3C — Data Structures and Algorithms, Foothill College
AI-assisted final project exploring generative AI as a development collaborator.

---