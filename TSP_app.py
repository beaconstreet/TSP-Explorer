"""
Matthew Heitman - CS3C - Traveling Salesman Problem - Final Project

TSP Explorer — interactive companion app for the Traveling Salesman Problem
final project. Drop cities on a real OpenStreetMap, run any of the four
implemented algorithms, and watch the tour build step by step.

Requires: tkintermapview  (pip install tkintermapview)
"""

import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import tkintermapview

from TSP_brute_force import brute_force
from TSP_held_karp import held_karp
from TSP_nearest_neighbor import nearest_neighbor
from TSP_two_opt import apply_swap, swap_improves_tour, tour_length, two_opt

# ── Safety limits measured empirically during the benchmarking phase ─────────
# Brute force: 11 cities already takes ~2.6 s; growth is O(n!)
BRUTE_FORCE_MAX_CITIES = 10
# Held-Karp:  16 cities takes several seconds; growth is O(n² · 2ⁿ)
HELD_KARP_MAX_CITIES = 15

# ── Sidebar fonts — one place to tune all text sizes ─────────────────────────
FONT_TITLE   = ("Helvetica", 18, "bold")   # app title
FONT_HEADER  = ("Helvetica", 14, "bold")   # section headers
FONT_BODY    = ("Helvetica", 12)           # normal text in sections
FONT_BODY_B  = ("Helvetica", 12, "bold")   # emphasized body text
FONT_NOTE    = ("Helvetica", 11)           # hints / warnings
FONT_MONO    = ("Courier", 13)             # numbers (time, cost)
FONT_MONO_SM = ("Courier", 12)             # city list, comparison table
FONT_BUTTON  = ("Helvetica", 12)           # buttons
BUTTON_FG    = "#222222"                   # dark text, consistent across buttons

# Short (1-2 sentence) explanations shown in the Algorithm info tooltip
ALGO_DESCRIPTIONS = {
    "Brute Force": "Tries every possible ordering of the cities and keeps "
                   "the cheapest one. Always finds the true optimal tour, "
                   "but the number of orderings explodes factorially as "
                   "cities are added.",
    "Held-Karp": "Finds the true optimal tour using dynamic programming, "
                "reusing the cost of shared partial routes instead of "
                "recomputing them. Exact like Brute Force, but scales "
                "much further before becoming too slow.",
    "Nearest Neighbor": "Builds a tour by always hopping to whichever "
                        "unvisited city is closest. Very fast, but greedy — "
                        "it isn't guaranteed to find the optimal tour.",
    "Nearest Neighbor + 2-opt": "Starts from a Nearest Neighbor tour, then "
                                "repeatedly uncrosses pairs of edges whenever "
                                "doing so shortens the tour. Usually gets "
                                "noticeably closer to optimal than Nearest "
                                "Neighbor alone.",
}

ABOUT_TEXT = (
    "TSP Explorer\n"
    "Created by Matt Heitman\n\n"
    "This app is a hands-on companion to a Traveling Salesman Problem "
    "final project. Place cities by clicking the map (or load a preset "
    "set of US cities), pick an algorithm, and run it to see the tour it "
    "finds.\n\n"
    "City 1 (in placement order) is always the fixed start and end point "
    "of the tour.\n\n"
    "• Run — computes instantly in the background, no animation.\n"
    "• Run with Animation — watch the algorithm build or search for "
    "the tour step by step.\n"
    "• Run All — runs every algorithm (within its safe city limit) on "
    "the same cities and compares cost and time side by side.\n\n"
    "Brute Force and Held-Karp are exact algorithms and are limited to a "
    "small number of cities since their running time grows very quickly. "
    "Nearest Neighbor and Nearest Neighbor + 2-opt are fast heuristics "
    "that scale to much larger city sets."
)


class Tooltip:
    """A small floating popup shown on hover, positioned near the cursor."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         bg="#ffffe0", fg="#222", relief=tk.SOLID, borderwidth=1,
                         font=FONT_NOTE, wraplength=360, padx=8, pady=6)
        label.pack()

    def _hide(self, event=None):
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


# Display labels shown in the algorithm dropdown
ALGO_BRUTE_FORCE = f"Brute Force (limit {BRUTE_FORCE_MAX_CITIES} cities)"
ALGO_HELD_KARP   = f"Held-Karp (limit {HELD_KARP_MAX_CITIES} cities)"
ALGO_NN          = "Nearest Neighbor"
ALGO_NN_2OPT     = "Nearest Neighbor + 2-opt"

# ms between animation frames for NN / 2-opt step-through
ANIMATION_DELAY_MS = 150
# ms to pause after each Brute Force improvement so it's visible on screen
BF_IMPROVEMENT_DELAY_MS = 600

# Tile cache — any tile you pan over gets written here and survives offline.
TILE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiles.db")

# Preset city files
PRESET_15_FILES = ["preset_cities.json", "TSP_preset_cities.py"]
PRESET_45_FILE  = "preset_cities_45.json"


# ── Distance helpers ──────────────────────────────────────────────────────────

def euclidean_distances(cities):
    """
    Flat Euclidean distance on (lat, lon) pairs — same formula used in
    TSP_benchmark.py's euclidean_distances(), just applied to geo coordinates
    instead of synthetic (x, y) points.
    """
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dlat = cities[i]["lat"] - cities[j]["lat"]
            dlon = cities[i]["lon"] - cities[j]["lon"]
            dist[i][j] = math.hypot(dlat, dlon)
    return dist


# ── Step generators for animation ────────────────────────────────────────────

def nn_steps(distances):
    """Yields the partial (closed) route after each city is added by NN."""
    n = len(distances)
    unvisited = set(range(1, n))
    current = 0
    route = [0]
    while unvisited:
        closest = min(unvisited, key=lambda c: distances[current][c])
        route.append(closest)
        unvisited.remove(closest)
        current = closest
        yield list(route) + [0]


def two_opt_steps(route, distances):
    """
    Yields the route after each improving 2-opt swap. Re-uses swap_improves_tour
    and apply_swap from TSP_two_opt directly so the logic stays in one place.
    """
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if swap_improves_tour(route, distances, i, j):
                    route = apply_swap(route, i, j)
                    improved = True
                    yield list(route)


# ── Main application ──────────────────────────────────────────────────────────

class TSPApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TSP Explorer")
        self.root.geometry("1200x750")

        self.cities: list[dict] = []   # {"name", "lat", "lon"}
        self.markers: list = []
        self.path_line = None

        self.result_queue: queue.Queue = queue.Queue()
        self.running = False
        self.animation_job = None
        self._timer_job = None
        self._elapsed_start = 0.0

        self._build_ui()
        self._poll_results()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar = tk.Frame(self.root, width=340, bg="#f5f5f5", relief=tk.FLAT)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0)
        sidebar.pack_propagate(False)

        pad = {"padx": 8, "pady": 3}
        wrap = 320   # wraplength for multi-line labels, matches sidebar width

        title_frame = tk.Frame(sidebar, bg="#f5f5f5")
        title_frame.pack(fill=tk.X, padx=8, pady=(10, 4))
        tk.Label(title_frame, text="TSP Explorer", bg="#f5f5f5",
                 font=FONT_TITLE).pack(side=tk.LEFT)
        tk.Button(title_frame, text="ⓘ About", fg=BUTTON_FG, font=FONT_NOTE,
                  relief=tk.FLAT, bg="#e8e8e8", padx=6, pady=2,
                  command=self._show_about).pack(side=tk.RIGHT)

        # City controls
        tk.Label(sidebar, text="Cities", bg="#f5f5f5",
                 font=FONT_HEADER).pack(anchor=tk.W, padx=8)
        tk.Button(sidebar, text="Load Preset Cities (10 US cities)", fg=BUTTON_FG,
                  font=FONT_BUTTON,
                  command=self.load_preset_10).pack(fill=tk.X, **pad)
        tk.Button(sidebar, text="Load Preset Cities (15 US cities)", fg=BUTTON_FG,
                  font=FONT_BUTTON,
                  command=self.load_preset_15).pack(fill=tk.X, **pad)
        tk.Button(sidebar, text="Load Preset Cities (45 US cities)", fg=BUTTON_FG,
                  font=FONT_BUTTON,
                  command=self.load_preset_45).pack(fill=tk.X, **pad)
        tk.Button(sidebar, text="Clear Cities", fg=BUTTON_FG, font=FONT_BUTTON,
                  command=self.clear_cities).pack(fill=tk.X, **pad)

        tk.Label(sidebar, text="Click the map to place cities. City 1 is "
                 "the fixed start/end.", bg="#f5f5f5", font=FONT_NOTE,
                 fg="#666", wraplength=wrap, justify=tk.LEFT).pack(anchor=tk.W, padx=8)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # Algorithm selector
        algo_header = tk.Frame(sidebar, bg="#f5f5f5")
        algo_header.pack(fill=tk.X, padx=8)
        tk.Label(algo_header, text="Algorithm", bg="#f5f5f5",
                 font=FONT_HEADER).pack(side=tk.LEFT)
        algo_info = tk.Label(algo_header, text=" ⓘ", bg="#f5f5f5", fg="#3a7bd5",
                             font=FONT_HEADER, cursor="hand2")
        algo_info.pack(side=tk.LEFT)
        algo_tooltip_text = "\n\n".join(
            f"{name}: {desc}" for name, desc in ALGO_DESCRIPTIONS.items()
        )
        Tooltip(algo_info, algo_tooltip_text)

        self.algo_var = tk.StringVar(value=ALGO_NN)
        self.algo_menu = ttk.Combobox(
            sidebar, textvariable=self.algo_var, state="readonly",
            font=FONT_BODY,
            values=[ALGO_BRUTE_FORCE, ALGO_HELD_KARP, ALGO_NN, ALGO_NN_2OPT],
        )
        self.algo_menu.pack(fill=tk.X, **pad)

        self._limit_note = tk.StringVar(value="")
        tk.Label(sidebar, textvariable=self._limit_note, bg="#f5f5f5",
                 font=FONT_NOTE, fg="#b85c00",
                 wraplength=wrap, justify=tk.LEFT).pack(anchor=tk.W, padx=8)
        self.algo_menu.bind("<<ComboboxSelected>>", self._update_limit_note)
        self.algo_menu.bind("<Configure>", self._update_limit_note)

        # Run buttons — row 1
        btn_frame1 = tk.Frame(sidebar, bg="#f5f5f5")
        btn_frame1.pack(fill=tk.X, **pad)
        self.run_btn = tk.Button(btn_frame1, text="▶  Run",
                                 command=self.run_algorithm,
                                 bg="#3a7bd5", fg=BUTTON_FG,
                                 font=("Helvetica", 12, "bold"),
                                 relief=tk.FLAT, padx=10, pady=4)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.run_all_btn = tk.Button(btn_frame1, text="Run All",
                                     command=self.run_all,
                                     bg="#4caf50", fg=BUTTON_FG, font=FONT_BUTTON,
                                     relief=tk.FLAT, padx=10, pady=4)
        self.run_all_btn.pack(side=tk.LEFT)

        # Run buttons — row 2
        self.run_anim_btn = tk.Button(sidebar, text="▶  Run with Animation",
                                      command=self.run_animated_algo,
                                      bg="#8e44ad", fg=BUTTON_FG, font=FONT_BUTTON,
                                      relief=tk.FLAT, padx=10, pady=4)
        self.run_anim_btn.pack(fill=tk.X, **pad)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # Status / results
        tk.Label(sidebar, text="Results", bg="#f5f5f5",
                 font=FONT_HEADER).pack(anchor=tk.W, padx=8)

        self.status_var = tk.StringVar(value="Ready — place cities or load a preset.")
        tk.Label(sidebar, textvariable=self.status_var, bg="#f5f5f5",
                 wraplength=wrap, justify=tk.LEFT, font=FONT_BODY,
                 fg="#333").pack(anchor=tk.W, padx=8, pady=2)

        result_frame = tk.Frame(sidebar, bg="#f5f5f5")
        result_frame.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(result_frame, text="Time:", bg="#f5f5f5",
                 font=FONT_BODY).grid(row=0, column=0, sticky=tk.W)
        self.timer_var = tk.StringVar(value="—")
        tk.Label(result_frame, textvariable=self.timer_var, bg="#f5f5f5",
                 font=FONT_MONO).grid(row=0, column=1, sticky=tk.W, padx=6)
        tk.Label(result_frame, text="Cost:", bg="#f5f5f5",
                 font=FONT_BODY).grid(row=1, column=0, sticky=tk.W)
        self.cost_var = tk.StringVar(value="—")
        tk.Label(result_frame, textvariable=self.cost_var, bg="#f5f5f5",
                 font=FONT_MONO).grid(row=1, column=1, sticky=tk.W, padx=6)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # City list — the flexible section: it absorbs vertical shrink when the
        # window gets smaller, so the controls above and table below stay visible.
        tk.Label(sidebar, text="Cities (in placement order)", bg="#f5f5f5",
                 font=FONT_HEADER).pack(anchor=tk.W, padx=8)
        list_frame = tk.Frame(sidebar, bg="#f5f5f5")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.city_listbox = tk.Listbox(list_frame, font=FONT_MONO_SM,
                                       yscrollcommand=scrollbar.set, height=4)
        self.city_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.city_listbox.yview)

        # Comparison table (shown after Run All)
        self.compare_frame = tk.Frame(sidebar, bg="#f5f5f5")
        self.compare_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        # ── Map ──────────────────────────────────────────────────────────────
        self.map_widget = tkintermapview.TkinterMapView(
            self.root,
            corner_radius=0,
            database_path=TILE_DB_PATH,   # tiles cached to disk for offline use
        )
        self.map_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.map_widget.set_position(39.5, -98.35)   # center of continental US
        self.map_widget.set_zoom(4)
        self.map_widget.add_left_click_map_command(self._on_map_click)

    def _show_about(self):
        messagebox.showinfo("About TSP Explorer", ABOUT_TEXT)

    # ── City management ───────────────────────────────────────────────────────

    def _on_map_click(self, coords):
        lat, lon = coords
        n = len(self.cities) + 1
        self._add_city({"name": f"City {n}", "lat": lat, "lon": lon})

    def _add_city(self, city: dict):
        n = len(self.cities) + 1
        label = f"{n}: {city['name']}"
        marker = self.map_widget.set_marker(city["lat"], city["lon"], text=label)
        self.cities.append(city)
        self.markers.append(marker)
        self.city_listbox.insert(tk.END, label)
        self._update_limit_note()

    def _load_cities_from_data(self, data: list, label: str):
        self.clear_cities()
        for city in data:
            self._add_city(city)
        self.map_widget.set_position(39.5, -98.35)
        self.map_widget.set_zoom(4)
        self.status_var.set(f"Loaded {label}. Select an algorithm and click Run.")

    def _read_preset_15(self) -> list | None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for fname in PRESET_15_FILES:
            path = os.path.join(script_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        return None

    def load_preset_10(self):
        data = self._read_preset_15()
        if data is None:
            messagebox.showerror("File not found",
                                 "Could not find preset_cities.json or TSP_preset_cities.py.")
            return
        self._load_cities_from_data(data[:10], "10 preset US cities")

    def load_preset_15(self):
        data = self._read_preset_15()
        if data is None:
            messagebox.showerror("File not found",
                                 "Could not find preset_cities.json or TSP_preset_cities.py.")
            return
        self._load_cities_from_data(data, "15 preset US cities")

    def load_preset_45(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, PRESET_45_FILE)
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Could not find {PRESET_45_FILE}.")
            return
        with open(path) as f:
            data = json.load(f)
        self._load_cities_from_data(data, "45 preset US cities")

    def clear_cities(self):
        self._cancel_animation()
        self._stop_timer()
        for m in self.markers:
            m.delete()
        self.markers.clear()
        self.cities.clear()
        self.city_listbox.delete(0, tk.END)
        self._clear_path()
        self._clear_compare_table()
        self.status_var.set("Ready — place cities or load a preset.")
        self.timer_var.set("—")
        self.cost_var.set("—")
        self._update_limit_note()

    # ── Path drawing ──────────────────────────────────────────────────────────

    def _clear_path(self):
        if self.path_line:
            self.path_line.delete()
            self.path_line = None

    def _draw_path(self, route: list[int]):
        self._clear_path()
        coords = [(self.cities[i]["lat"], self.cities[i]["lon"]) for i in route]
        self.path_line = self.map_widget.set_path(coords)

    # ── Algorithm limit notes ─────────────────────────────────────────────────

    def _update_limit_note(self, _event=None):
        algo = self.algo_var.get()
        n = len(self.cities)
        if algo == ALGO_BRUTE_FORCE and n > BRUTE_FORCE_MAX_CITIES:
            self._limit_note.set(
                f"⚠ Too many cities — limit is {BRUTE_FORCE_MAX_CITIES} (you have {n})."
            )
        elif algo == ALGO_HELD_KARP and n > HELD_KARP_MAX_CITIES:
            self._limit_note.set(
                f"⚠ Too many cities — limit is {HELD_KARP_MAX_CITIES} (you have {n})."
            )
        else:
            self._limit_note.set("")

    # ── Run (instant, no animation) ───────────────────────────────────────────

    def run_algorithm(self):
        self._validated_start(animated=False)

    def run_animated_algo(self):
        self._validated_start(animated=True)

    def _validated_start(self, animated: bool):
        if self.running:
            return
        n = len(self.cities)
        if n < 2:
            messagebox.showwarning("Too few cities", "Place at least 2 cities first.")
            return
        algo = self.algo_var.get()
        ok, msg = self._check_limits(algo, n)
        if not ok:
            messagebox.showwarning("City limit", msg)
            return
        self._start_run(algo, animated)

    def _check_limits(self, algo: str, n: int) -> tuple[bool, str]:
        if algo == ALGO_BRUTE_FORCE and n > BRUTE_FORCE_MAX_CITIES:
            return False, (f"Brute Force is limited to {BRUTE_FORCE_MAX_CITIES} cities "
                           f"(you have {n}). Use Nearest Neighbor or Held-Karp instead.")
        if algo == ALGO_HELD_KARP and n > HELD_KARP_MAX_CITIES:
            return False, (f"Held-Karp is limited to {HELD_KARP_MAX_CITIES} cities "
                           f"(you have {n}). Use Nearest Neighbor + 2-opt instead.")
        return True, ""

    def _start_run(self, algo: str, animated: bool):
        self._cancel_animation()
        self._clear_path()
        self._clear_compare_table()
        self.running = True
        self._set_buttons(False)
        self.status_var.set(f"Running {algo}…")
        self.cost_var.set("—")
        self.timer_var.set("—")

        distances = euclidean_distances(self.cities)
        self._elapsed_start = time.perf_counter()

        if animated:
            if algo in (ALGO_NN, ALGO_NN_2OPT):
                # Compute steps synchronously (sub-ms), show timing, then animate.
                self._run_heuristic_animated(algo, distances)
            elif algo == ALGO_BRUTE_FORCE:
                # Stream each new best-tour-found from a background thread.
                self._tick_timer()
                t = threading.Thread(target=self._bf_animated_thread,
                                     args=(distances,), daemon=True)
                t.start()
            else:  # Held-Karp
                # Compute in background, then animate final route edge-by-edge.
                self._tick_timer()
                t = threading.Thread(target=self._hk_animated_thread,
                                     args=(distances,), daemon=True)
                t.start()
        else:
            # All four algorithms: background thread, instant result, no animation.
            self._tick_timer()
            t = threading.Thread(target=self._run_in_thread,
                                 args=(algo, distances), daemon=True)
            t.start()

    def _run_in_thread(self, algo: str, distances: list):
        """Instant (non-animated) runner for all four algorithms."""
        start = time.perf_counter()
        try:
            if algo == ALGO_BRUTE_FORCE:
                cost, route = brute_force(distances)
            elif algo == ALGO_HELD_KARP:
                cost, route = held_karp(distances)
            elif algo == ALGO_NN:
                cost, route = nearest_neighbor(distances)
            else:  # NN + 2-opt
                _, nn_route = nearest_neighbor(distances)
                cost, route = two_opt(nn_route, distances)
            elapsed = time.perf_counter() - start
            self.result_queue.put(("done", algo, cost, route, elapsed))
        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def _bf_animated_thread(self, distances: list):
        """
        Brute Force animated: collects every new-best-tour snapshot at full
        computation speed, then sends them all at once so the main thread can
        replay them slowly without inflating the elapsed time.
        """
        from itertools import permutations
        n = len(distances)
        other_cities = list(range(1, n))
        best_cost = None
        best_route = None
        snapshots = []
        start = time.perf_counter()
        try:
            for ordering in permutations(other_cities):
                full_tour = [0] + list(ordering) + [0]
                cost = sum(distances[full_tour[i]][full_tour[i + 1]]
                           for i in range(len(full_tour) - 1))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_route = list(full_tour)
                    snapshots.append(best_route)
            elapsed = time.perf_counter() - start
            self.result_queue.put(("bf_animate", ALGO_BRUTE_FORCE,
                                   best_cost, best_route, elapsed, snapshots))
        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def _hk_animated_thread(self, distances: list):
        """
        Held-Karp animated: computes the optimal tour, then sends it back for
        edge-by-edge animation on the main thread.
        """
        start = time.perf_counter()
        try:
            cost, route = held_karp(distances)
            elapsed = time.perf_counter() - start
            self.result_queue.put(("hk_animate", ALGO_HELD_KARP, cost, route, elapsed))
        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def _poll_results(self):
        # Process ONE item per tick so animation frames are paced at ~50 ms each
        # rather than being drained all at once (which would skip intermediate frames).
        try:
            item = self.result_queue.get_nowait()
            tag = item[0]
            if tag == "done":
                _, algo, cost, route, elapsed = item
                self._on_run_done(algo, cost, route, elapsed)
            elif tag == "bf_animate":
                _, algo, cost, route, elapsed, snapshots = item
                self._stop_timer()
                self.timer_var.set(f"{elapsed * 1000:.3f} ms")
                self.cost_var.set(f"{cost:.4f}")
                self.status_var.set(f"{algo} — done. (Animating search…)")
                self._play_steps(snapshots, delay_ms=BF_IMPROVEMENT_DELAY_MS,
                                 on_complete=self._finish_run)
            elif tag == "hk_animate":
                _, algo, cost, route, elapsed = item
                self._stop_timer()
                self.timer_var.set(f"{elapsed * 1000:.3f} ms")
                self.cost_var.set(f"{cost:.4f}")
                self.status_var.set(f"{algo} — done. (Animating route…)")
                self._animate_edges(route, on_done=self._finish_run)
            elif tag == "run_all_done":
                self._on_run_all_done(item[1])
            elif tag == "error":
                messagebox.showerror("Error", item[1])
                self._finish_run()
        except queue.Empty:
            pass
        self.root.after(50, self._poll_results)

    def _on_run_done(self, algo: str, cost: float, route: list, elapsed: float):
        self._stop_timer()
        self._draw_path(route)
        self.timer_var.set(f"{elapsed * 1000:.3f} ms")
        self.cost_var.set(f"{cost:.4f}")
        self.status_var.set(f"{algo} — done.")
        self._finish_run()

    def _finish_run(self):
        self.running = False
        self._set_buttons(True)

    def _set_buttons(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.run_btn.config(state=state)
        self.run_all_btn.config(state=state)
        self.run_anim_btn.config(state=state)

    # ── Live stopwatch ────────────────────────────────────────────────────────

    def _tick_timer(self):
        if self.running:
            elapsed = time.perf_counter() - self._elapsed_start
            self.timer_var.set(f"{elapsed * 1000:.3f} ms")
            self._timer_job = self.root.after(50, self._tick_timer)

    def _stop_timer(self):
        if self._timer_job:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None

    # ── Animation ─────────────────────────────────────────────────────────────

    def _cancel_animation(self):
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None

    def _run_heuristic_animated(self, algo: str, distances: list):
        """Pre-compute all animation steps (fast), display timing, then play."""
        if algo == ALGO_NN:
            t0 = time.perf_counter()
            steps = list(nn_steps(distances))
            elapsed = time.perf_counter() - t0
            _, final_route = nearest_neighbor(distances)
            cost = tour_length(final_route, distances)
            self._show_result(algo, elapsed, cost)
            self._play_steps(steps)
        else:
            t0 = time.perf_counter()
            nn_step_list = list(nn_steps(distances))
            _, nn_route = nearest_neighbor(distances)
            opt_steps = list(two_opt_steps(nn_route, distances))
            elapsed = time.perf_counter() - t0
            final_route = opt_steps[-1] if opt_steps else nn_route
            cost = tour_length(final_route, distances)
            self._show_result(ALGO_NN_2OPT, elapsed, cost)
            self._play_steps(
                nn_step_list,
                on_complete=lambda: self._play_steps(opt_steps or [nn_route]),
            )

    def _show_result(self, algo: str, elapsed: float, cost: float):
        self.timer_var.set(f"{elapsed * 1000:.3f} ms")
        self.cost_var.set(f"{cost:.4f}")
        self.status_var.set(f"{algo} — done. (Animating tour…)")

    def _play_steps(self, steps: list, delay_ms: int = ANIMATION_DELAY_MS,
                    on_complete=None):
        """Animate a list of pre-computed route snapshots at delay_ms per frame."""
        if not steps:
            if on_complete:
                on_complete()
            else:
                self._clear_status_animating()
                self._finish_run()
            return

        idx = [0]

        def tick():
            if idx[0] < len(steps):
                self._draw_path(steps[idx[0]])
                idx[0] += 1
                self.animation_job = self.root.after(delay_ms, tick)
            else:
                self.animation_job = None
                if on_complete:
                    on_complete()
                else:
                    self._clear_status_animating()
                    self._finish_run()

        tick()

    def _animate_edges(self, route: list, on_done=None):
        """Draw a route one edge at a time. Used for Held-Karp post-computation animation."""
        idx = [2]  # route[0..1] = first edge

        def tick():
            if idx[0] <= len(route):
                self._draw_path(route[:idx[0]])
                idx[0] += 1
                self.animation_job = self.root.after(ANIMATION_DELAY_MS, tick)
            else:
                self.animation_job = None
                self._clear_status_animating()
                if on_done:
                    on_done()

        tick()

    def _clear_status_animating(self):
        self.status_var.set(self.status_var.get().replace(" (Animating tour…)", "")
                                                  .replace(" (Animating route…)", ""))

    # ── Run All ───────────────────────────────────────────────────────────────

    def run_all(self):
        if self.running:
            return
        n = len(self.cities)
        if n < 2:
            messagebox.showwarning("Too few cities", "Place at least 2 cities first.")
            return
        self._cancel_animation()
        self._clear_path()
        self._clear_compare_table()
        self.running = True
        self._set_buttons(False)
        self.status_var.set("Running all algorithms…")
        self.timer_var.set("—")
        self.cost_var.set("—")

        distances = euclidean_distances(self.cities)
        self._elapsed_start = time.perf_counter()
        self._tick_timer()

        t = threading.Thread(
            target=self._run_all_thread, args=(distances, n), daemon=True
        )
        t.start()

    def _run_all_thread(self, distances: list, n: int):
        algorithms = [
            ("Nearest Neighbor", lambda d: nearest_neighbor(d)),
            ("NN + 2-opt",       lambda d: two_opt(nearest_neighbor(d)[1], d)),
        ]
        if n <= HELD_KARP_MAX_CITIES:
            algorithms.append(("Held-Karp",   lambda d: held_karp(d)))
        if n <= BRUTE_FORCE_MAX_CITIES:
            algorithms.append(("Brute Force", lambda d: brute_force(d)))

        results = []
        for name, fn in algorithms:
            t0 = time.perf_counter()
            cost, route = fn(distances)
            elapsed = time.perf_counter() - t0
            results.append((name, cost, elapsed, route))

        self.result_queue.put(("run_all_done", results))

    def _on_run_all_done(self, results: list):
        total_elapsed = time.perf_counter() - self._elapsed_start
        self._stop_timer()
        self.timer_var.set(f"{total_elapsed * 1000:.3f} ms (total)")
        self.cost_var.set("—")
        self.status_var.set("Run All complete. Best tour drawn.")

        # Draw the best-cost tour on the map
        best = min(results, key=lambda r: r[1])
        self._draw_path(best[3])

        self._show_compare_table(results)
        self._finish_run()

    def _clear_compare_table(self):
        for w in self.compare_frame.winfo_children():
            w.destroy()

    def _show_compare_table(self, results: list):
        self._clear_compare_table()

        tk.Label(self.compare_frame, text="Algorithm Comparison",
                 bg="#f5f5f5", font=FONT_HEADER).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(4, 2))

        headers = ("Algorithm", "Cost", "Time (ms)")
        for col, h in enumerate(headers):
            tk.Label(self.compare_frame, text=h, bg="#e8e8e8",
                     font=FONT_BODY_B, relief=tk.FLAT,
                     padx=4).grid(row=1, column=col, sticky=tk.EW, padx=1)

        best_cost = min(r[1] for r in results)
        for row, (name, cost, elapsed, _) in enumerate(results, start=2):
            is_best = abs(cost - best_cost) < 1e-9
            fg = "#006400" if is_best else "#222"
            name_font = FONT_BODY_B if is_best else FONT_BODY
            tk.Label(self.compare_frame, text=name, bg="#f5f5f5",
                     font=name_font, fg=fg).grid(
                row=row, column=0, sticky=tk.W, padx=2)
            tk.Label(self.compare_frame, text=f"{cost:.4f}", bg="#f5f5f5",
                     font=FONT_MONO_SM, fg=fg).grid(
                row=row, column=1, sticky=tk.W, padx=4)
            tk.Label(self.compare_frame, text=f"{elapsed * 1000:.3f}", bg="#f5f5f5",
                     font=FONT_MONO_SM).grid(
                row=row, column=2, sticky=tk.W)

        for col in range(3):
            self.compare_frame.columnconfigure(col, weight=1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()
