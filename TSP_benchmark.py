"""
Matthew Heitman - CS3C - Traveling Salesman Problem - Final Project

Benchmark script - measures how each TSP algorithm's running time (and
tour quality, where comparable) changes as the number of cities grows.
Saves everything to a CSV file so it can be charted later.

Brute force and Held-Karp are exact algorithms with exponential-family
growth, so they're only run up to a city count where they still finish in
a reasonable amount of time. Nearest Neighbor and 2-opt are polynomial, so
they're pushed much further to show how differently they scale.
"""

import csv
import math
import random
import time

from TSP_held_karp import held_karp
from TSP_brute_force import brute_force
from TSP_nearest_neighbor import nearest_neighbor
from TSP_two_opt import two_opt

BRUTE_FORCE_MAX_CITIES = 10   # 11 cities already takes ~2.6s; growth is O(n!)
HELD_KARP_MAX_CITIES = 18     # 19 cities takes ~17s; growth is O(n^2 * 2^n)
HEURISTIC_EXTENDED_SIZES = [25, 50, 75, 100, 150, 200, 300]


def random_city_coordinates(number_of_cities, rng):
    """Places cities at random (x, y) points in a 1000x1000 square - a
    realistic stand-in for, say, delivery stops across a city map."""
    return [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(number_of_cities)]


def euclidean_distances(coordinates):
    """Builds a distance matrix from a list of (x, y) coordinates."""
    number_of_cities = len(coordinates)
    distances = [[0.0] * number_of_cities for _ in range(number_of_cities)]
    for i in range(number_of_cities):
        for j in range(number_of_cities):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distances[i][j] = math.hypot(x1 - x2, y1 - y2)
    return distances


def nearest_neighbor_then_two_opt(distances):
    """Runs the full heuristic pipeline: build a tour with nearest
    neighbor, then improve it with 2-opt. Returned as one combined
    algorithm so it can be timed and recorded the same way as the others."""
    _, starting_route = nearest_neighbor(distances)
    return two_opt(starting_route, distances)


def time_algorithm(algorithm, distances):
    """Runs an algorithm once and returns (seconds_elapsed, cost)."""
    start = time.perf_counter()
    cost, _ = algorithm(distances)
    elapsed = time.perf_counter() - start
    return elapsed, cost


def run_benchmarks():
    rng = random.Random(2024)
    results = []  # each entry: (city_count, algorithm_name, seconds, cost)

    # Shared sweep: every algorithm that can handle this size runs on the
    # exact same city layout, so timing AND tour quality are comparable.
    for city_count in range(4, HELD_KARP_MAX_CITIES + 1):
        distances = euclidean_distances(random_city_coordinates(city_count, rng))

        seconds, cost = time_algorithm(nearest_neighbor, distances)
        results.append((city_count, "Nearest Neighbor", seconds, cost))

        seconds, cost = time_algorithm(nearest_neighbor_then_two_opt, distances)
        results.append((city_count, "Nearest Neighbor + 2-opt", seconds, cost))

        seconds, cost = time_algorithm(held_karp, distances)
        results.append((city_count, "Held-Karp", seconds, cost))

        if city_count <= BRUTE_FORCE_MAX_CITIES:
            seconds, cost = time_algorithm(brute_force, distances)
            results.append((city_count, "Brute Force", seconds, cost))

        print(f"city_count={city_count} done")

    # Extended sweep: only the two heuristics, much further out, to show
    # how their growth compares to each other at a scale the exact
    # algorithms can never reach.
    for city_count in HEURISTIC_EXTENDED_SIZES:
        distances = euclidean_distances(random_city_coordinates(city_count, rng))

        seconds, cost = time_algorithm(nearest_neighbor, distances)
        results.append((city_count, "Nearest Neighbor", seconds, cost))

        seconds, cost = time_algorithm(nearest_neighbor_then_two_opt, distances)
        results.append((city_count, "Nearest Neighbor + 2-opt", seconds, cost))

        print(f"city_count={city_count} (extended) done")

    return results


if __name__ == "__main__":
    results = run_benchmarks()

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["city_count", "algorithm", "seconds", "tour_cost"])
        writer.writerows(results)

    print(f"\nSaved {len(results)} results to benchmark_results.csv")