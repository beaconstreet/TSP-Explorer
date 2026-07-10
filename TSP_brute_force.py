"""
Matthew Heitman - CS3C - Traveling Salesman Problem - Final Project

Traveling Salesman Problem - Brute Force Algorithm

Finds the shortest possible round trip by trying every single possible
ordering of the cities and keeping whichever one comes out cheapest. This
guarantees the optimal answer, just like Held-Karp, but does it by
checking every full tour separately instead of reusing shared partial work.
"""

from itertools import permutations


def brute_force(distances):
    """
    distances: a 2D list where distances[i][j] is the distance from city i
               to city j. City 0 is treated as the fixed starting city.

    Returns: (shortest_total_distance, optimal_route)
             optimal_route is a list of city numbers in the order visited,
             starting and ending at city 0.
    """
    number_of_cities = len(distances)
    other_cities = list(range(1, number_of_cities))

    shortest_total_distance = None
    optimal_route = None

    # Try every possible order to visit the other cities in.
    for ordering in permutations(other_cities):
        full_tour = [0] + list(ordering) + [0]
        total_distance = tour_length(full_tour, distances)

        if shortest_total_distance is None or total_distance < shortest_total_distance:
            shortest_total_distance = total_distance
            optimal_route = full_tour

    return shortest_total_distance, optimal_route


def tour_length(tour, distances):
    """Adds up the distance of traveling the tour in order, one stop to the next."""
    total = 0
    for i in range(len(tour) - 1):
        total += distances[tour[i]][tour[i + 1]]
    return total


if __name__ == "__main__":
    # Same 4-city example used for Held-Karp, so the two can be compared directly.
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]

    total_distance, route = brute_force(distances)
    print("Shortest total distance:", total_distance)
    print("Optimal route:", route)
    # Expected: 80, matching Held-Karp's answer on the same example