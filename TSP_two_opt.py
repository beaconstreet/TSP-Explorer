"""
Matthew Heitman - CS3C - Traveling Salesman Problem - Final Project

Traveling Salesman Problem - 2-opt Local Search

Takes an existing tour (typically from nearest neighbor) and repeatedly
looks for two edges that, if reconnected the other way, would shorten the
tour. This "uncrosses" the tour until no such improvement exists. Like
nearest neighbor, this isn't guaranteed to reach the true optimal tour -
but it reliably improves on whatever tour it starts from.
"""


def two_opt(route, distances):
    """
    route: a starting tour - a list of city numbers starting and ending at
           city 0 (e.g. whatever nearest_neighbor produced).
    distances: a 2D list where distances[i][j] is the distance from city i
               to city j.

    Returns: (total_distance, improved_route)
    """
    improved = True

    while improved:
        improved = False

        # Consider every pair of edges in the tour. Position 0 and the
        # last position are both city 0, so we only need the cities in
        # between as candidates for the section being reversed.
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if swap_improves_tour(route, distances, i, j):
                    route = apply_swap(route, i, j)
                    improved = True

    return tour_length(route, distances), route


def swap_improves_tour(route, distances, i, j):
    """
    Checks whether reversing the section of the route between position i
    and position j would shorten the tour. Reversing a section only
    changes the two edges entering and leaving it, so only those two
    edges need comparing - everything inside the section stays the same
    length, just walked in the opposite direction.
    """
    city_before_section, start_of_section = route[i - 1], route[i]
    end_of_section, city_after_section = route[j], route[j + 1]

    current_edges = (distances[city_before_section][start_of_section]
                      + distances[end_of_section][city_after_section])

    swapped_edges = (distances[city_before_section][end_of_section]
                      + distances[start_of_section][city_after_section])

    return swapped_edges < current_edges


def apply_swap(route, i, j):
    """Reverses the section of the route between position i and position j."""
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def tour_length(route, distances):
    """Adds up the distance of traveling the route in order, one stop to the next."""
    total = 0
    for k in range(len(route) - 1):
        total += distances[route[k]][route[k + 1]]
    return total


if __name__ == "__main__":
    from TSP_nearest_neighbor import nearest_neighbor

    # Same 4-city example used throughout.
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]

    starting_distance, starting_route = nearest_neighbor(distances)
    print("Starting (nearest neighbor) distance:", starting_distance)
    print("Starting route:", starting_route)

    improved_distance, improved_route = two_opt(starting_route, distances)
    print("After 2-opt distance:", improved_distance)
    print("After 2-opt route:", improved_route)