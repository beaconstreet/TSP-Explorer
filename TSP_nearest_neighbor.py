"""
Matthew Heitman - CS3C - Traveling Salesman Problem - Final Project

Traveling Salesman Problem - Nearest Neighbor Heuristic

Builds a tour by always moving to whichever unvisited city is closest to
where you currently are. This is fast and simple, but - unlike Held-Karp
and brute force - it isn't guaranteed to find the optimal tour. It's meant
as a quick starting point that 2-opt can then improve on.
"""


def nearest_neighbor(distances):
    """
    distances: a 2D list where distances[i][j] is the distance from city i
               to city j. City 0 is treated as the fixed starting city.

    Returns: (total_distance, route)
             route is a list of city numbers in the order visited,
             starting and ending at city 0.
    """
    number_of_cities = len(distances)
    unvisited_cities = set(range(1, number_of_cities))

    current_city = 0
    route = [0]
    total_distance = 0

    while unvisited_cities:
        # Find whichever unvisited city is closest to where we are now.
        closest_city = min(unvisited_cities, key=lambda city: distances[current_city][city])

        total_distance += distances[current_city][closest_city]
        route.append(closest_city)
        unvisited_cities.remove(closest_city)
        current_city = closest_city

    # Head back to the start to complete the tour.
    total_distance += distances[current_city][0]
    route.append(0)

    return total_distance, route


if __name__ == "__main__":
    # Same 4-city example used for Held-Karp and brute force.
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]

    total_distance, route = nearest_neighbor(distances)
    print("Nearest neighbor total distance:", total_distance)
    print("Nearest neighbor route:", route)