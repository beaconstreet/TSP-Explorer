"""
Matthew Heitman - CS3C - Traveling Salesman Problem - Final Project

Unit tests for all four TSP approaches: Held-Karp, brute force, nearest
neighbor, and 2-opt.

Distances are assumed symmetric: the distance from city i to city j is
always the same as from city j to city i. All four implementations rely
on that assumption, so it carries over here.
"""

import math
import random
import unittest

from TSP_held_karp import held_karp, combinations_of_size, rebuild_route
from TSP_brute_force import brute_force, permutations_of
from TSP_nearest_neighbor import nearest_neighbor
from TSP_two_opt import two_opt, swap_improves_tour, apply_swap


def tour_length(route, distances):
    """Adds up the distance of traveling the route in order, one stop to the next.
    Written independently of the implementations being tested, so it acts as
    an honest double-check rather than reusing the same logic being tested."""
    total = 0
    for i in range(len(route) - 1):
        total += distances[route[i]][route[i + 1]]
    return total


def assert_route_is_valid(test_case, route, distances, reported_cost):
    """Shared structural checks every returned route should satisfy,
    regardless of which algorithm produced it."""
    number_of_cities = len(distances)

    test_case.assertEqual(route[0], 0, "route should start at city 0")
    test_case.assertEqual(route[-1], 0, "route should end at city 0")

    cities_visited = sorted(route[1:-1])  # everything except the start/end city 0
    test_case.assertEqual(
        cities_visited, list(range(1, number_of_cities)),
        "route should visit every other city exactly once"
    )

    test_case.assertAlmostEqual(
        tour_length(route, distances), reported_cost,
        msg="the reported cost should match the route's actual length"
    )


def euclidean_distances(coordinates):
    """Builds a distance matrix from a list of (x, y) coordinates, using
    straight-line distance between each pair."""
    number_of_cities = len(coordinates)
    distances = [[0.0] * number_of_cities for _ in range(number_of_cities)]
    for i in range(number_of_cities):
        for j in range(number_of_cities):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distances[i][j] = math.hypot(x1 - x2, y1 - y2)
    return distances


def random_symmetric_distances(number_of_cities, rng):
    """Builds a random distance matrix where distances[i][j] == distances[j][i]
    and a city's distance to itself is 0."""
    distances = [[0] * number_of_cities for _ in range(number_of_cities)]
    for i in range(number_of_cities):
        for j in range(i + 1, number_of_cities):
            distance = rng.randint(1, 100)
            distances[i][j] = distance
            distances[j][i] = distance
    return distances


class TestHeldKarp(unittest.TestCase):

    def test_four_city_example(self):
        distances = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        total_distance, route = held_karp(distances)
        self.assertEqual(total_distance, 80)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_two_city_trivial(self):
        distances = [[0, 7], [7, 0]]
        total_distance, route = held_karp(distances)
        self.assertEqual(total_distance, 14)
        self.assertEqual(route, [0, 1, 0])

    def test_all_same_location(self):
        distances = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        total_distance, route = held_karp(distances)
        self.assertEqual(total_distance, 0)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_square_layout(self):
        coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
        distances = euclidean_distances(coordinates)
        total_distance, route = held_karp(distances)
        self.assertAlmostEqual(total_distance, 4.0)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_single_city_trip(self):
        # With no other cities to visit, the trip is just staying home.
        distances = [[0]]
        total_distance, route = held_karp(distances)
        self.assertEqual(total_distance, 0)
        self.assertEqual(route, [0, 0])

    def test_zero_cities_does_not_crash(self):
        # An empty distances matrix is a meaningless input - there's no
        # city 0 to even start from - but it's worth documenting what
        # actually happens rather than assuming: the same guard clause
        # that handles "no other cities" also catches this, since it
        # never touches distances at all before returning.
        total_distance, route = held_karp([])
        self.assertEqual(total_distance, 0)
        self.assertEqual(route, [0, 0])

    def test_asymmetric_distances(self):
        # distances[i][j] doesn't have to equal distances[j][i] - Held-Karp
        # never assumes that, so this checks it holds up when the trip
        # there costs something different than the trip back.
        distances = [
            [0, 5, 12, 9],
            [8, 0, 4, 11],
            [13, 6, 0, 3],
            [7, 10, 2, 0],
        ]
        hk_cost, hk_route = held_karp(distances)
        bf_cost, _ = brute_force(distances)
        self.assertEqual(hk_cost, bf_cost)
        assert_route_is_valid(self, hk_route, distances, hk_cost)

    def test_matches_brute_force_at_larger_sizes(self):
        # The existing cross-validation only goes up to 7 cities. Brute
        # force itself is still trustworthy up to about 10, so this pushes
        # the comparison further than it's ever been checked.
        rng = random.Random(2024)
        for n in range(8, 11):
            distances = random_symmetric_distances(n, rng)
            hk_cost, hk_route = held_karp(distances)
            bf_cost, _ = brute_force(distances)
            self.assertEqual(
                hk_cost, bf_cost,
                f"Held-Karp and brute force disagree at {n} cities"
            )
            assert_route_is_valid(self, hk_route, distances, hk_cost)

    def test_combinations_of_size_returns_every_group_once(self):
        # Every group of size 2 from {1, 2, 3}, no duplicates, nothing missing.
        groups = [frozenset(g) for g in combinations_of_size([1, 2, 3], 2)]
        expected = {frozenset({1, 2}), frozenset({1, 3}), frozenset({2, 3})}

        self.assertEqual(len(groups), 3)
        self.assertEqual(set(groups), expected)

    def test_rebuild_route_walks_back_correctly(self):
        # A small, hand-built came_from table - city 3 came from city 1,
        # which came from the start. Should rebuild to [0, 1, 3, 0].
        came_from = {
            (frozenset({1}), 1): 0,
            (frozenset({1, 3}), 3): 1,
        }
        route = rebuild_route(came_from, frozenset({1, 3}), 3)
        self.assertEqual(route, [0, 1, 3, 0])

    def test_exact_route_when_answer_is_unique(self):
        # Unlike the 4-city example, this layout has exactly one optimal
        # route with no tie, so the literal route can be checked, not just
        # its cost - a stronger check on rebuild_route specifically.
        distances = [
            [0, 10, 20, 15],
            [10, 0, 12, 25],
            [20, 12, 0, 8],
            [15, 25, 8, 0],
        ]
        total_distance, route = held_karp(distances)
        self.assertEqual(total_distance, 45)
        self.assertEqual(route, [0, 3, 2, 1, 0])

    def test_ten_real_cities(self):
        # Ten real US cities (the first 10 from preset_cities.json) as a
        # concrete, reproducible example instead of random data - and not
        # a random choice of size either: 10 is brute force's own
        # practical limit, so this is the largest case where brute force
        # can still act as a trustworthy cross-check at all.
        cities = [
            (40.7128, -74.0060),   # New York, NY
            (42.3601, -71.0589),   # Boston, MA
            (39.9526, -75.1652),   # Philadelphia, PA
            (33.7490, -84.3880),   # Atlanta, GA
            (25.7617, -80.1918),   # Miami, FL
            (41.8781, -87.6298),   # Chicago, IL
            (32.7767, -96.7970),   # Dallas, TX
            (30.2672, -97.7431),   # Austin, TX
            (29.4241, -98.4936),   # San Antonio, TX
            (29.7601, -95.3701),   # Houston, TX
        ]
        distances = euclidean_distances(cities)

        hk_cost, hk_route = held_karp(distances)
        bf_cost, _ = brute_force(distances)

        self.assertAlmostEqual(hk_cost, bf_cost, places=6)
        self.assertAlmostEqual(hk_cost, 76.896701, places=5)
        self.assertEqual(hk_route, [0, 1, 4, 3, 9, 8, 7, 6, 5, 2, 0])
        assert_route_is_valid(self, hk_route, distances, hk_cost)


class TestBruteForce(unittest.TestCase):

    def test_four_city_example(self):
        distances = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        total_distance, route = brute_force(distances)
        self.assertEqual(total_distance, 80)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_two_city_trivial(self):
        distances = [[0, 7], [7, 0]]
        total_distance, route = brute_force(distances)
        self.assertEqual(total_distance, 14)
        self.assertEqual(route, [0, 1, 0])

    def test_all_same_location(self):
        distances = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        total_distance, route = brute_force(distances)
        self.assertEqual(total_distance, 0)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_square_layout(self):
        coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
        distances = euclidean_distances(coordinates)
        total_distance, route = brute_force(distances)
        self.assertAlmostEqual(total_distance, 4.0)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_single_city_trip(self):
        # With no other cities to visit, the trip is just staying home.
        distances = [[0]]
        total_distance, route = brute_force(distances)
        self.assertEqual(total_distance, 0)
        self.assertEqual(route, [0, 0])

    def test_zero_cities_raises_a_clear_error(self):
        # Unlike "one city", zero cities is a meaningless input - there's
        # no city 0 to start from at all. This documents that it's
        # expected to fail loudly rather than silently return something
        # misleading.
        with self.assertRaises(IndexError):
            brute_force([])

    def test_asymmetric_distances(self):
        # distances[i][j] doesn't have to equal distances[j][i] - brute
        # force never assumes that, since it just reads whichever
        # direction the tour happens to be traveling.
        distances = [
            [0, 5, 12, 9],
            [8, 0, 4, 11],
            [13, 6, 0, 3],
            [7, 10, 2, 0],
        ]
        bf_cost, bf_route = brute_force(distances)
        hk_cost, _ = held_karp(distances)
        self.assertEqual(bf_cost, hk_cost)
        assert_route_is_valid(self, bf_route, distances, bf_cost)

    def test_exact_route_when_answer_is_unique(self):
        # Same layout used for Held-Karp's equivalent test - both exact
        # algorithms land on the same cost, even though the specific
        # direction of the route they each return is different.
        distances = [
            [0, 10, 20, 15],
            [10, 0, 12, 25],
            [20, 12, 0, 8],
            [15, 25, 8, 0],
        ]
        total_distance, route = brute_force(distances)
        self.assertEqual(total_distance, 45)
        self.assertEqual(route, [0, 1, 2, 3, 0])

    def test_permutations_of_returns_every_ordering_once(self):
        # Every ordering of [1, 2, 3], no duplicates, nothing missing -
        # there should be exactly 3! = 6 of them.
        orderings = [tuple(o) for o in permutations_of([1, 2, 3])]
        expected = {
            (1, 2, 3), (1, 3, 2), (2, 1, 3),
            (2, 3, 1), (3, 1, 2), (3, 2, 1),
        }
        self.assertEqual(len(orderings), 6)
        self.assertEqual(set(orderings), expected)


class TestNearestNeighbor(unittest.TestCase):
    """Nearest neighbor isn't guaranteed to find the optimal tour, so these
    tests focus on structural correctness and cases simple enough to have
    an obvious right answer, rather than asserting an exact optimal cost
    in general - that's handled separately, by checking it against the
    true optimum below."""

    def test_two_city_trivial(self):
        # Only one possible tour exists, so nearest neighbor has no choice
        # to get wrong.
        distances = [[0, 7], [7, 0]]
        total_distance, route = nearest_neighbor(distances)
        self.assertEqual(total_distance, 14)
        self.assertEqual(route, [0, 1, 0])

    def test_all_same_location(self):
        distances = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        total_distance, route = nearest_neighbor(distances)
        self.assertEqual(total_distance, 0)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_returns_a_valid_route(self):
        distances = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        total_distance, route = nearest_neighbor(distances)
        assert_route_is_valid(self, route, distances, total_distance)

    def test_single_city_trip(self):
        # With no other cities to visit, the trip is just staying home.
        distances = [[0]]
        total_distance, route = nearest_neighbor(distances)
        self.assertEqual(total_distance, 0)
        self.assertEqual(route, [0, 0])

    def test_zero_cities_raises_a_clear_error(self):
        # Same reasoning as the other algorithms - zero cities means no
        # city 0 to start from, so this documents the expected failure
        # rather than assuming it works.
        with self.assertRaises(IndexError):
            nearest_neighbor([])

    def test_asymmetric_distances(self):
        # Nearest neighbor only ever looks at distances[current_city][c],
        # so the trip there and the trip back are never assumed equal.
        distances = [
            [0, 5, 12, 9],
            [8, 0, 4, 11],
            [13, 6, 0, 3],
            [7, 10, 2, 0],
        ]
        nn_cost, nn_route = nearest_neighbor(distances)
        optimal_cost, _ = held_karp(distances)
        self.assertGreaterEqual(nn_cost, optimal_cost)
        assert_route_is_valid(self, nn_route, distances, nn_cost)

    def test_greedy_choice_can_create_a_real_gap(self):
        # Cities 1 and 2 are tied for closest to the start (both 5 away),
        # so this also exercises tie-breaking - but more importantly,
        # picking either one first locks in an expensive forced return
        # later. This is a deliberately constructed adversarial example:
        # greedy genuinely loses here. The true optimum (Held-Karp, 26)
        # is well below what nearest neighbor actually finds (40).
        distances = [
            [0, 5, 5, 20],
            [5, 0, 8, 9],
            [5, 8, 0, 7],
            [20, 9, 7, 0],
        ]
        nn_cost, nn_route = nearest_neighbor(distances)
        optimal_cost, _ = held_karp(distances)

        self.assertEqual(nn_cost, 40)
        self.assertEqual(optimal_cost, 26)
        self.assertGreater(
            nn_cost, optimal_cost,
            "this layout is specifically constructed so greedy loses - "
            "if nearest neighbor matches optimal here, something changed"
        )
        assert_route_is_valid(self, nn_route, distances, nn_cost)


class TestTwoOpt(unittest.TestCase):
    """2-opt only ever improves (or leaves unchanged) whatever starting
    tour it's handed, so these tests check structural correctness plus
    that 2-opt never makes a tour worse than what it started with."""

    def test_two_city_trivial(self):
        # With only one possible tour, there's nothing for 2-opt to swap.
        distances = [[0, 7], [7, 0]]
        starting_route = [0, 1, 0]
        total_distance, route = two_opt(starting_route, distances)
        self.assertEqual(total_distance, 14)
        self.assertEqual(route, [0, 1, 0])

    def test_never_makes_a_random_tour_worse(self):
        rng = random.Random(7)

        for number_of_cities in range(4, 8):
            distances = random_symmetric_distances(number_of_cities, rng)

            starting_distance, starting_route = nearest_neighbor(distances)
            improved_distance, improved_route = two_opt(starting_route, distances)

            self.assertLessEqual(
                improved_distance, starting_distance,
                "2-opt should never produce a longer tour than its starting point"
            )
            assert_route_is_valid(self, improved_route, distances, improved_distance)

    def test_single_city_trip(self):
        # With no other cities, there's nothing to swap.
        distances = [[0]]
        total_distance, route = two_opt([0, 0], distances)
        self.assertEqual(total_distance, 0)
        self.assertEqual(route, [0, 0])

    def test_zero_cities_raises_a_clear_error(self):
        with self.assertRaises(IndexError):
            two_opt([0, 0], [])

    def test_asymmetric_distances(self):
        # 2-opt's swap check compares distances in specific directions,
        # so it should never assume the trip back costs the same as the
        # trip there.
        distances = [
            [0, 5, 12, 9],
            [8, 0, 4, 11],
            [13, 6, 0, 3],
            [7, 10, 2, 0],
        ]
        _, nn_route = nearest_neighbor(distances)
        result_cost, result_route = two_opt(nn_route, distances)
        optimal_cost, _ = held_karp(distances)
        self.assertGreaterEqual(result_cost, optimal_cost)
        assert_route_is_valid(self, result_route, distances, result_cost)

    def test_leaves_an_already_optimal_tour_unchanged(self):
        # If 2-opt is handed a tour that's already optimal, there's no
        # improving swap to find - the output should match the input
        # exactly, cost and route both.
        distances = [
            [0, 10, 20, 15],
            [10, 0, 12, 25],
            [20, 12, 0, 8],
            [15, 25, 8, 0],
        ]
        optimal_cost, optimal_route = held_karp(distances)
        result_cost, result_route = two_opt(optimal_route, distances)
        self.assertEqual(result_cost, optimal_cost)
        self.assertEqual(result_route, optimal_route)

    def test_converges_on_the_known_five_city_example(self):
        # The exact crossing tour shown in the 2-opt diagram, on the same
        # 5-city map used throughout the presentation - should converge
        # to the same answer every other algorithm found on this map,
        # 1048.0.
        coordinates = [(260, 300), (380, 100), (580, 80), (640, 280), (420, 360)]
        distances = euclidean_distances(coordinates)
        starting_route = [0, 2, 4, 1, 3, 0]

        final_cost, final_route = two_opt(starting_route, distances)

        self.assertAlmostEqual(final_cost, 1048.0, places=1)
        self.assertEqual(final_route, [0, 1, 2, 3, 4, 0])
        assert_route_is_valid(self, final_route, distances, final_cost)

    def test_swap_improves_tour_detects_a_real_improvement(self):
        # On the same crossing tour, swapping the two diagonal edges is a
        # known improvement - this checks that fact directly, instead of
        # only seeing it indirectly through the final cost.
        coordinates = [(260, 300), (380, 100), (580, 80), (640, 280), (420, 360)]
        distances = euclidean_distances(coordinates)
        route = [0, 2, 4, 1, 3, 0]
        self.assertTrue(swap_improves_tour(route, distances, 1, 2))

    def test_apply_swap_reverses_the_correct_section(self):
        # Swapping positions 1 and 3 should reverse exactly that middle
        # section and leave the rest of the route untouched.
        route = [0, 1, 2, 3, 4, 0]
        self.assertEqual(apply_swap(route, 1, 3), [0, 3, 2, 1, 4, 0])


class TestHeuristicsRespectTheOptimalBound(unittest.TestCase):
    """The exact algorithms (Held-Karp, brute force) define the best
    possible answer for a given layout. No heuristic should ever beat
    that - if one appears to, that's a sign of a bug, not a better
    algorithm."""

    def test_heuristics_never_beat_the_true_optimum(self):
        rng = random.Random(99)

        for number_of_cities in range(4, 8):
            for _ in range(5):
                distances = random_symmetric_distances(number_of_cities, rng)

                optimal_cost, _ = held_karp(distances)

                nn_cost, nn_route = nearest_neighbor(distances)
                self.assertGreaterEqual(
                    nn_cost, optimal_cost,
                    "nearest neighbor found a tour shorter than the true optimum"
                )
                assert_route_is_valid(self, nn_route, distances, nn_cost)

                two_opt_cost, two_opt_route = two_opt(nn_route, distances)
                self.assertGreaterEqual(
                    two_opt_cost, optimal_cost,
                    "2-opt found a tour shorter than the true optimum"
                )
                self.assertLessEqual(
                    two_opt_cost, nn_cost,
                    "2-opt should never make the nearest neighbor tour worse"
                )
                assert_route_is_valid(self, two_opt_route, distances, two_opt_cost)


class TestAlgorithmsAgree(unittest.TestCase):
    """Held-Karp and brute force are both exact algorithms, so they should
    always agree. This checks that agreement across many random city
    layouts, not just the ones we hand-picked above."""

    def test_random_small_layouts(self):
        rng = random.Random(42)

        for number_of_cities in range(4, 8):
            for _ in range(5):
                distances = random_symmetric_distances(number_of_cities, rng)

                held_karp_cost, held_karp_route = held_karp(distances)
                brute_force_cost, brute_force_route = brute_force(distances)

                self.assertEqual(
                    held_karp_cost, brute_force_cost,
                    f"Held-Karp and brute force disagree on a "
                    f"{number_of_cities}-city layout"
                )

                assert_route_is_valid(self, held_karp_route, distances, held_karp_cost)
                assert_route_is_valid(self, brute_force_route, distances, brute_force_cost)


if __name__ == "__main__":
    unittest.main()