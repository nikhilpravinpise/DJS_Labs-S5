from dataclasses import dataclass


DISTANCE_MATRIX = {
    'A': {'A': 0, 'B': 10, 'C': 15},
    'B': {'A': 10, 'B': 0, 'C': 35},
    'C': {'A': 15, 'B': 35, 'C': 0},
}
CITIES = {'A', 'B', 'C'}


@dataclass
class Node:
    state: tuple[str, frozenset, int]
    parent: "Node | None"
    action: str | None
    cost: int


def goal_test(state: tuple[str, frozenset, int], all_cities: set, start_city: str) -> bool:
    current, visited, _ = state
    return visited == all_cities and current == start_city


def move_gen(
    state: tuple[str, frozenset, int], distance_matrix: dict, all_cities: set, start_city: str
) -> list[tuple[str, tuple[str, frozenset, int]]]:
    current, visited, dist = state
    moves = []

    if visited == all_cities:
        return_dist = distance_matrix[current][start_city]
        new_state = (start_city, visited, dist + return_dist)
        moves.append((f"Return to {start_city}", new_state))
    else:
        for city in all_cities:
            if city not in visited:
                travel_dist = distance_matrix[current][city]
                new_visited = visited | {city}
                new_state = (city, new_visited, dist + travel_dist)
                moves.append((f"Go to {city}", new_state))

    return moves


if __name__ == "__main__":
    print("=== Travelling Salesman Problem (3 Cities) ===\n")

    start_city = input("Enter start city (A, B, or C): ").strip().upper()
    if start_city not in CITIES:
        start_city = 'A'
        print(f"Invalid input. Using default: {start_city}")

    initial_state = (start_city, frozenset({start_city}), 0)

    print(f"\n--- Testing GoalTest ---")
    print(f"State: {initial_state}, GoalTest: {goal_test(initial_state, CITIES, start_city)}")

    print(f"\n--- Testing MoveGen ---")
    print(f"State: {initial_state}")
    print(f"\nPossible moves:")
    for action, next_state in move_gen(initial_state, DISTANCE_MATRIX, CITIES, start_city):
        current, visited, total_dist = next_state
        print(f"  {action} -> ({current}, {set(visited)}, {total_dist})")
