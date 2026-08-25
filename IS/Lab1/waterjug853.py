from dataclasses import dataclass


CAPACITY_A = 8
CAPACITY_B = 5
CAPACITY_C = 3
TOTAL_WATER = 8


@dataclass
class Node:
    state: tuple[int, int, int]
    parent: "Node | None"
    action: str | None
    cost: int


def goal_test(state: tuple[int, int, int], target: int) -> bool:
    a, b, c = state
    return a == target or b == target or c == target


def move_gen(state: tuple[int, int, int]) -> list[tuple[str, tuple[int, int, int]]]:
    a, b, c = state
    moves = []

    candidates = []

    if a < CAPACITY_A:
        candidates.append((f"Fill {CAPACITY_A}L jug", (CAPACITY_A, b, c)))

    candidates.append((f"Empty {CAPACITY_B}L jug", (a, 0, c)))
    candidates.append((f"Empty {CAPACITY_C}L jug", (a, b, 0)))

    pour = min(a, CAPACITY_B - b)
    candidates.append((f"Pour {CAPACITY_A}L jug into {CAPACITY_B}L jug", (a - pour, b + pour, c)))

    pour = min(a, CAPACITY_C - c)
    candidates.append((f"Pour {CAPACITY_A}L jug into {CAPACITY_C}L jug", (a - pour, b, c + pour)))

    pour = min(b, CAPACITY_A - a)
    candidates.append((f"Pour {CAPACITY_B}L jug into {CAPACITY_A}L jug", (a + pour, b - pour, c)))

    pour = min(b, CAPACITY_C - c)
    candidates.append((f"Pour {CAPACITY_B}L jug into {CAPACITY_C}L jug", (a, b - pour, c + pour)))

    pour = min(c, CAPACITY_A - a)
    candidates.append((f"Pour {CAPACITY_C}L jug into {CAPACITY_A}L jug", (a + pour, b, c - pour)))

    pour = min(c, CAPACITY_B - b)
    candidates.append((f"Pour {CAPACITY_C}L jug into {CAPACITY_B}L jug", (a, b + pour, c - pour)))

    seen = set()
    for action, next_state in candidates:
        total = sum(next_state)
        if next_state != state and next_state not in seen and total <= TOTAL_WATER:
            seen.add(next_state)
            moves.append((action, next_state))

    return moves


if __name__ == "__main__":
    print("=== Three Water Jug Problem (8L, 5L, 3L) ===\n")

    target = int(input("Enter target volume to measure: "))

    initial_a = int(input("Enter initial water in 8L jug (0-8): "))
    initial_b = int(input("Enter initial water in 5L jug (0-5): "))
    initial_c = int(input("Enter initial water in 3L jug (0-3): "))
    initial_state = (initial_a, initial_b, initial_c)

    print(f"\n--- Testing GoalTest ---")
    print(f"State: {initial_state}, Target: {target} -> GoalTest: {goal_test(initial_state, target)}")

    print(f"\n--- Testing MoveGen ---")
    print(f"State: {initial_state}")
    print(f"\nPossible moves:")
    for action, next_state in move_gen(initial_state):
        print(f"  {action} -> {next_state} (total: {sum(next_state)}L)")
