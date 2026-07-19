from dataclasses import dataclass


CAPACITY_A = 3
CAPACITY_B = 7
TARGET = 6


@dataclass
class Node:
    state: tuple[int, int]
    parent: "Node | None"
    action: str | None
    cost: int


def goal_test(state: tuple[int, int], target: int = TARGET) -> bool:
    a, b = state
    return a == target or b == target


def move_gen(state: tuple[int, int]) -> list[tuple[str, tuple[int, int]]]:
    a, b = state
    moves = []

    candidates = [
        ("Fill 3L jug", (CAPACITY_A, b)),
        ("Fill 7L jug", (a, CAPACITY_B)),
        ("Empty 3L jug", (0, b)),
        ("Empty 7L jug", (a, 0)),
    ]

    pour = min(a, CAPACITY_B - b)
    candidates.append(("Pour 3L jug into 7L jug", (a - pour, b + pour)))

    pour = min(b, CAPACITY_A - a)
    candidates.append(("Pour 7L jug into 3L jug", (a + pour, b - pour)))

    seen = set()
    for action, next_state in candidates:
        if next_state != state and next_state not in seen:
            seen.add(next_state)
            moves.append((action, next_state))

    return moves


print("move_gen((3, 0)):")
for action, next_state in move_gen((3, 0)):
    print(f"  {action}: {next_state}")