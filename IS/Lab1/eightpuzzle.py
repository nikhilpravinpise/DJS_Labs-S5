from dataclasses import dataclass


GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


@dataclass
class Node:
    state: tuple[int, ...]
    parent: "Node | None"
    action: str | None
    cost: int


def find_blank(state: tuple[int, ...]) -> int:
    return state.index(0)


def swap(state: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    lst = list(state)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


def goal_test(state: tuple[int, ...]) -> bool:
    return state == GOAL


def move_gen(state: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    blank_idx = find_blank(state)
    row, col = blank_idx // 3, blank_idx % 3
    moves = []

    if row > 0:
        new_state = swap(state, blank_idx, blank_idx - 3)
        moves.append(("Up", new_state))

    if row < 2:
        new_state = swap(state, blank_idx, blank_idx + 3)
        moves.append(("Down", new_state))

    if col > 0:
        new_state = swap(state, blank_idx, blank_idx - 1)
        moves.append(("Left", new_state))

    if col < 2:
        new_state = swap(state, blank_idx, blank_idx + 1)
        moves.append(("Right", new_state))

    return moves


def print_state(state: tuple[int, ...]) -> None:
    for i in range(0, 9, 3):
        row = state[i:i+3]
        print(" ".join(str(x) if x != 0 else "_" for x in row))


if __name__ == "__main__":
    print("=== 8 Puzzle Problem ===\n")

    print("Enter initial state as 9 space-separated integers (0 for blank):")
    print("(default: 1 2 3 4 0 6 7 5 8)")
    user_input = input().strip()

    try:
        if user_input:
            initial_state = tuple(map(int, user_input.split()))
            if len(initial_state) != 9:
                raise ValueError
        else:
            initial_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    except ValueError:
        initial_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)
        print("Invalid input. Using default state.")

    print(f"\n--- Initial State ---")
    print_state(initial_state)

    print(f"\n--- Testing GoalTest ---")
    print(f"GoalTest: {goal_test(initial_state)}")

    print(f"\n--- Testing MoveGen ---")
    print(f"Possible moves:")
    for action, next_state in move_gen(initial_state):
        print(f"\n  {action}:")
        print_state(next_state)
