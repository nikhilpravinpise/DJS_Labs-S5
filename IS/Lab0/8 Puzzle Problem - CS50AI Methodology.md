# 8 Puzzle Problem Using the CS50AI Search Methodology

Source assignment: `IS Lab 1.md`, Assignment 3: 8 Puzzle Problem.

Reference methodology: CS50AI Lecture 0 Search notes describe a search problem using an agent, states, initial state, actions, transition model, state space, goal test, path cost, nodes, frontier, and explored set.

## 1. Problem Statement

The 8-puzzle is a sliding puzzle:
- 3x3 grid with 8 numbered tiles and 1 blank space
- Tiles adjacent to the blank can slide into the blank
- Goal: reach a target configuration from an initial configuration

Example Initial State:
```
1 2 3
4 _ 6
7 5 8
```

Goal State:
```
1 2 3
4 5 6
7 8 _
```

The blank (`_`) represents the empty space that allows tile movement.

## 2. CS50AI Search Representation

| CS50AI term | 8 Puzzle meaning |
| --- | --- |
| Agent | The program that chooses which tile to slide |
| State | A tuple of 9 elements representing the 3x3 grid (row-major order) |
| Initial State | The starting configuration of the puzzle |
| Actions | Slide a tile into the blank (Up, Down, Left, Right relative to blank) |
| Transition Model | The result of sliding a tile, swapping its position with the blank |
| State Space | All reachable permutations of 9 elements (max 9! = 362,880 states, but only half are reachable) |
| Goal Test | Check whether the current configuration matches the goal configuration |
| Path Cost | Number of moves (each move costs 1) |
| Solution | Sequence of moves from initial state to goal state |

Breadth-First Search (BFS) is optimal when each move has the same cost, guaranteeing the shortest solution.

## 3. GoalTest Function

The `GoalTest` function checks whether the current state matches the goal configuration.

```text
GoalTest(state, goal):
    if state == goal:
        return true
    return false
```

For this assignment:

```text
GoalTest(state):
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    return state == goal
```

Where `0` represents the blank space.

## 4. MoveGen Function

The `MoveGen` function generates all legal successor states by moving the blank.

For a given state, find the blank position and generate moves:

```text
MoveGen(state):
    blank_index = find position of 0 in state
    successors = empty list
    row = blank_index // 3
    col = blank_index % 3

    # Move blank Up (swap with tile above)
    if row > 0:
        new_state = swap(state, blank_index, blank_index - 3)
        add ("Up", new_state)

    # Move blank Down (swap with tile below)
    if row < 2:
        new_state = swap(state, blank_index, blank_index + 3)
        add ("Down", new_state)

    # Move blank Left (swap with tile to the left)
    if col > 0:
        new_state = swap(state, blank_index, blank_index - 1)
        add ("Left", new_state)

    # Move blank Right (swap with tile to the right)
    if col < 2:
        new_state = swap(state, blank_index, blank_index + 1)
        add ("Right", new_state)

    return successors
```

## 5. BFS Algorithm in CS50AI Style

Each search position is a node containing:

- `state`
- `parent`
- `action`
- `path cost`

BFS uses a queue frontier and an explored set to avoid revisiting states.

```text
BFS(initial_state, goal):
    frontier = queue containing Node(state=initial_state, parent=null, action=null)
    explored = empty set

    while frontier is not empty:
        node = frontier.remove_oldest()

        if GoalTest(node.state, goal):
            return solution path by following parent links

        add node.state to explored

        for action, state in MoveGen(node.state):
            if state not in explored and state not in frontier:
                child = Node(state=state, parent=node, action=action)
                frontier.add(child)

    return no solution
```

## 6. Solution Trace for Example

Initial state:

```text
(1, 2, 3, 4, 0, 6, 7, 5, 8)
```

Grid representation:
```
1 2 3
4 _ 6
7 5 8
```

BFS finds this solution path:

| Step | Action | New State (row-major) | Grid |
| --- | --- | --- | --- |
| 0 | Start | `(1,2,3,4,0,6,7,5,8)` | `1 2 3 / 4 _ 6 / 7 5 8` |
| 1 | Down | `(1,2,3,4,5,6,7,0,8)` | `1 2 3 / 4 5 6 / 7 _ 8` |
| 2 | Right | `(1,2,3,4,5,6,7,8,0)` | `1 2 3 / 4 5 6 / 7 8 _` |

Goal reached:

```text
GoalTest((1,2,3,4,5,6,7,8,0)) = true
```

The solution moves are: Down, Right (2 moves)

## 7. Python Implementation

```python
from collections import deque
from dataclasses import dataclass


GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


@dataclass
class Node:
    state: tuple[int, ...]
    parent: "Node | None"
    action: str | None
    cost: int


def goal_test(state: tuple[int, ...]) -> bool:
    return state == GOAL


def find_blank(state: tuple[int, ...]) -> int:
    return state.index(0)


def swap(state: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    lst = list(state)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


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


def reconstruct_path(node: Node) -> list[tuple[str, tuple[int, ...]]]:
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path


def bfs(initial_state: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]] | None:
    frontier = deque([Node(initial_state, None, None, 0)])
    frontier_states = {initial_state}
    explored = set()

    while frontier:
        node = frontier.popleft()
        frontier_states.remove(node.state)

        if goal_test(node.state):
            return reconstruct_path(node)

        explored.add(node.state)

        for action, state in move_gen(node.state):
            if state not in explored and state not in frontier_states:
                frontier.append(Node(state, node, action, node.cost + 1))
                frontier_states.add(state)

    return None


def print_state(state: tuple[int, ...]) -> None:
    for i in range(0, 9, 3):
        row = state[i:i+3]
        print(" ".join(str(x) if x != 0 else "_" for x in row))


initial_state = (1, 2, 3, 4, 0, 6, 7, 5, 8)

print("Initial state:")
print_state(initial_state)
print()

solution = bfs(initial_state)

if solution is None:
    print("No solution found.")
else:
    print("Solution path:")
    for step, (action, state) in enumerate(solution, start=1):
        print(f"\n{step}. Move blank {action}:")
        print_state(state)
    print(f"\nTotal moves: {len(solution)}")
```

Expected output:

```text
Initial state:
1 2 3
4 _ 6
7 5 8

Solution path:

1. Move blank Down:
1 2 3
4 5 6
7 _ 8

2. Move blank Right:
1 2 3
4 5 6
7 8 _

Total moves: 2
```

## 8. More Complex Example

For a shuffled initial state:

```
1 2 3
4 6 5
7 8 _
```

State: `(1, 2, 3, 4, 6, 5, 7, 8, 0)`

This is already at the goal.

For a harder initial state:

```
1 6 2
5 3 4
_ 7 8
```

State: `(1, 6, 2, 5, 3, 4, 0, 7, 8)`

BFS will find the optimal solution (may require many moves depending on configuration).

## 9. Final Answer for the Lab

Use the CS50AI search methodology by treating each puzzle configuration as a state, using `MoveGen` to generate all legal moves from the current blank position, using `GoalTest` to check whether the goal configuration has been reached, and using BFS to find the shortest sequence of moves.

For the example:

```text
Initial: (1, 2, 3, 4, 0, 6, 7, 5, 8)
Goal:    (1, 2, 3, 4, 5, 6, 7, 8, 0)

Moves: Down, Right
```

The blank moves Down (swapping with 5), then Right (swapping with 8) to reach the goal.
