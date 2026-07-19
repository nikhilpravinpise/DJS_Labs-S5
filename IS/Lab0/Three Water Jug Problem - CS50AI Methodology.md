# Three Water Jug Problem Using the CS50AI Search Methodology

Source assignment: Extended Water Jug Problem with three jugs.

Reference methodology: CS50AI Lecture 0 Search notes describe a search problem using an agent, states, initial state, actions, transition model, state space, goal test, path cost, nodes, frontier, and explored set.

## 1. Problem Statement

We have three jugs:

- Jug A capacity: 8 litres
- Jug B capacity: 5 litres
- Jug C capacity: 3 litres
- Target: measure exactly 4 litres

The target can be represented as having exactly 4 litres in any jug. Since only the 8L jug can hold 4 litres, the goal is to reach 4 litres in either the 8L or 5L jug.

## 2. CS50AI Search Representation

| CS50AI term | Water Jug meaning |
| --- | --- |
| Agent | The program that chooses jug operations |
| State | A triple `(a, b, c)`, where `a` is water in the 8L jug, `b` in the 5L jug, and `c` in the 3L jug |
| Initial State | `(8, 0, 0)` (assuming full 8L jug) or `(0, 0, 0)` (all empty) |
| Actions | Fill a jug, empty a jug, or pour from one jug to another |
| Transition Model | The result of applying one legal action to a state |
| State Space | All legal triples `(a, b, c)` where `0 <= a <= 8`, `0 <= b <= 5`, `0 <= c <= 3` |
| Goal Test | Check whether `a == 4` or `b == 4` or `c == 4` |
| Path Cost | Number of jug operations used |
| Solution | Sequence of actions from initial state to a goal state |

Because every action has the same cost, Breadth-First Search (BFS) finds the shortest sequence of actions.

## 3. GoalTest Function

The `GoalTest` function checks whether the current state satisfies the required target.

```text
GoalTest(state, target):
    a, b, c = state
    if a == target or b == target or c == target:
        return true
    return false
```

For this assignment:

```text
GoalTest((a, b, c), 4):
    return a == 4 or b == 4 or c == 4
```

Note: `c == 4` is impossible as Jug C has capacity 3.

## 4. MoveGen Function

The `MoveGen` function generates all legal successor states from the current state.

For state `(a, b, c)`:

- Fill Jug A: `(8, b, c)`
- Fill Jug B: `(a, 5, c)`
- Fill Jug C: `(a, b, 3)`
- Empty Jug A: `(0, b, c)`
- Empty Jug B: `(a, 0, c)`
- Empty Jug C: `(a, b, 0)`
- Pour A into B until A is empty or B is full
- Pour A into C until A is empty or C is full
- Pour B into A until B is empty or A is full
- Pour B into C until B is empty or C is full
- Pour C into A until C is empty or A is full
- Pour C into B until C is empty or B is full

```text
MoveGen(state):
    a, b, c = state
    successors = empty list

    add ("Fill 8L jug", (8, b, c))
    add ("Fill 5L jug", (a, 5, c))
    add ("Fill 3L jug", (a, b, 3))
    add ("Empty 8L jug", (0, b, c))
    add ("Empty 5L jug", (a, 0, c))
    add ("Empty 3L jug", (a, b, 0))

    pour = min(a, 5 - b)
    add ("Pour 8L jug into 5L jug", (a - pour, b + pour, c))

    pour = min(a, 3 - c)
    add ("Pour 8L jug into 3L jug", (a - pour, b, c + pour))

    pour = min(b, 8 - a)
    add ("Pour 5L jug into 8L jug", (a + pour, b - pour, c))

    pour = min(b, 3 - c)
    add ("Pour 5L jug into 3L jug", (a, b - pour, c + pour))

    pour = min(c, 8 - a)
    add ("Pour 3L jug into 8L jug", (a + pour, b, c - pour))

    pour = min(c, 5 - b)
    add ("Pour 3L jug into 5L jug", (a, b + pour, c - pour))

    remove duplicate states
    remove the original state if it was generated again
    return successors
```

Assume unlimited water source for filling.

## 5. BFS Algorithm in CS50AI Style

CS50AI represents each search position as a node containing:

- `state`
- `parent`
- `action`
- `path cost`

BFS stores nodes in a queue frontier. It removes the oldest node first, checks the goal, expands successors using `MoveGen`, and avoids revisiting states using an explored set.

```text
BFS(initial_state, target):
    frontier = queue containing Node(state=initial_state, parent=null, action=null)
    explored = empty set

    while frontier is not empty:
        node = frontier.remove_oldest()

        if GoalTest(node.state, target):
            return solution path by following parent links

        add node.state to explored

        for action, state in MoveGen(node.state):
            if state not in explored and state not in frontier:
                child = Node(state=state, parent=node, action=action)
                frontier.add(child)

    return no solution
```

## 6. Shortest Solution Trace

### Starting with full 8L jug: `(8, 0, 0)`

BFS finds this shortest path:

| Step | Action | New State `(8L, 5L, 3L)` |
| --- | --- | --- |
| 0 | Start | `(8, 0, 0)` |
| 1 | Pour 8L jug into 5L jug | `(3, 5, 0)` |
| 2 | Pour 5L jug into 3L jug | `(3, 2, 3)` |
| 3 | Empty 3L jug | `(3, 2, 0)` |
| 4 | Pour 5L jug into 3L jug | `(3, 0, 2)` |
| 5 | Pour 8L jug into 5L jug | `(0, 5, 2)` |
| 6 | Pour 5L jug into 3L jug | `(0, 4, 3)` |

Goal reached:

```text
GoalTest((0, 4, 3), 4) = true
```

The 5L jug now contains exactly 4 litres.

### Alternative Solution: Multiple ways exist

Another path:

| Step | Action | New State `(8L, 5L, 3L)` |
| --- | --- | --- |
| 0 | Start | `(8, 0, 0)` |
| 1 | Pour 8L jug into 5L jug | `(3, 5, 0)` |
| 2 | Pour 5L jug into 3L jug | `(3, 2, 3)` |
| 3 | Pour 3L jug into 8L jug | `(6, 2, 0)` |
| 4 | Pour 5L jug into 3L jug | `(6, 0, 2)` |
| 5 | Pour 8L jug into 5L jug | `(1, 5, 2)` |
| 6 | Pour 5L jug into 3L jug | `(1, 4, 3)` |

Goal reached:

```text
GoalTest((1, 4, 3), 4) = true
```

## 7. Python Implementation

```python
from collections import deque
from dataclasses import dataclass


CAPACITY_A = 8
CAPACITY_B = 5
CAPACITY_C = 3
TARGET = 4


@dataclass
class Node:
    state: tuple[int, int, int]
    parent: "Node | None"
    action: str | None
    cost: int


def goal_test(state: tuple[int, int, int], target: int = TARGET) -> bool:
    a, b, c = state
    return a == target or b == target or c == target


def move_gen(state: tuple[int, int, int]) -> list[tuple[str, tuple[int, int, int]]]:
    a, b, c = state
    moves = []

    candidates = [
        ("Fill 8L jug", (CAPACITY_A, b, c)),
        ("Fill 5L jug", (a, CAPACITY_B, c)),
        ("Fill 3L jug", (a, b, CAPACITY_C)),
        ("Empty 8L jug", (0, b, c)),
        ("Empty 5L jug", (a, 0, c)),
        ("Empty 3L jug", (a, b, 0)),
    ]

    pour = min(a, CAPACITY_B - b)
    candidates.append(("Pour 8L jug into 5L jug", (a - pour, b + pour, c)))

    pour = min(a, CAPACITY_C - c)
    candidates.append(("Pour 8L jug into 3L jug", (a - pour, b, c + pour)))

    pour = min(b, CAPACITY_A - a)
    candidates.append(("Pour 5L jug into 8L jug", (a + pour, b - pour, c)))

    pour = min(b, CAPACITY_C - c)
    candidates.append(("Pour 5L jug into 3L jug", (a, b - pour, c + pour)))

    pour = min(c, CAPACITY_A - a)
    candidates.append(("Pour 3L jug into 8L jug", (a + pour, b, c - pour)))

    pour = min(c, CAPACITY_B - b)
    candidates.append(("Pour 3L jug into 5L jug", (a, b + pour, c - pour)))

    seen = set()
    for action, next_state in candidates:
        if next_state != state and next_state not in seen:
            seen.add(next_state)
            moves.append((action, next_state))

    return moves


def reconstruct_path(node: Node) -> list[tuple[str, tuple[int, int, int]]]:
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path


def bfs(initial_state: tuple[int, int, int]) -> list[tuple[str, tuple[int, int, int]]] | None:
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


initial_state = (8, 0, 0)
solution = bfs(initial_state)

if solution is None:
    print("No solution found.")
else:
    print(f"Initial state: {initial_state}")
    for step, (action, state) in enumerate(solution, start=1):
        print(f"{step}. {action} -> {state}")
    print(f"\nGoal reached: 4L measured")
```

Expected output:

```text
Initial state: (8, 0, 0)
1. Pour 8L jug into 5L jug -> (3, 5, 0)
2. Pour 5L jug into 3L jug -> (3, 2, 3)
3. Empty 3L jug -> (3, 2, 0)
4. Pour 5L jug into 3L jug -> (3, 0, 2)
5. Pour 8L jug into 5L jug -> (0, 5, 2)
6. Pour 5L jug into 3L jug -> (0, 4, 3)

Goal reached: 4L measured
```

## 8. Final Answer for the Lab

Use the CS50AI search methodology by treating each jug configuration as a state, using `MoveGen` to generate all legal next states, using `GoalTest` to check whether 4 litres has been measured, and using BFS to find the shortest sequence of operations.

The final solution starting from `(8, 0, 0)`:

```text
(8, 0, 0) -> (3, 5, 0) -> (3, 2, 3) -> (3, 2, 0) -> (3, 0, 2) -> (0, 5, 2) -> (0, 4, 3)
```

At `(0, 4, 3)`, the 5L jug contains exactly 4 litres.
