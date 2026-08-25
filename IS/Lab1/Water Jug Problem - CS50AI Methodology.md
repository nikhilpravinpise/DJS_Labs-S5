# Water Jug Problem Using the CS50AI Search Methodology

Source assignment: `IS Lab 1.md`, Assignment 1: Water Jug Problem.

Reference methodology: CS50AI Lecture 0 Search notes describe a search problem using an agent, states, initial state, actions, transition model, state space, goal test, path cost, nodes, frontier, and explored set.

## 1. Problem Statement

We have two jugs:

- Jug A capacity: 3 litres
- Jug B capacity: 7 litres
- Target: measure exactly 6 litres

The target can be represented as having exactly 6 litres in either jug. Since Jug A can hold only 3 litres, the practical goal is to reach 6 litres in Jug B.

## 2. CS50AI Search Representation

| CS50AI term | Water Jug meaning |
| --- | --- |
| Agent | The program that chooses jug operations |
| State | A pair `(a, b)`, where `a` is water in the 3L jug and `b` is water in the 7L jug |
| Initial State | `(0, 0)` because both jugs start empty |
| Actions | Fill, empty, or pour between jugs |
| Transition Model | The result of applying one legal action to a state |
| State Space | All legal pairs `(a, b)` where `0 <= a <= 3` and `0 <= b <= 7` |
| Goal Test | Check whether `a == 6` or `b == 6`; for this problem, only `b == 6` is possible |
| Path Cost | Number of jug operations used |
| Solution | Sequence of actions from `(0, 0)` to a goal state |

Because every action has the same cost, Breadth-First Search (BFS) is the best fit from the CS50AI notes when we want the shortest sequence of actions.

## 3. GoalTest Function

The `GoalTest` function checks whether the current state satisfies the required target.

```text
GoalTest(state, target):
    a, b = state
    if a == target or b == target:
        return true
    return false
```

For this assignment:

```text
GoalTest((a, b), 6):
    return b == 6
```

`a == 6` is impossible because Jug A has capacity 3, but keeping the general form makes the function reusable for other jug sizes.

## 4. MoveGen Function

The `MoveGen` function generates all legal successor states from the current state.

For state `(a, b)`:

- Fill Jug A: `(3, b)`
- Fill Jug B: `(a, 7)`
- Empty Jug A: `(0, b)`
- Empty Jug B: `(a, 0)`
- Pour Jug A into Jug B until Jug A is empty or Jug B is full
- Pour Jug B into Jug A until Jug B is empty or Jug A is full

```text
MoveGen(state):
    a, b = state
    successors = empty list

    add ("Fill 3L jug", (3, b))
    add ("Fill 7L jug", (a, 7))
    add ("Empty 3L jug", (0, b))
    add ("Empty 7L jug", (a, 0))

    pour = min(a, 7 - b)
    add ("Pour 3L jug into 7L jug", (a - pour, b + pour))

    pour = min(b, 3 - a)
    add ("Pour 7L jug into 3L jug", (a + pour, b - pour))

    remove duplicate states
    remove the original state if it was generated again
    return successors
```

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

Initial state:

```text
(0, 0)
```

BFS finds this shortest path:

| Step | Action | New State `(3L, 7L)` |
| --- | --- | --- |
| 0 | Start | `(0, 0)` |
| 1 | Fill 3L jug | `(3, 0)` |
| 2 | Pour 3L jug into 7L jug | `(0, 3)` |
| 3 | Fill 3L jug | `(3, 3)` |
| 4 | Pour 3L jug into 7L jug | `(0, 6)` |

Goal reached:

```text
GoalTest((0, 6), 6) = true
```

So the answer is: fill the 3L jug twice and pour it into the 7L jug each time. The 7L jug then contains exactly 6 litres.

## 7. Python Implementation

```python
from collections import deque
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


def reconstruct_path(node: Node) -> list[tuple[str, tuple[int, int]]]:
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path


def bfs() -> list[tuple[str, tuple[int, int]]] | None:
    initial_state = (0, 0)
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


solution = bfs()

if solution is None:
    print("No solution found.")
else:
    print("Initial state: (0, 0)")
    for step, (action, state) in enumerate(solution, start=1):
        print(f"{step}. {action} -> {state}")
```

Expected output:

```text
Initial state: (0, 0)
1. Fill 3L jug -> (3, 0)
2. Pour 3L jug into 7L jug -> (0, 3)
3. Fill 3L jug -> (3, 3)
4. Pour 3L jug into 7L jug -> (0, 6)
```

## 8. Final Answer for the Lab

Use the CS50AI search methodology by treating each jug configuration as a state, using `MoveGen` to generate all legal next states, using `GoalTest` to check whether 6 litres has been measured, and using BFS to find the shortest sequence of operations.

The final solution is:

```text
(0, 0) -> (3, 0) -> (0, 3) -> (3, 3) -> (0, 6)
```

At `(0, 6)`, the 7L jug contains exactly 6 litres.
