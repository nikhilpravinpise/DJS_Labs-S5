# Travelling Salesman Problem Using the CS50AI Search Methodology

Source assignment: `IS Lab 1.md`, Assignment 2: Travelling Salesman Problem.

Reference methodology: CS50AI Lecture 0 Search notes describe a search problem using an agent, states, initial state, actions, transition model, state space, goal test, path cost, nodes, frontier, and explored set.

## 1. Problem Statement

A salesman needs to visit a set of cities:
- Visit each city exactly once
- Return to the starting city
- Minimize the total travel distance (or cost)

This is represented as finding the shortest Hamiltonian cycle through all cities.

Example with 4 cities (A, B, C, D) with the following distance matrix:

| From/To | A | B | C | D |
| --- | --- | --- | --- | --- |
| A | 0 | 10 | 15 | 20 |
| B | 10 | 0 | 35 | 25 |
| C | 15 | 35 | 0 | 30 |
| D | 20 | 25 | 30 | 0 |

## 2. CS50AI Search Representation

| CS50AI term | TSP meaning |
| --- | --- |
| Agent | The program that chooses which city to visit next |
| State | A tuple `(current_city, visited_cities, total_distance)` where `visited_cities` is a set or list of cities already visited |
| Initial State | `(start_city, {start_city}, 0)` where start_city can be any city (e.g., 'A') |
| Actions | Move to any unvisited city, or return to start if all cities visited |
| Transition Model | The result of moving from current city to a new city, updating visited set and distance |
| State Space | All permutations of city visits (n! possible tours where n = number of cities) |
| Goal Test | Check whether all cities have been visited AND the salesman has returned to the start |
| Path Cost | Sum of distances traveled (to be minimized) |
| Solution | The complete tour with minimum total distance |

Because we want to minimize distance, we can use Uniform Cost Search (UCS) or Branch and Bound. For small instances, BFS with path cost tracking works.

## 3. GoalTest Function

The `GoalTest` function checks whether all cities have been visited and the salesman is back at the start.

```text
GoalTest(state, all_cities):
    current_city, visited_cities, total_distance = state
    if visited_cities contains all cities and current_city == start_city:
        return true
    return false
```

For this assignment:

```text
GoalTest((current, visited, dist), {'A', 'B', 'C', 'D'}):
    return visited == {'A', 'B', 'C', 'D'} and current == 'A'
```

## 4. MoveGen Function

The `MoveGen` function generates all legal successor states from the current state.

For state `(current_city, visited_cities, total_distance)`:

- If all cities visited: can only return to start city
- Otherwise: can move to any unvisited city

```text
MoveGen(state, distance_matrix, all_cities, start_city):
    current, visited, dist = state
    successors = empty list

    if visited == all_cities:
        # Return to start
        return_dist = distance_matrix[current][start_city]
        add ("Return to start", (start_city, visited, dist + return_dist))
    else:
        for city in all_cities:
            if city not in visited:
                travel_dist = distance_matrix[current][city]
                new_visited = visited union {city}
                add ("Go to " + city, (city, new_visited, dist + travel_dist))

    return successors
```

## 5. BFS Algorithm in CS50AI Style

For TSP, we track each search position as a node containing:

- `state`
- `parent`
- `action`
- `path cost`

BFS explores all possible tours level by level. Since we want minimum distance, we track the best complete tour found.

```text
TSP_BFS(start_city, distance_matrix, all_cities):
    initial_state = (start_city, {start_city}, 0)
    frontier = queue containing Node(state=initial_state, parent=null, action=null)
    explored = set()  # Tracks (current, visited) pairs
    best_tour = null
    best_distance = infinity

    while frontier is not empty:
        node = frontier.remove_oldest()
        current, visited, dist = node.state

        if GoalTest(node.state, all_cities) and current == start_city:
            if dist < best_distance:
                best_tour = node
                best_distance = dist
            continue

        state_key = (current, frozenset(visited))
        if state_key in explored:
            continue
        add state_key to explored

        for action, state in MoveGen(node.state, distance_matrix, all_cities, start_city):
            child = Node(state=state, parent=node, action=action)
            frontier.add(child)

    return reconstruct_path(best_tour)
```

## 6. Optimal Solution Trace for Example

Initial state:

```text
('A', {'A'}, 0)
```

BFS explores all permutations. One optimal path:

| Step | Action | Current | Visited | Distance |
| --- | --- | --- | --- | --- |
| 0 | Start | A | {A} | 0 |
| 1 | Go to B | B | {A, B} | 10 |
| 2 | Go to D | D | {A, B, D} | 35 |
| 3 | Go to C | C | {A, B, D, C} | 65 |
| 4 | Return to A | A | {A, B, D, C} | 80 |

Total tour: A → B → D → C → A
Total distance: 10 + 25 + 30 + 15 = 80

Alternative optimal tour: A → C → D → B → A (also 80)

## 7. Python Implementation

```python
from collections import deque
from dataclasses import dataclass


DISTANCE_MATRIX = {
    'A': {'A': 0, 'B': 10, 'C': 15, 'D': 20},
    'B': {'A': 10, 'B': 0, 'C': 35, 'D': 25},
    'C': {'A': 15, 'C': 0, 'D': 30},
    'D': {'A': 20, 'B': 25, 'C': 30, 'D': 0},
}
CITIES = {'A', 'B', 'C', 'D'}
START_CITY = 'A'


@dataclass
class Node:
    state: tuple[str, frozenset, int]
    parent: "Node | None"
    action: str | None


def goal_test(state: tuple[str, frozenset, int], all_cities: set) -> bool:
    current, visited, _ = state
    return visited == all_cities and current == START_CITY


def move_gen(
    state: tuple[str, frozenset, int], distance_matrix: dict, all_cities: set
) -> list[tuple[str, tuple[str, frozenset, int]]]:
    current, visited, dist = state
    moves = []

    if visited == all_cities:
        return_dist = distance_matrix[current][START_CITY]
        new_state = (START_CITY, visited, dist + return_dist)
        moves.append((f"Return to {START_CITY}", new_state))
    else:
        for city in all_cities:
            if city not in visited:
                travel_dist = distance_matrix[current][city]
                new_visited = visited | {city}
                new_state = (city, new_visited, dist + travel_dist)
                moves.append((f"Go to {city}", new_state))

    return moves


def reconstruct_path(node: Node) -> list[tuple[str, tuple[str, frozenset, int]]]:
    path = []
    while node.parent is not None:
        path.append((node.action, node.state))
        node = node.parent
    path.reverse()
    return path


def tsp_bfs() -> Node | None:
    initial_state = (START_CITY, frozenset({START_CITY}), 0)
    frontier = deque([Node(initial_state, None, None)])
    explored = set()
    best_node = None
    best_distance = float('inf')

    while frontier:
        node = frontier.popleft()
        current, visited, dist = node.state

        if goal_test(node.state, CITIES):
            if dist < best_distance:
                best_distance = dist
                best_node = node
            continue

        state_key = (current, visited)
        if state_key in explored:
            continue
        explored.add(state_key)

        for action, state in move_gen(node.state, DISTANCE_MATRIX, CITIES):
            frontier.append(Node(state, node, action))

    return best_node


solution = tsp_bfs()

if solution is None:
    print("No solution found.")
else:
    print(f"Initial state: ({START_CITY}, {{{START_CITY}}}, 0)")
    path = reconstruct_path(solution)
    for step, (action, state) in enumerate(path, start=1):
        current, visited, dist = state
        print(f"{step}. {action} -> ({current}, {set(visited)}, {dist})")
    print(f"\nOptimal tour distance: {solution.state[2]}")
```

Expected output:

```text
Initial state: (A, {'A'}, 0)
1. Go to B -> (B, {'A', 'B'}, 10)
2. Go to D -> (D, {'A', 'B', 'D'}, 35)
3. Go to C -> (C, {'A', 'B', 'D', 'C'}, 65)
4. Return to A -> (A, {'A', 'B', 'D', 'C'}, 80)

Optimal tour distance: 80
```

## 8. Final Answer for the Lab

Use the CS50AI search methodology by treating each partial tour as a state, using `MoveGen` to generate all legal next city visits, using `GoalTest` to check whether a complete tour has been formed, and using BFS to explore all possible tours while tracking the minimum distance.

The optimal tour for the example:

```text
A → B → D → C → A
Total distance: 10 + 25 + 30 + 15 = 80
```

Or equivalently:

```text
A → C → D → B → A
Total distance: 15 + 30 + 25 + 10 = 80
```
