# Water Jug Problem (8L, 5L, 3L) - Code Explanation

## Overview

This Python file implements the domain-specific functions `GoalTest` and `MoveGen` for the three water jug problem using CS50AI search methodology.

---

## Line-by-Line Explanation

### imports and Constants (Lines 1-7)

```python
1: from dataclasses import dataclass
```
- Imports `dataclass` decorator from Python's `dataclasses` module
- Used to automatically generate `__init__`, `__repr__`, `__eq__` methods for the `Node` class
- Reduces boilerplate code

```python
4: CAPACITY_A = 8
5: CAPACITY_B = 5
6: CAPACITY_C = 3
7: TOTAL_WATER = 8
```
- Constants defining jug capacities:
  - Jug A: 8 liters (largest)
  - Jug B: 5 liters (medium)
  - Jug C: 3 liters (smallest)
- `TOTAL_WATER = 8` constrains maximum water in the system
- Using constants makes code readable and easy to modify

---

### Node Class (Lines 10-15)

```python
10: @dataclass
11: class Node:
```
- `@dataclass` decorator automatically creates:
  - `__init__`: initializes all fields
  - `__repr__`: string representation for debugging
  - `__eq__`: equality comparison
- Without this, you'd need to write all these methods manually

```python
12:     state: tuple[int, int, int]
```
- `state` stores the current water configuration
- Type: tuple of 3 integers representing water in Jug A, B, C
- Example: `(8, 0, 0)` means 8L in jug A, 0L in jug B, 0L in jug C
- Why tuple?
  - Immutable: cannot be accidentally modified
  - Hashable: can be added to sets for duplicate checking
  - Fixed size: always exactly 3 values

```python
13:     parent: "Node | None"
```
- Points to the previous node in the search path
- Used to reconstruct the solution path after finding the goal
- Type hint `"Node | None"` means it can be a Node or None (for the initial node)
- Quotes needed because `Node` class is being defined

```python
14:     action: str | None
```
- Stores the action taken to reach this state
- Example: `"Pour 8L jug into 5L jug"`
- `None` for the initial state (no action taken yet)

```python
15:     cost: int
```
- Path cost from initial state to this node
- For water jug problem, each move costs 1
- Used to find shortest solution in BFS

---

### GoalTest Function (Lines 18-20)

```python
18: def goal_test(state: tuple[int, int, int], target: int) -> bool:
```
- Function signature:
  - `state`: current water configuration (3 integers)
  - `target`: goal volume to measure (integer)
  - Returns: `True` if goal reached, `False` otherwise

```python
19:     a, b, c = state
```
- Unpacks the tuple into three variables
- `a` = water in 8L jug
- `b` = water in 5L jug
- `c` = water in 3L jug
- Makes code more readable than `state[0]`, `state[1]`, `state[2]`

```python
20:     return a == target or b == target or c == target
```
- Checks if any jug contains exactly the target volume
- Uses short-circuit evaluation: stops at first `True`
- Example: If target is 4, `(1, 4, 3)` returns True (jug B has 4L)

---

### MoveGen Function (Lines 23-60)

```python
23: def move_gen(state: tuple[int, int, int]) -> list[tuple[str, tuple[int, int, int]]]:
```
- Returns a list of possible moves
- Each move is a tuple: `(action_description, new_state)`
- Example: `("Pour 8L jug into 5L jug", (3, 5, 0))`

```python
24:     a, b, c = state
```
- Unpacks current state for easy access

```python
25:     moves = []
```
- Initializes empty list to store valid moves

```python
27:     candidates = []
```
- Stores all candidate moves before filtering
- Filtering removes invalid/duplicate moves

---

#### Fill Operations (Lines 29-30)

```python
29:     if a < CAPACITY_A:
```
- Only fill 8L jug if it's not already full
- Prevents generating redundant states

```python
30:         candidates.append((f"Fill {CAPACITY_A}L jug", (CAPACITY_A, b, c)))
```
- Creates fill action for 8L jug
- New state: `(8, b, c)` - jug A becomes full
- Note: We only fill the 8L jug (water source) because other jugs would exceed total water constraint

---

#### Empty Operations (Lines 32-33)

```python
32:     candidates.append((f"Empty {CAPACITY_B}L jug", (a, 0, c)))
```
- Empty 5L jug: set `b` to 0
- `a` and `c` remain unchanged

```python
33:     candidates.append((f"Empty {CAPACITY_C}L jug", (a, b, 0)))
```
- Empty 3L jug: set `c` to 0
- `a` and `b` remain unchanged

Note: We don't empty 8L jug because that would lose all water

---

#### Pour Operations (Lines 35-51)

The pour logic follows this pattern:
```python
pour = min(source_water, destination_space)
```
- `pour` = amount that can be transferred
- Takes minimum of:
  - Water available in source jug
  - Empty space in destination jug

```python
35:     pour = min(a, CAPACITY_B - b)
36:     candidates.append((f"Pour {CAPACITY_A}L jug into {CAPACITY_B}L jug", (a - pour, b + pour, c)))
```
- Pour from 8L jug to 5L jug:
  - `a` = available water in 8L jug
  - `CAPACITY_B - b` = space in 5L jug
  - Result: 8L jug loses `pour` liters, 5L jug gains `pour` liters

```python
38:     pour = min(a, CAPACITY_C - c)
39:     candidates.append((f"Pour {CAPACITY_A}L jug into {CAPACITY_C}L jug", (a - pour, b, c + pour)))
```
- Pour from 8L jug to 3L jug

```python
41:     pour = min(b, CAPACITY_A - a)
42:     candidates.append((f"Pour {CAPACITY_B}L jug into {CAPACITY_A}L jug", (a + pour, b - pour, c)))
```
- Pour from 5L jug to 8L jug
- Note: 8L jug is the main reservoir, so we can pour back into it

```python
44:     pour = min(b, CAPACITY_C - c)
45:     candidates.append((f"Pour {CAPACITY_B}L jug into {CAPACITY_C}L jug", (a, b - pour, c + pour)))
```
- Pour from 5L jug to 3L jug

```python
47:     pour = min(c, CAPACITY_A - a)
48:     candidates.append((f"Pour {CAPACITY_C}L jug into {CAPACITY_A}L jug", (a + pour, b, c - pour)))
```
- Pour from 3L jug to 8L jug

```python
50:     pour = min(c, CAPACITY_B - b)
51:     candidates.append((f"Pour {CAPACITY_C}L jug into {CAPACITY_B}L jug", (a, b + pour, c - pour)))
```
- Pour from 3L jug to 5L jug

---

#### Filtering Valid Moves (Lines 53-58)

```python
53:     seen = set()
```
- Creates a set to track already-seen states
- Sets provide O(1) lookup time for duplicates

```python
54:     for action, next_state in candidates:
```
- Iterate through all candidate moves

```python
55:         total = sum(next_state)
```
- Calculate total water in the new state
- `sum()` adds all elements of the tuple

```python
56:         if next_state != state and next_state not in seen and total <= TOTAL_WATER:
```
- Three conditions for a valid move:
  1. `next_state != state`: Must change the state (no self-loops)
  2. `next_state not in seen`: No duplicate states
  3. `total <= TOTAL_WATER`: Cannot exceed 8L total water

```python
57:             seen.add(next_state)
```
- Add state to seen set to prevent duplicates

```python
58:             moves.append((action, next_state))
```
- Add valid move to the result list

```python
60:     return moves
```
- Return list of all valid moves

---

### Main Execution Block (Lines 63-80)

```python
63: if __name__ == "__main__":
```
- Ensures code only runs when file is executed directly
- Won't run when imported as a module

```python
64:     print("=== Three Water Jug Problem (8L, 5L, 3L) ===\n")
```
- Title display

```python
66:     target = int(input("Enter target volume to measure: "))
```
- User inputs the goal volume (e.g., 4)

```python
68:     initial_a = int(input("Enter initial water in 8L jug (0-8): "))
69:     initial_b = int(input("Enter initial water in 5L jug (0-5): "))
70:     initial_c = int(input("Enter initial water in 3L jug (0-3): "))
```
- User inputs initial state for each jug
- Values constrained by jug capacities

```python
71:     initial_state = (initial_a, initial_b, initial_c)
```
- Combines inputs into a tuple representing the state

```python
73:     print(f"\n--- Testing GoalTest ---")
74:     print(f"State: {initial_state}, Target: {target} -> GoalTest: {goal_test(initial_state, target)}")
```
- Tests `goal_test` function on initial state
- Shows whether initial state is already a goal

```python
76:     print(f"\n--- Testing MoveGen ---")
77:     print(f"State: {initial_state}")
78:     print(f"\nPossible moves:")
79:     for action, next_state in move_gen(initial_state):
80:         print(f"  {action} -> {next_state} (total: {sum(next_state)}L)")
```
- Tests `move_gen` function
- Prints all possible moves from initial state
- Shows action, resulting state, and total water

---

## Key Design Decisions Summary

| Feature | Why? |
|---------|------|
| `tuple` for state | Immutable, hashable, fixed size |
| `@dataclass` for Node | Auto-generates boilerplate methods |
| `set()` for seen | O(1) duplicate checking |
| `min()` for pour | Prevents overfilling |
| Constants at top | Easy to modify, readable |
| Filter total <= 8 | Physical constraint on water |
