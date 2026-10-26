# Tic-Tac-Toe AI (Minimax with Alpha-Beta Pruning)

**Course**: Intelligent Systems Laboratory (`Course Code: DJ19DSC502`)  
**Department**: Department of Computer Science and Engineering (Data Science)  
**Experiment**: Experiment 7 - Adversarial Search  
**Roll Number**: D095  

An optimal, unbeatable Tic-Tac-Toe AI player using the Minimax algorithm with Alpha-Beta Pruning, game-theoretic analysis, and graphical interface.

---

## 🎯 Overview

This project implements an unbeatable AI agent for Tic-Tac-Toe using adversarial search. By exploring game states recursively and utilizing Alpha-Beta pruning with strategic move ordering, the agent evaluates moves in milliseconds and is mathematically guaranteed never to lose under any circumstances.

- **Detailed Lab Report & Visual Outputs**: [`IS_Lab_7_Minimax_TicTacToe.md`](IS_Lab_7_Minimax_TicTacToe.md)
- **Pre-computed Jupyter Notebook**: [`D095_IS_Lab7.ipynb`](D095_IS_Lab7.ipynb)

---

## 🧠 Algorithm & Theory

The implementation features:
- **Minimax Algorithm**: Recursively computes the optimal move by maximizing AI utility while assuming an optimal minimizing opponent.
- **Alpha-Beta Pruning**: Maintains $[\alpha, \beta]$ cutoff bounds, pruning irrelevant branches and slashing state evaluations by **98.7%** on the initial board (from 549,945 to 7,044 nodes).
- **Strategic Move Ordering**: Prioritizes Center $(1,1) \to$ Corners $\to$ Edges, maximizing alpha/beta cutoffs and reducing execution time from 2.75 seconds to ~40 ms (**67.4x speedup**).
- **Game-Theoretic Equilibrium**: Under optimal play from both players, the game value is $0$ (guaranteed tie).

---

## 📁 Repository Structure

```
IS/Lab7/
├── tictactoe.py                     # Core game logic, state transitions, Minimax & Alpha-Beta algorithms
├── runner.py                        # Interactive Pygame GUI for human-vs-AI gameplay
├── simulate_experiments.py          # Benchmark suite & high-res visualization generator
├── generate_lab7_notebook.py        # Automated notebook builder & executor
├── D095_IS_Lab7.ipynb               # Fully executed Jupyter notebook with outputs & plots
├── IS_Lab_7_Minimax_TicTacToe.md    # Comprehensive lab report with visual figures & theory
├── requirements.txt                 # Python project dependencies
├── OpenSans-Regular.ttf             # Font asset for Pygame interface
└── assets/                          # Generated visual figures & benchmark results
    ├── minimax_game_tree.png        # Minimax decision tree diagram with pruning cutoffs
    ├── benchmark_comparison.png     # Nodes explored & latency charts across plies
    ├── opening_moves_heatmap.png    # 3x3 initial move strategic heatmap
    ├── game_playthrough.png         # Visual step-by-step game progression traces
    ├── gui_simulation.png           # Pygame graphical interface screen mockups
    └── benchmark_results.json       # Raw empirical metrics & Monte Carlo results
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Play Against the AI (GUI)
```bash
python runner.py
```
- Click **Play as X** to go first or **Play as O** to let the AI move first.
- Click empty cells to place moves.

### 3. Run Benchmarks & Regenerate Visual Assets
```bash
python simulate_experiments.py
```
Runs 100 Monte Carlo games per configuration, profiles node evaluations across Plies 0 to 8, and generates all high-resolution diagrams in `assets/`.

### 4. Build & Execute Jupyter Notebook
```bash
python generate_lab7_notebook.py
```
Rebuilds and re-executes [`D095_IS_Lab7.ipynb`](D095_IS_Lab7.ipynb).

---

## 📊 Key Experimental Findings

| Metric | Standard Minimax | Alpha-Beta Pruning | Improvement |
| :--- | :--- | :--- | :--- |
| **Turn 0 Nodes Explored** | 549,945 | 7,044 | **98.7% Reduction** |
| **Turn 0 Latency** | 2,754.59 ms | 40.85 ms | **67.4x Speedup** |
| **Win/Tie Rate vs Random** | 100% (98 Wins, 2 Draws) | 100% (98 Wins, 2 Draws) | **0% Loss Rate** |
| **AI Self-Play Outcome** | 100% Draw | 100% Draw | **Nash Equilibrium** |

For complete mathematical derivations, game tree illustrations, and depth analysis, see [`IS_Lab_7_Minimax_TicTacToe.md`](IS_Lab_7_Minimax_TicTacToe.md).
