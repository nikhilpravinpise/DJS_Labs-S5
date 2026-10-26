"""
Tic Tac Toe Player

An implementation of the Minimax algorithm and Alpha-Beta Pruning for optimal
Tic-Tac-Toe gameplay, adversarial search analysis, and empirical benchmarking.
"""

import math
import time
from copy import deepcopy
from typing import Optional, Set, Tuple, List, Dict, Any, Callable

# Type alias for board representation
Board = List[List[Optional[str]]]

X = "X"
O = "O"
EMPTY = None


def initial_state() -> Board:
    """
    Returns the starting state of an empty 3x3 Tic-Tac-Toe board.
    
    Returns:
        A 3x3 2D list filled with EMPTY (None) values
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board: Board) -> str:
    """
    Determines which player has the next turn on a board.
    
    X always goes first. If X has made more moves than O, it's O's turn.
    Otherwise, it's X's turn.
    
    Args:
        board: The current game board state
        
    Returns:
        X or O, indicating whose turn it is
    """
    count_X = 0
    count_O = 0

    for row in board:
        count_X += row.count(X)
        count_O += row.count(O)

    if count_X > count_O:
        return O
    else:
        return X


def actions(board: Board) -> Set[Tuple[int, int]]:
    """
    Returns set of all possible actions (i, j) available on the board.
    
    An action is represented as a tuple (i, j) where i is the row index
    and j is the column index (both 0-indexed).
    
    Args:
        board: The current game board state
        
    Returns:
        A set of tuples representing empty positions on the board
    """
    possible_moves = set()

    for row_index, row in enumerate(board):
        for column_index, item in enumerate(row):
            if item is None:
                possible_moves.add((row_index, column_index))

    return possible_moves


def ordered_actions(board: Board) -> List[Tuple[int, int]]:
    """
    Returns actions sorted by strategic priority (Center > Corners > Edges).
    Move ordering significantly improves Alpha-Beta pruning efficiency.
    """
    moves = list(actions(board))

    def priority(m: Tuple[int, int]) -> int:
        r, c = m
        if r == 1 and c == 1:
            return 0  # Center
        if (r, c) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return 1  # Corners
        return 2  # Edges

    return sorted(moves, key=priority)


def result(board: Board, action: Tuple[int, int]) -> Board:
    """
    Returns the board that results from making move (i, j) on the board.
    
    Creates a deep copy of the board and applies the action without modifying
    the original board.
    
    Args:
        board: The current game board state
        action: A tuple (i, j) representing the move to make
        
    Returns:
        A new board state with the move applied
        
    Raises:
        Exception: If the action is invalid (cell already occupied)
    """
    player_move = player(board)

    new_board = deepcopy(board)
    i, j = action

    if board[i][j] is not None:
        raise Exception(f"Invalid action ({i}, {j}): cell already occupied")
    else:
        new_board[i][j] = player_move

    return new_board


def winner(board: Board) -> Optional[str]:
    """
    Returns the winner of the game, if there is one.
    
    Checks all rows, columns, and diagonals for three identical non-empty tokens.
    
    Args:
        board: The current game board state
        
    Returns:
        X if X has won, O if O has won, None if no winner yet
    """
    for p in (X, O):
        # Check rows
        for row in board:
            if row == [p, p, p]:
                return p

        # Check columns
        for c in range(3):
            if board[0][c] == p and board[1][c] == p and board[2][c] == p:
                return p

        # Check main diagonal
        if board[0][0] == p and board[1][1] == p and board[2][2] == p:
            return p

        # Check anti-diagonal
        if board[0][2] == p and board[1][1] == p and board[2][0] == p:
            return p

    return None


def terminal(board: Board) -> bool:
    """
    Returns True if game is over, False otherwise.
    
    The game is over if there's a winner or if the board is completely full (tie).
    
    Args:
        board: The current game board state
        
    Returns:
        True if the game has ended, False otherwise
    """
    if winner(board) is not None:
        return True

    for row in board:
        if EMPTY in row:
            return False

    return True


def utility(board: Board) -> int:
    """
    Returns the utility value of a terminal board state.
    
    Args:
        board: A terminal game board state
        
    Returns:
        1 if X has won the game, -1 if O has won, 0 for a tie
    """
    winning_player = winner(board)

    if winning_player == X:
        return 1
    elif winning_player == O:
        return -1
    else:
        return 0


def minimax_standard(board: Board) -> Optional[Tuple[int, int]]:
    """
    Standard Minimax algorithm without pruning (explores complete game tree).
    
    Args:
        board: The current game board state
        
    Returns:
        The optimal action as a tuple (i, j), or None if game is over
    """
    if terminal(board):
        return None

    def max_value(b: Board) -> Tuple[int, Tuple[int, int]]:
        if terminal(b):
            return utility(b), ()
        v = -math.inf
        best_move = ()
        for action in actions(b):
            minval = min_value(result(b, action))[0]
            if minval > v:
                v = minval
                best_move = action
        return v, best_move

    def min_value(b: Board) -> Tuple[int, Tuple[int, int]]:
        if terminal(b):
            return utility(b), ()
        v = math.inf
        best_move = ()
        for action in actions(b):
            maxval = max_value(result(b, action))[0]
            if maxval < v:
                v = maxval
                best_move = action
        return v, best_move

    curr_player = player(board)
    if curr_player == X:
        return max_value(board)[1]
    else:
        return min_value(board)[1]


def minimax_alphabeta(board: Board) -> Optional[Tuple[int, int]]:
    """
    Minimax algorithm with Alpha-Beta Pruning and strategic move ordering.
    Guarantees optimal game play while pruning unviable subtrees.
    
    Args:
        board: The current game board state
        
    Returns:
        The optimal action as a tuple (i, j), or None if game is over
    """
    if terminal(board):
        return None

    def max_value(b: Board, alpha: float, beta: float) -> int:
        if terminal(b):
            return utility(b)
        v = -math.inf
        for action in ordered_actions(b):
            v = max(v, min_value(result(b, action), alpha, beta))
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return v

    def min_value(b: Board, alpha: float, beta: float) -> int:
        if terminal(b):
            return utility(b)
        v = math.inf
        for action in ordered_actions(b):
            v = min(v, max_value(result(b, action), alpha, beta))
            beta = min(beta, v)
            if beta <= alpha:
                break
        return v

    curr_player = player(board)
    alpha = -math.inf
    beta = math.inf
    best_move = None

    if curr_player == X:
        v = -math.inf
        for action in ordered_actions(board):
            score = min_value(result(board, action), alpha, beta)
            if score > v:
                v = score
                best_move = action
            alpha = max(alpha, v)
            if v == 1:
                break  # Maximum possible score in Tic-Tac-Toe
        return best_move
    else:
        v = math.inf
        for action in ordered_actions(board):
            score = max_value(result(board, action), alpha, beta)
            if score < v:
                v = score
                best_move = action
            beta = min(beta, v)
            if v == -1:
                break  # Minimum possible score in Tic-Tac-Toe
        return best_move


# Standard entry point used by runner.py and external scripts
def minimax(board: Board) -> Optional[Tuple[int, int]]:
    """
    Default Minimax entry point. Employs Alpha-Beta Pruning for instant,
    provably optimal decision-making.
    """
    return minimax_alphabeta(board)


def minimax_profiled(board: Board, method: str = "alphabeta") -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    """
    Evaluates optimal action while profiling algorithm performance metrics
    (nodes visited, alpha/beta cuts, execution time, utility score).
    
    Args:
        board: Game board state
        method: "standard" or "alphabeta"
        
    Returns:
        (optimal_action, stats_dict)
    """
    if terminal(board):
        return None, {"nodes_explored": 0, "cuts": 0, "time_ms": 0.0, "score": utility(board)}

    nodes = 0
    cuts = 0
    t0 = time.perf_counter()

    if method == "standard":
        def max_v(b: Board) -> int:
            nonlocal nodes
            nodes += 1
            if terminal(b):
                return utility(b)
            v = -math.inf
            for a in actions(b):
                v = max(v, min_v(result(b, a)))
            return v

        def min_v(b: Board) -> int:
            nonlocal nodes
            nodes += 1
            if terminal(b):
                return utility(b)
            v = math.inf
            for a in actions(b):
                v = min(v, max_v(result(b, a)))
            return v

        curr = player(board)
        best_act = None
        if curr == X:
            best_sc = -math.inf
            for a in actions(board):
                sc = min_v(result(board, a))
                if sc > best_sc:
                    best_sc = sc
                    best_act = a
        else:
            best_sc = math.inf
            for a in actions(board):
                sc = max_v(result(board, a))
                if sc < best_sc:
                    best_sc = sc
                    best_act = a

    else:  # alphabeta
        def max_v(b: Board, a_val: float, b_val: float) -> int:
            nonlocal nodes, cuts
            nodes += 1
            if terminal(b):
                return utility(b)
            v = -math.inf
            for a in ordered_actions(b):
                v = max(v, min_v(result(b, a), a_val, b_val))
                a_val = max(a_val, v)
                if b_val <= a_val:
                    cuts += 1
                    break
            return v

        def min_v(b: Board, a_val: float, b_val: float) -> int:
            nonlocal nodes, cuts
            nodes += 1
            if terminal(b):
                return utility(b)
            v = math.inf
            for a in ordered_actions(b):
                v = min(v, max_v(result(b, a), a_val, b_val))
                b_val = min(b_val, v)
                if b_val <= a_val:
                    cuts += 1
                    break
            return v

        curr = player(board)
        alpha = -math.inf
        beta = math.inf
        best_act = None
        if curr == X:
            best_sc = -math.inf
            for a in ordered_actions(board):
                sc = min_v(result(board, a), alpha, beta)
                if sc > best_sc:
                    best_sc = sc
                    best_act = a
                alpha = max(alpha, best_sc)
                if best_sc == 1:
                    break
        else:
            best_sc = math.inf
            for a in ordered_actions(board):
                sc = max_v(result(board, a), alpha, beta)
                if sc < best_sc:
                    best_sc = sc
                    best_act = a
                beta = min(beta, best_sc)
                if best_sc == -1:
                    break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = {
        "method": method,
        "nodes_explored": nodes,
        "cuts": cuts,
        "time_ms": elapsed_ms,
        "score": best_sc
    }
    return best_act, stats


def board_to_string(board: Board) -> str:
    """
    Returns an ASCII grid representation of the board.
    """
    lines = []
    lines.append("  0   1   2")
    for i, row in enumerate(board):
        row_str = " | ".join(cell if cell is not None else " " for cell in row)
        lines.append(f"{i} {row_str}")
        if i < 2:
            lines.append("  ---+---+---")
    return "\n".join(lines)


def print_board(board: Board) -> None:
    """
    Prints an ASCII representation of the board to standard output.
    """
    print(board_to_string(board))


def play_game(
    player_X_fn: Callable[[Board], Tuple[int, int]],
    player_O_fn: Callable[[Board], Tuple[int, int]],
    verbose: bool = False
) -> Tuple[Optional[str], List[Tuple[str, Tuple[int, int], Board]]]:
    """
    Simulates a full game between two policies.
    
    Returns:
        (winner, history) where history is a list of (player, action, resulting_board)
    """
    board = initial_state()
    history = []

    if verbose:
        print("--- Start of Game ---")
        print_board(board)
        print()

    turn = 1
    while not terminal(board):
        curr = player(board)
        fn = player_X_fn if curr == X else player_O_fn
        act = fn(board)
        board = result(board, act)
        history.append((curr, act, board))

        if verbose:
            print(f"Turn {turn}: Player {curr} moves to {act}")
            print_board(board)
            print()
        turn += 1

    win = winner(board)
    if verbose:
        if win is None:
            print("Game Over: It's a Tie!")
        else:
            print(f"Game Over: Player {win} Wins!")

    return win, history
