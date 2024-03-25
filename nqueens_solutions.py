import time
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB

# Set the seed
np.random.seed(0)

def get_j_matrix(n: int) -> np.ndarray:
    """
    Get the coupling J matrix for an n x n chessboard
    """

    j_mat = np.zeros((n**2, n**2), dtype=int)
    for i in range(n**2):
        for j in range(n**2):
            j_mat[i][j] = get_cell_compatibility(n, i, j)
    
    return j_mat

def get_cell_compatibility(n: int, x: int, y: int) -> int:
    """
    Check if variables x and y are on the same row, column, or diagonal
    """

    x1, x2 = np.divmod(x, n)
    y1, y2 = np.divmod(y, n)

    if x1 == y1 and x2 == y2:
        return 0
    elif x1 == y1:
        return -1
    elif x2 == y2:
        return -1
    elif np.abs(x1 - y1) == np.abs(x2 - y2):
        return -1
    else:
        return 0

def get_qubo_terms(n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Get the terms in the QUBO mapping
    The penalty term is sum_rows (sum_i x_i - 1)^2 + sum_cols (sum_i x_i - 1)^2
    """

    STD_FACTOR = 1.0
    # Penalty factor can be zero when running Gurobi
    # Since we can add constraints explicitly
    PEN_FACTOR = 1.0

    Q_total = np.zeros((n**2, n**2))
    
    for i in range(n):
        row, col = np.zeros(n**2), np.zeros(n**2)
        row[i*n : (i + 1)*n] = 1
        for j in range(n):
            col[i + j*n] = 1

        Q_row = np.outer(row, row)
        Q_col = np.outer(col, col)
        Q_total += Q_row + Q_col

    diag_of_Q = np.diagonal(Q_total)
    np.fill_diagonal(Q_total, 0)
    Q_std = get_j_matrix(n)
    Q_pen = -2*Q_total

    q_pen = -2*np.ones(n**2) + diag_of_Q

    c_pen = 2*n

    Q = STD_FACTOR * Q_std + PEN_FACTOR * Q_pen
    q = PEN_FACTOR * q_pen
    c = PEN_FACTOR * c_pen

    return Q, q, c

def binary_to_ising(n: int, quadratic_matrix: np.ndarray, linear_vector: np.ndarray, const: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert binary QUBO mapping to Ising if necessary
    """

    one_vec = np.ones(n**2)
    Q_prime = 0.25 * quadratic_matrix
    q_prime = 0.5 * linear_vector - 0.25 * quadratic_matrix @ one_vec
    c_prime = -0.125 * one_vec.T @ quadratic_matrix @ one_vec + 0.5 * linear_vector.T @ one_vec + const

    return Q_prime, q_prime, c_prime

def block_positions(quadratic_matrix: np.ndarray, linear_vector: np.ndarray, pos_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove blocked positions from the formulation
    """

    new_matrix = np.delete(quadratic_matrix, pos_indices, 0)
    new_matrix = np.delete(new_matrix, pos_indices, 1)

    new_vector = np.delete(linear_vector, pos_indices)

    return new_matrix, new_vector

def add_positions(blocked_positions: np.ndarray, queen_vars: np.ndarray) -> np.ndarray:
    """
    Add previously blocked positions to the variable array
    """

    for index in blocked_positions:
        queen_vars = np.insert(queen_vars, index, 0)
    
    return queen_vars

def plot_chessboard(n: int, queen_vars: np.ndarray, blocking_matrix: Optional[np.ndarray] = None) -> None:
    """
    Plot the chessboard for a given configuration
    """

    board = np.reshape(queen_vars, (n, n))
    fig, ax = plt.subplots()
    im = ax.imshow(board, cmap='gray_r', interpolation=None, vmin=0, vmax=1)
    if blocking_matrix is not None:
        im = ax.imshow(blocking_matrix, cmap='Reds', interpolation=None, vmin=0, vmax=1, alpha=0.5)
    ax.hlines(np.linspace(0.5, n-1.5, num=n-1), xmin=-0.5, xmax=n-0.5, color='black')
    ax.vlines(np.linspace(0.5, n-1.5, num=n-1), ymin=-0.5, ymax=n-0.5, color='black')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    #fig.colorbar(im)
    fig.tight_layout()
    plt.show()

def check_solution_validity(n: int, queen_vars: np.ndarray) -> bool:
    """
    Check if n-queens solution is valid
    """

    j_matrix = get_j_matrix(n)
    obj_value = -0.5 * queen_vars.T @ j_matrix @ queen_vars
    n_queens = int(np.sum(queen_vars))

    if obj_value < 1e-6 and n_queens == n:
        return True
    else:
        return False

########################################
############## MAIN CODE ###############
########################################

N = 4
j_matrix, linear_field, offset = get_qubo_terms(N)
blocking_matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
#blocking_matrix = np.zeros((N, N))
blocked_positions = np.sort(np.where(np.resize(blocking_matrix, N**2) == 1)[0])
n_blocked = len(blocked_positions)
n_vars = N**2 - n_blocked
j_matrix, linear_field = block_positions(j_matrix, linear_field, blocked_positions)

# Gurobi
start_time = time.perf_counter()
m = gp.Model("model1")
#m.Params.LogToConsole = 0
m.Params.BestObjStop = 0 + 1e-4
#m.setParam('TimeLimit', 30) # 30 seconds
x = m.addMVar(shape = n_vars, vtype = GRB.BINARY, name = "x")
m.addConstr(x.sum() == N)
m.setObjective(- x @ j_matrix @ x + linear_field @ x + offset, GRB.MINIMIZE)
m.optimize()
end_time = time.perf_counter()
solution_time = end_time - start_time
print("Time to solution (TTS) =", solution_time, "[s]")

binary_output = x.X
queen_positions = add_positions(blocked_positions, binary_output)
objective_value = m.objVal

# Results
print("Queens =", queen_positions)
print("Validity =", check_solution_validity(N, queen_positions))
plot_chessboard(N, queen_positions, blocking_matrix)