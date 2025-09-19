from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# Drafts of multi-criteria decision analysis based on ChatGPT.
# Based on standard MCDA techniques like TOPSIS and ELECTRE.
# See https://chatgpt.com/c/67ba7b54-04ac-800f-a3ba-4faf1855ed59


def normalize_matrix(matrix, criteria_types):
    """
    Normalize the decision matrix.

    - Uses Min-Max normalization if negative values exist.
    - Uses standard vector normalization otherwise.
    - Converts cost criteria to benefit-style scaling.
    """
    norm_matrix = np.zeros(matrix.shape)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]

        # If column has negative values, apply Min-Max normalization
        if np.any(col < 0):
            norm_matrix[:, j] = (col - np.min(col)) / (np.max(col) - np.min(col))
        else:
            # Apply vector normalization
            norm_matrix[:, j] = col / np.sqrt(np.sum(col**2))

        # If it's a cost criterion, invert the values
        if criteria_types[j] == 'cost':
            norm_matrix[:, j] = 1 - norm_matrix[:, j]

    return norm_matrix


def compute_weighted_matrix(norm_matrix, weights):
    """Apply weights to the normalized matrix."""
    weighted_matrix = norm_matrix * weights
    return weighted_matrix


def compute_concordance_matrix(weighted_matrix):
    """Compute the concordance matrix."""
    num_alternatives = weighted_matrix.shape[0]
    concordance_matrix = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                concordance_matrix[i, j] = np.sum(weighted_matrix[i] >= weighted_matrix[j])

    return concordance_matrix / weighted_matrix.shape[1]  # Normalize by number of criteria


def compute_discordance_matrix(weighted_matrix):
    """Compute the discordance matrix."""
    num_alternatives = weighted_matrix.shape[0]
    discordance_matrix = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                max_diff = np.max(np.abs(weighted_matrix[i] - weighted_matrix[j]))
                discordance_matrix[i, j] = max_diff / np.max(weighted_matrix, axis=0).max()

    return discordance_matrix


def electre_decision(concordance_matrix, discordance_matrix, c_threshold=0.5, d_threshold=0.5):
    """Determine the outranking relationships based on thresholds."""
    num_alternatives = concordance_matrix.shape[0]
    outranking_matrix = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j and concordance_matrix[i, j] >= c_threshold and discordance_matrix[i, j] <= d_threshold:
                outranking_matrix[i, j] = 1

    return outranking_matrix


def constraint_function(D):  # noqa: N803
    """Constraints: Study budget limit and dependencies between decisions."""
    budget = 100  # Example total budget
    costs = [30, 40, 50]  # Cost of each action

    # Budget constraint
    budget_constraint = budget - sum(D[i] * costs[i] for i in range(len(D)))  # Must be >= 0

    # Dependency constraints
    dependency_constraints = []
    if D[0] == 1:  # If first action is chosen, second action must also be chosen
        dependency_constraints.append(D[1] - 1)  # Must be 0 or positive
    if D[2] == 1 and D[1] == 0:  # If third action is chosen, second action must also be chosen
        dependency_constraints.append(-D[1])  # Must be 0 or positive

    # Exclusivity constraints (mutual exclusion)
    exclusivity_constraints = []
    if D[0] == 1 and D[2] == 1:  # Actions 1 and 3 cannot both be chosen
        exclusivity_constraints.append(-1)  # Must be 0 or positive (invalid case)

    # Synergy constraints (actions work better together)
    synergy_constraints = []
    if D[1] == 1 and D[2] == 1:  # If actions 2 and 3 are taken, must allocate at least 20% extra budget
        synergy_constraints.append(budget - sum(D[i] * costs[i] for i in range(len(D))) - 20)  # Must be >= 0

    return [budget_constraint] + dependency_constraints + exclusivity_constraints + synergy_constraints


# Define constraints
constraints = {'type': 'ineq', 'fun': constraint_function}

# Example Decision Matrix (Alternatives x Criteria)
decision_matrix = np.array(
    [
        [50, 2000, 100000],  # Bus
        [100, 500, 500000],  # Subway
        [20, 50, 50000],  # Bikes
    ]
)

# Criteria Weights
weights = np.array([0.4, 0.3, 0.3])

criteria_types = np.array(['cost', 'cost', 'benefit'])

# Normalize and Weight
decision_matrix_norm = normalize_matrix(decision_matrix, criteria_types)
weighted_matrix = compute_weighted_matrix(decision_matrix_norm, weights)

# Compute Concordance and Discordance Matrices
concordance_matrix = compute_concordance_matrix(weighted_matrix)
discordance_matrix = compute_discordance_matrix(weighted_matrix)

# Apply ELECTRE Decision Rule
outranking_matrix = electre_decision(concordance_matrix, discordance_matrix)

# Display Results
print('Concordance Matrix:\n', concordance_matrix)
print('Discordance Matrix:\n', discordance_matrix)
print('Outranking Matrix:\n', outranking_matrix)

# TOPSIS


def determine_ideal_solutions(weighted_matrix):
    """Determine the ideal (best) and anti-ideal (worst) solutions."""
    ideal_solution = np.max(weighted_matrix, axis=0)
    anti_ideal_solution = np.min(weighted_matrix, axis=0)
    return ideal_solution, anti_ideal_solution


def compute_distances(weighted_matrix, ideal_solution, anti_ideal_solution):
    """Compute the Euclidean distance from each alternative to the ideal and anti-ideal solutions."""
    d_plus = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted_matrix - anti_ideal_solution) ** 2, axis=1))
    return d_plus, d_minus


def compute_relative_closeness(d_plus, d_minus):
    """Compute the relative closeness to the ideal solution."""
    return d_minus / (d_plus + d_minus)


# Normalize and Weight
decision_matrix_norm = normalize_matrix(decision_matrix, criteria_types)
weighted_matrix = compute_weighted_matrix(decision_matrix_norm, weights)

# Determine Ideal and Anti-Ideal Solutions
ideal_solution, anti_ideal_solution = determine_ideal_solutions(weighted_matrix)

# Compute Distances
d_plus, d_minus = compute_distances(weighted_matrix, ideal_solution, anti_ideal_solution)

# Compute Relative Closeness Scores
relative_closeness = compute_relative_closeness(d_plus, d_minus)

# Rank Alternatives
ranking = np.argsort(relative_closeness)[::-1]  # Sort descending

# Display Results
print('Ideal Solution:', ideal_solution)
print('Anti-Ideal Solution:', anti_ideal_solution)
print('Distances to Ideal:', d_plus)
print('Distances to Anti-Ideal:', d_minus)
print('Relative Closeness Scores:', relative_closeness)
print('Ranking of Alternatives:', ranking + 1)  # Adding 1 to match human indexing

# Find interactions between actions


# Define an example objective function F(D)
def objective_function(D):  # noqa: N803
    """Complex model where F(D) naturally includes interactions."""
    D = np.array(D)
    return -(
        100 * D[0]
        + 150 * D[1]
        + 120 * D[2]  # Base contributions
        + 50 * D[0] * D[1]  # Synergy between D0 & D1
        - 40 * D[1] * D[2]  # Antagonism between D1 & D2
        + 30 * D[0] * D[2]
    )  # Weak synergy between D0 & D2


# Define constraints (budget, dependencies, etc.)
budget = 100
costs = [30, 40, 50]  # Example costs per action

# Initial decision guess
D_init = np.random.randint(0, 2, size=3)  # noqa: NPY002

# Define constraint dictionary
constraints = {'type': 'ineq', 'fun': constraint_function}

# Run full optimization
full_opt = minimize(objective_function, D_init, constraints=constraints, method='SLSQP', bounds=[(0, 1)] * 3)
F_full = -full_opt.fun
D_full = np.round(full_opt.x)

# Run single-action optimizations
F_single = []
for i in range(3):
    D_test = np.zeros(3)  # Only one action at a time
    D_test[i] = 1
    result = minimize(objective_function, D_test, constraints=constraints, method='SLSQP', bounds=[(0, 1)] * 3)
    F_single.append(-result.fun)

# Run pairwise optimizations to detect interactions
F_pairwise = np.zeros((3, 3))
for i in range(3):
    for j in range(i + 1, 3):
        D_test = np.zeros(3)
        D_test[i] = 1
        D_test[j] = 1
        result = minimize(objective_function, D_test, constraints=constraints, method='SLSQP', bounds=[(0, 1)] * 3)
        F_pairwise[i, j] = -result.fun

# Compute synergy/antagonism scores
S_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(i + 1, 3):
        S_matrix[i, j] = F_pairwise[i, j] - (F_single[i] + F_single[j])

# Print results
print('Full optimization decision:', D_full)
print('Full optimization benefit:', F_full)
print('\nIndividual benefits:', F_single)
print('\nPairwise benefits:\n', F_pairwise)
print('\nSynergy/Antagonism Matrix:\n', S_matrix)
