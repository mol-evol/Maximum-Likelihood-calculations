import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def compute_matrices(R, Pi):
    """
    Compute the Q matrix.
    R: rate matrix
    Pi: base composition vector
    """

    # Step 1: Compute the unscaled Q matrix
    print("Step 1: Compute the unscaled Q matrix")
    unscaledQ = R * Pi
    print(f"Unscaled Q:\n{unscaledQ}\n")

    # Step 2: Normalize the Q matrix
    print("Step 2: Normalize the Q matrix")
    row_sums = np.sum(unscaledQ, axis=1, keepdims=True)
    Q0 = unscaledQ - np.diagflat(row_sums)
    print(f"Normalized Q:\n{Q0}\n")

    # Step 3: Scale the Q matrix
    print("Step 3: Scale the Q matrix")
    diagPi = np.diag(Pi)
    b = -np.sum(np.diag(diagPi @ Q0))
    scaledQ = Q0 / b
    print(f"Scaled Q:\n{scaledQ}\n")

    return scaledQ

def compute_p_matrix(scaledQ, v):
    """
    Compute the P matrix.
    scaledQ: normalized Q matrix
    v: branch length
    """
    # Step 4: Compute the P matrix
    print(f"Step 4: Compute the P matrix for branch length {v}")
    P = expm(scaledQ * v)
    print(f"P for branch length {v}:\n{P}\n")

    return P

def compute_likelihood(P, s1, s2):
    """
    Compute the likelihood of evolutionary change.
    P: P matrix
    s1, s2: sequences of DNA bases
    """
    # Step 5: Compute the likelihood of evolutionary change
    print("Step 5: Compute the likelihood of evolutionary change")
    likelihood = np.prod([P[b1, b2] for b1, b2 in zip(s1, s2)])
    print(f"Likelihood: {likelihood}\n")

    return likelihood

def plot_likelihoods(likelihoods, branch_lengths):
    """
    Plot the likelihoods versus branch lengths.
    likelihoods: list of likelihoods
    branch_lengths: list of branch lengths
    """
    plt.figure(figsize=(10, 6))
    plt.plot(branch_lengths, likelihoods, 'b-', linewidth=2, label='Sequence Likelihood')
    
    # Add labels and title
    plt.xlabel('Branch Length (substitutions per site)', fontsize=12)
    plt.ylabel('Likelihood', fontsize=12)
    plt.title('Likelihood of Sequence Evolution vs. Branch Length', fontsize=14, pad=20)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

# Define the substitution process and base composition vector
R = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])  # relative rate matrix
Pi = np.array([0.25, 0.25, 0.25, 0.25])  # base composition vector

# Compute the Q and P matrices
print("Computing the Q and P matrices...")
scaledQ = compute_matrices(R, Pi)

# Compute the likelihoods for different branch lengths
print("Computing the likelihoods for different branch lengths...")
branch_lengths = np.arange(0, 1.1, 0.1)
likelihoods = [compute_likelihood(compute_p_matrix(scaledQ, v), [0, 1, 2, 3], [1, 2, 3, 0]) for v in branch_lengths]

# Plot the likelihoods
print("Plotting the likelihoods...")
plot_likelihoods(likelihoods, branch_lengths)
