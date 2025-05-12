import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import functools
import unittest

# Set numpy print formatting
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def display_header():
    """Display the header for the program."""
    print('''

\n\nThis script takes you through the calculations for combining a substitution process
and a composition vector in order to obtain a P matrix that can be used for estimating
the likelihood of evolutionary change from one DNA sequence to another
I provide a relative rate matrix and a composition vector and all calculations are derived from
this starting point. In practice, the \"maximum\" likelihood will be calculated through optimisation of
these matrices - the composition and the process matrices
''')

def create_rate_matrix(matrix_type="exemplar"):
    """
    Create and return a rate matrix of specified type.

    Parameters:
    ----------
    matrix_type : str
        Type of the rate matrix. Options: "exemplar", "jukes_cantor", "kimura2p", "gtr"

    Returns:
    -------
    numpy.ndarray
        The 4x4 rate matrix
    """
    if matrix_type == "exemplar":
        return np.array([0.0, 0.3, 0.4, 0.3,
                         0.3, 0.0, 0.3, 0.4,
                         0.4, 0.3, 0.0, 0.3,
                         0.3, 0.4, 0.3, 0.0]).reshape(4,4)
    elif matrix_type == "jukes_cantor":
        return np.array([0.0, 1.0, 1.0, 1.0,
                         1.0, 0.0, 1.0, 1.0,
                         1.0, 1.0, 0.0, 1.0,
                         1.0, 1.0, 1.0, 0.0]).reshape(4,4)
    elif matrix_type == "kimura2p":
        # Kimura 2-parameter model: transitions (A<->G, C<->T) have different rates than transversions
        return np.array([0.0, 1.0, 2.0, 1.0,
                         1.0, 0.0, 1.0, 2.0,
                         2.0, 1.0, 0.0, 1.0,
                         1.0, 2.0, 1.0, 0.0]).reshape(4,4)
    elif matrix_type == "hky85":
        # HKY85 model: accounts for base frequencies and differs transition/transversion rates
        return np.array([0.0, 1.0, 2.0, 1.0,
                         1.0, 0.0, 1.0, 2.0,
                         2.0, 1.0, 0.0, 1.0,
                         1.0, 2.0, 1.0, 0.0]).reshape(4,4)
    elif matrix_type == "gtr":
        # General Time Reversible: the most general model
        return np.array([0.0, 1.0, 2.0, 3.0,
                         1.0, 0.0, 4.0, 5.0,
                         2.0, 4.0, 0.0, 6.0,
                         3.0, 5.0, 6.0, 0.0]).reshape(4,4)
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

def create_composition_vector(vector_type="exemplar", custom=None):
    """
    Create and return a composition vector of specified type.

    Parameters:
    ----------
    vector_type : str
        Type of the composition vector. Options: "exemplar", "equal"
    custom : list, optional
        Custom composition vector

    Returns:
    -------
    numpy.ndarray
        The composition vector
    """
    if custom is not None:
        # Validate
        if len(custom) != 4 or abs(sum(custom) - 1.0) > 1e-10:
            raise ValueError("Custom composition vector must have 4 elements and sum to 1")
        return np.array(custom)
    elif vector_type == "exemplar":
        return np.array([0.1, 0.4, 0.2, 0.3])
    elif vector_type == "equal":
        return np.array([0.25, 0.25, 0.25, 0.25])
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")

def interactive_matrix_input():
    """
    Allow user to input custom rate matrix values.

    Returns:
    -------
    numpy.ndarray
        The 4x4 rate matrix with user-provided values
    """
    print("Enter values for the rate matrix (4x4):")
    base_names = ['A', 'C', 'G', 'T']
    R = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            if i != j:
                try:
                    R[i, j] = float(input(f"R[{base_names[i]},{base_names[j]}] (from {base_names[i]} to {base_names[j]}): "))
                except ValueError:
                    print("Invalid input. Using default value 1.0")
                    R[i, j] = 1.0

    return R

def interactive_composition_input():
    """
    Allow user to input custom composition vector values.

    Returns:
    -------
    numpy.ndarray
        The composition vector with user-provided values
    """
    print("Enter base frequencies (must sum to 1.0):")
    base_names = ['A', 'C', 'G', 'T']
    Pi = np.zeros(4)

    for i in range(4):
        try:
            Pi[i] = float(input(f"Frequency of {base_names[i]}: "))
        except ValueError:
            print(f"Invalid input. Using default value 0.25 for {base_names[i]}")
            Pi[i] = 0.25

    # Normalize to ensure sum is 1.0
    total = np.sum(Pi)
    if total != 1.0:
        print(f"Values sum to {total}, normalizing to 1.0")
        Pi = Pi / total

    return Pi

def sum_of_off_diags(matrix):
    """
    Calculate the sum of off-diagonal elements in a matrix.

    Parameters:
    ----------
    matrix : numpy.ndarray
        The input matrix

    Returns:
    -------
    float
        Sum of the off-diagonal elements
    """
    diag_sum = sum(np.diagonal(matrix))
    off_diag_sum = np.sum(matrix) - diag_sum
    return off_diag_sum

def calculate_scaled_q(r_matrix, pi_vector, explain=True):
    """
    Calculate scaled Q matrix from rate matrix R and composition vector Pi.

    Parameters:
    ----------
    r_matrix : numpy.ndarray
        The 4x4 rate matrix
    pi_vector : numpy.ndarray
        The composition vector
    explain : bool
        Whether to print explanatory text

    Returns:
    -------
    numpy.ndarray
        The scaled Q matrix
    """
    try:
        # Calculate unscaled Q
        if explain:
            print("\n\nThis is R, the relative rate matrix.")
            print("The order of the columns is A, C, G, T")
            print("and the order of the rows is A, C, G, T:\n")
            print("R =")
            print(r_matrix)

            print("\nThis is Pi, a composition vector and the order of the bases is A, C, G, T:\n")
            print("Pi = ", pi_vector)

        unscaled_q = r_matrix * pi_vector

        if explain:
            print("\n\nThis is unscaledQ (R multiplied by Pi)")
            print(unscaled_q)
            print('''
While this matrix gives some idea of the relative evolvability
of each kind of substitution, it needs some scaling:
''')

        # Make rows sum to zero
        row_sums = unscaled_q.sum(axis=1)

        if explain:
            print("These are the row sums of unscaledQ:")
            print(row_sums)

        # Set diagonals to negative of row sums
        for i in range(4):
            unscaled_q[i][i] = -row_sums[i]

        if explain:
            print("\nThe diagonal of the unscaled Q matrix now has negative of row sums (new unscaledQ):")
            print(unscaled_q)
            print("\nAs you can see, every row now sums to Zero\n")

            print("This is a diagonal matrix of the composition vector (diagPi):")

        diag_pi = np.diag(pi_vector)

        if explain:
            print(diag_pi)

        # Calculate matrix multiplication for scaling
        b = np.dot(diag_pi, unscaled_q)

        if explain:
            print("\nThis is b (b = unscaledQ * diagPi):")
            print(b)
            print("\nWe just constructed this matrix temporarily in order to get a scaling factor")

        this_sum_of_off_diags = sum_of_off_diags(b)

        if explain:
            print('''
This is the sum of the off diagonals of b,
and this sum can be used as a scaling factor
''')
            print("\t", f'{this_sum_of_off_diags:.3f}')

        scaled_q = unscaled_q / this_sum_of_off_diags

        if explain:
            print('''\nThe appropriately scaled Q matrix
is derived by dividing the unscaledQ matrix
by the sum of the off diagonals of the b matrix
i.e. scaledQ = unscaledQ / thisSumOfOffDiags\n
''')
            print(scaled_q)

            print('''
We use this scaledQ matrix in the equation: P(v) = e^Qv

The code is:
    P(BranchLengthInSubsPerSite) = linalg.expm(scaledQ * BranchLengthInSubsPerSite)''')

        return scaled_q

    except Exception as e:
        print(f"Error calculating scaled Q matrix: {e}")
        return None

@functools.lru_cache(maxsize=128)
def get_p_matrix(branch_length, q_matrix_tuple):
    """
    Calculate P matrix with caching for performance.

    Parameters:
    ----------
    branch_length : float
        Branch length in substitutions per site
    q_matrix_tuple : tuple
        Tuple representation of the Q matrix for caching

    Returns:
    -------
    numpy.ndarray
        The P matrix
    """
    q_matrix = np.array(q_matrix_tuple)  # Convert back from tuple
    return linalg.expm(q_matrix * branch_length)

def calculate_p_matrices(scaled_q, branch_lengths, explain=True):
    """
    Calculate P matrices for multiple branch lengths.

    Parameters:
    ----------
    scaled_q : numpy.ndarray
        The scaled Q matrix
    branch_lengths : list
        List of branch lengths to calculate P matrices for
    explain : bool
        Whether to print explanatory text

    Returns:
    -------
    dict
        Dictionary of P matrices keyed by branch length
    """
    p_matrices = {}

    # Convert Q matrix to tuple for caching
    q_tuple = tuple(map(tuple, scaled_q))

    for branch_length in branch_lengths:
        p_mat = get_p_matrix(branch_length, q_tuple)
        p_matrices[branch_length] = p_mat

        if explain:
            print(f"\nThis is P(v), for a branch length (v) of {branch_length} substitutions per site.")
            print(p_mat)

            # For the first branch length, explain what P matrix means in more detail
            if branch_length == branch_lengths[0]:
                print("\nInterpreting the P matrix:")
                print("The P matrix shows the probability of changing from one nucleotide to another.")
                print("For example, P[0][1] is the probability of changing from A to C.")
                print("Similarly, P[1][1] is the probability of C remaining as C (no change).")

                base_names = ['A', 'C', 'G', 'T']
                # Print a more readable version of the matrix
                print("\nP matrix in nucleotide form:")
                print("    A      C      G      T")
                for i in range(4):
                    row = f"{base_names[i]} "
                    for j in range(4):
                        row += f"{p_mat[i][j]:.4f} "
                    print(row)

            # Visualize the P matrix as a heatmap for an informative branch length
            if branch_length in [0.1, 0.2, 0.5]:
                plt.figure(figsize=(8, 6))
                bases = ['A', 'C', 'G', 'T']

                plt.imshow(p_mat, cmap='Blues', vmin=0, vmax=1)
                plt.colorbar(label='Probability')
                plt.title(f'P Matrix Visualization for Branch Length {branch_length}')

                # Add labels
                plt.xticks(np.arange(4), bases)
                plt.yticks(np.arange(4), bases)
                plt.xlabel('To')
                plt.ylabel('From')

                # Add text annotations
                for i in range(4):
                    for j in range(4):
                        text_color = 'white' if p_mat[i, j] > 0.7 else 'black'
                        plt.text(j, i, f'{p_mat[i, j]:.4f}',
                                ha='center', va='center', color=text_color)

                plt.show()

    return p_matrices

def nucleotide_to_index(nucleotide):
    """
    Convert nucleotide character to index.

    Parameters:
    ----------
    nucleotide : str
        Nucleotide character (A, C, G, or T)

    Returns:
    -------
    int
        Index (0 for A, 1 for C, 2 for G, 3 for T)
    """
    nucleotide = nucleotide.upper()
    if nucleotide == 'A':
        return 0
    elif nucleotide == 'C':
        return 1
    elif nucleotide == 'G':
        return 2
    elif nucleotide == 'T':
        return 3
    else:
        raise ValueError(f"Invalid nucleotide: {nucleotide}")

def calculate_likelihood(p_matrix, pi_vector, seq1, seq2):
    """
    Calculate the likelihood of evolving from seq1 to seq2.

    Parameters:
    ----------
    p_matrix : numpy.ndarray
        The P matrix
    pi_vector : numpy.ndarray
        The composition vector
    seq1 : str
        First sequence
    seq2 : str
        Second sequence

    Returns:
    -------
    float
        The likelihood
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length")

    likelihood = 1.0

    for i in range(len(seq1)):
        idx1 = nucleotide_to_index(seq1[i])
        idx2 = nucleotide_to_index(seq2[i])

        # Multiply by Pi for starting nucleotide and probability of change
        likelihood *= pi_vector[idx1] * p_matrix[idx1][idx2]

    return likelihood

def explain_likelihood(p_matrix, pi_vector, seq1, seq2, branch_length):
    """
    Explain the likelihood calculation for teaching purposes.

    Parameters:
    ----------
    p_matrix : numpy.ndarray
        The P matrix
    pi_vector : numpy.ndarray
        The composition vector
    seq1 : str
        First sequence
    seq2 : str
        Second sequence
    branch_length : float
        Branch length used for the P matrix

    Returns:
    -------
    float
        The calculated likelihood
    """
    print(f'''
Consider this two sequence alignment.
We will calculate the likelihood of evolving from seq1 to seq2,
using the P matrix, with a branch length of {branch_length}.

  seq1 {' '.join(seq1)}
       {' '.join(['|' if seq1[i] == seq2[i] else '+' for i in range(len(seq1))])}
  seq2 {' '.join(seq2)}
  ''')

    print(f'''
Starting with seq1, we will calculate the likelihood of evolving to seq2
on a 'tree' with a branch length of {branch_length}.
To remind ourselves what the P matrix for this branch length is
here it is:
''')
    print(p_matrix)

    print(f'''
The likelihood of evolving from seq1 to seq2 on tree with a
branch length of {branch_length}, with our specified base composition vector is:
''')

    # Build the formula
    formula = "Likelihood (Data | Model) = "
    calculation = ""

    for i in range(len(seq1)):
        idx1 = nucleotide_to_index(seq1[i])
        idx2 = nucleotide_to_index(seq2[i])

        if i > 0:
            formula += " * "
            calculation += " * "

        formula += f"Prob{seq1[i]} * P{seq1[i]}->{seq2[i]}"
        calculation += f"{pi_vector[idx1]:.3f} * {p_matrix[idx1][idx2]:.3f}"

    print(formula)
    print(f"\ni.e. \n\nL(D|M) = {calculation}")

    likelihood = calculate_likelihood(p_matrix, pi_vector, seq1, seq2)

    print("\nThat is to say, the likelihood is the product of the probabilities of")
    print("observing a particular nucleotide multiplied by the probability of change")
    print("In this case, the likelihood of the data, given the model is:\n")

    print(f"L(D) = P(D|M) = {likelihood:.7f}")
    print(f'''model:
    Alignment:
      seq1 {' '.join(seq1)}
           {' '.join(['|' if seq1[i] == seq2[i] else '+' for i in range(len(seq1))])}
      seq2 {' '.join(seq2)}
    Branch length:
      {branch_length} substitutions per site
    Base composition:''')
    print("      ", pi_vector)

    return likelihood

def plot_likelihood_vs_branch_length(scaled_q, pi_vector, seq1, seq2, branch_lengths):
    """
    Plot likelihood as a function of branch length.

    Parameters:
    ----------
    scaled_q : numpy.ndarray
        The scaled Q matrix
    pi_vector : numpy.ndarray
        The composition vector
    seq1 : str
        First sequence
    seq2 : str
        Second sequence
    branch_lengths : list or numpy.ndarray
        Range of branch lengths to plot

    Returns:
    -------
    list
        List of calculated likelihoods
    """
    q_tuple = tuple(map(tuple, scaled_q))
    likelihoods = []

    for brlen in branch_lengths:
        p_mat = get_p_matrix(brlen, q_tuple)
        like = calculate_likelihood(p_mat, pi_vector, seq1, seq2)
        likelihoods.append(like)

    plt.figure(figsize=(10, 6))
    plt.plot(branch_lengths, likelihoods)
    plt.xlabel('Branch Length (substitutions/site)')
    plt.ylabel('Likelihood')
    plt.title('Likelihood vs Branch Length')
    plt.grid(True)

    # Find and mark the maximum likelihood
    max_likelihood_idx = np.argmax(likelihoods)
    max_likelihood = likelihoods[max_likelihood_idx]
    max_brlen = branch_lengths[max_likelihood_idx]

    plt.scatter([max_brlen], [max_likelihood], color='red', s=100)
    plt.annotate(f'Maximum: {max_likelihood:.7f} at {max_brlen:.4f}',
                xy=(max_brlen, max_likelihood),
                xytext=(max_brlen+0.05, max_likelihood),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Add a log-scale subplot for better visualization
    plt.figure(figsize=(10, 6))
    plt.semilogy(branch_lengths, likelihoods)
    plt.xlabel('Branch Length (substitutions/site)')
    plt.ylabel('Likelihood (log scale)')
    plt.title('Likelihood vs Branch Length (Log Scale)')
    plt.grid(True)

    # Mark the maximum likelihood on log plot too
    plt.scatter([max_brlen], [max_likelihood], color='red', s=100)
    plt.annotate(f'Maximum: {max_likelihood:.7f} at {max_brlen:.4f}',
                xy=(max_brlen, max_likelihood),
                xytext=(max_brlen+0.05, max_likelihood),
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()

    # Create a visualization of the rate matrix
    plt.figure(figsize=(8, 6))
    bases = ['A', 'C', 'G', 'T']

    # Plot the rate matrix as a heatmap
    plt.imshow(scaled_q, cmap='viridis')
    plt.colorbar(label='Rate')
    plt.title('Scaled Q Matrix Visualization')

    # Add labels
    plt.xticks(np.arange(4), bases)
    plt.yticks(np.arange(4), bases)
    plt.xlabel('To')
    plt.ylabel('From')

    # Add text annotations
    for i in range(4):
        for j in range(4):
            text_color = 'white' if abs(scaled_q[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{scaled_q[i, j]:.3f}',
                     ha='center', va='center', color=text_color)

    plt.show()

    return likelihoods

def display_footer():
    """Display the footer for the program."""
    print('''Likelihood of an alignment ''')

class TestDNALikelihoodCalculator(unittest.TestCase):
    """Unit tests for the DNA likelihood calculator functions."""

    def test_sum_of_off_diags(self):
        """Test the sum_of_off_diags function."""
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = 2 + 3 + 4 + 6 + 7 + 8  # Sum of off-diagonals
        result = sum_of_off_diags(test_matrix)
        self.assertEqual(result, expected)

    def test_nucleotide_to_index(self):
        """Test the nucleotide_to_index function."""
        self.assertEqual(nucleotide_to_index('A'), 0)
        self.assertEqual(nucleotide_to_index('C'), 1)
        self.assertEqual(nucleotide_to_index('G'), 2)
        self.assertEqual(nucleotide_to_index('T'), 3)

        # Test case insensitivity
        self.assertEqual(nucleotide_to_index('a'), 0)

        # Test error handling
        with self.assertRaises(ValueError):
            nucleotide_to_index('X')

    def test_create_rate_matrix(self):
        """Test the create_rate_matrix function."""
        # Test exemplar matrix
        exemplar = create_rate_matrix("exemplar")
        self.assertEqual(exemplar.shape, (4, 4))

        # Test Jukes-Cantor matrix
        jc = create_rate_matrix("jukes_cantor")
        self.assertEqual(jc.shape, (4, 4))
        self.assertEqual(jc[0][1], 1.0)

        # Test error handling
        with self.assertRaises(ValueError):
            create_rate_matrix("unknown_matrix")

    def test_create_composition_vector(self):
        """Test the create_composition_vector function."""
        # Test exemplar vector
        exemplar = create_composition_vector("exemplar")
        self.assertEqual(len(exemplar), 4)
        self.assertAlmostEqual(np.sum(exemplar), 1.0)

        # Test equal frequencies vector
        equal = create_composition_vector("equal")
        self.assertEqual(len(equal), 4)
        self.assertAlmostEqual(np.sum(equal), 1.0)
        self.assertAlmostEqual(equal[0], 0.25)

        # Test custom vector
        custom = create_composition_vector(custom=[0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(custom), 4)
        self.assertAlmostEqual(np.sum(custom), 1.0)

        # Test error handling
        with self.assertRaises(ValueError):
            create_composition_vector(custom=[0.1, 0.2, 0.3])

        with self.assertRaises(ValueError):
            create_composition_vector(custom=[0.1, 0.2, 0.3, 0.5])  # Sum > 1.0

    def test_calculate_likelihood(self):
        """Test the calculate_likelihood function."""
        # Create a simple test case
        p_matrix = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97]
        ])
        pi_vector = np.array([0.25, 0.25, 0.25, 0.25])

        # Test with identical sequences
        likelihood = calculate_likelihood(p_matrix, pi_vector, "ACGT", "ACGT")
        expected = 0.25 * 0.97 * 0.25 * 0.97 * 0.25 * 0.97 * 0.25 * 0.97
        self.assertAlmostEqual(likelihood, expected)

        # Test with one difference
        likelihood = calculate_likelihood(p_matrix, pi_vector, "ACGT", "ACTT")
        expected = 0.25 * 0.97 * 0.25 * 0.97 * 0.25 * 0.01 * 0.25 * 0.97
        self.assertAlmostEqual(likelihood, expected)

        # Test error handling
        with self.assertRaises(ValueError):
            calculate_likelihood(p_matrix, pi_vector, "ACGT", "ACG")

def run_example():
    """Run an example calculation with explanation."""
    display_header()

    # 1. Create rate matrix and composition vector
    r = create_rate_matrix("exemplar")
    pi = create_composition_vector("exemplar")

    # 2. Calculate scaled Q matrix
    scaled_q = calculate_scaled_q(r, pi)

    # 3. Calculate P matrices for different branch lengths
    branch_lengths = [0.02, 0.1, 0.2, 20.0]
    p_matrices = calculate_p_matrices(scaled_q, branch_lengths)

    # 4. Calculate and explain likelihood for example sequences
    seq1 = "CCAT"
    seq2 = "CCGT"

    # 5. Explain likelihood calculation for branch length 0.02
    explain_likelihood(p_matrices[0.02], pi, seq1, seq2, 0.02)

    # 6. Explain likelihood calculation for branch length 0.1
    explain_likelihood(p_matrices[0.1], pi, seq1, seq2, 0.1)

    # 7. Plot likelihood vs branch length
    print("\nPlotting likelihood vs branch length...")
    branch_lengths_fine = np.arange(0.001, 1.0, 0.01)
    plot_likelihood_vs_branch_length(scaled_q, pi, seq1, seq2, branch_lengths_fine)

    # 8. Show optimal branch length in more detail
    print("\nExamining the change in likelihood near the optimal branch length:")

    # First, find an approximate optimal branch length using a coarse grid
    coarse_range = np.arange(0.1, 0.7, 0.01)
    coarse_likelihoods = []

    for brlen in coarse_range:
        p_mat = linalg.expm(scaled_q * brlen)
        like = calculate_likelihood(p_mat, pi, seq1, seq2)
        coarse_likelihoods.append(like)

    # Find the approximate optimal point
    approx_max_idx = np.argmax(coarse_likelihoods)
    approx_optimal = coarse_range[approx_max_idx]

    # Create a finer grid around the approximate optimal point
    optimal_range = np.arange(max(0.001, approx_optimal - 0.05),
                             approx_optimal + 0.05, 0.001)
    likelihoods = []

    # Calculate likelihoods for the fine grid
    for pwr in optimal_range:
        p_mat = linalg.expm(scaled_q * pwr)
        like_test = calculate_likelihood(p_mat, pi, seq1, seq2)
        likelihoods.append(like_test)
        print(f"branch length = {pwr:,.5f} subs/site; Likelihood = {like_test:.7f}")

    # Find the maximum likelihood and corresponding branch length
    max_idx = np.argmax(likelihoods)
    max_likelihood = likelihoods[max_idx]
    optimal_branch_length = optimal_range[max_idx]

    print(f"\nMaximum likelihood is {max_likelihood:.7f} at branch length {optimal_branch_length:.5f}")

    # Visualize the optimization process
    plt.figure(figsize=(12, 6))

    # Plot the coarse search
    plt.subplot(1, 2, 1)
    plt.plot(coarse_range, coarse_likelihoods, 'b-', label='Coarse search')
    plt.scatter([approx_optimal], [coarse_likelihoods[approx_max_idx]],
                color='red', s=100, label='Approx. optimal')
    plt.xlabel('Branch Length (substitutions/site)')
    plt.ylabel('Likelihood')
    plt.title('Coarse Search for Optimal Branch Length')
    plt.grid(True)
    plt.legend()

    # Plot the fine search
    plt.subplot(1, 2, 2)
    plt.plot(optimal_range, likelihoods, 'g-', label='Fine search')
    plt.scatter([optimal_branch_length], [max_likelihood],
                color='red', s=100, label='Optimal')
    plt.xlabel('Branch Length (substitutions/site)')
    plt.ylabel('Likelihood')
    plt.title('Fine Search for Optimal Branch Length')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    display_footer()

def run_interactive():
    """Run the interactive version of the program."""
    display_header()

    print("Welcome to the interactive DNA likelihood calculator!")
    print("This program will help you understand the calculations for estimating")
    print("the likelihood of evolutionary change from one DNA sequence to another.")

    # 1. Choose or input rate matrix
    print("\n=== Rate Matrix Selection ===")
    print("1. Exemplar")
    print("2. Jukes-Cantor")
    print("3. Kimura 2-parameter")
    print("4. HKY85")
    print("5. General Time Reversible (GTR)")
    print("6. Custom (input your own)")

    matrix_choice = input("Choose a rate matrix (1-6): ")

    if matrix_choice == "1":
        r = create_rate_matrix("exemplar")
    elif matrix_choice == "2":
        r = create_rate_matrix("jukes_cantor")
    elif matrix_choice == "3":
        r = create_rate_matrix("kimura2p")
    elif matrix_choice == "4":
        r = create_rate_matrix("hky85")
    elif matrix_choice == "5":
        r = create_rate_matrix("gtr")
    elif matrix_choice == "6":
        r = interactive_matrix_input()
    else:
        print("Invalid choice. Using exemplar matrix.")
        r = create_rate_matrix("exemplar")

    # 2. Choose or input composition vector
    print("\n=== Composition Vector Selection ===")
    print("1. exemplar (A=0.1, C=0.4, G=0.2, T=0.3)")
    print("2. Equal (A=0.25, C=0.25, G=0.25, T=0.25)")
    print("3. Custom (input your own)")

    comp_choice = input("Choose a composition vector (1-3): ")

    if comp_choice == "1":
        pi = create_composition_vector("exemplar")
    elif comp_choice == "2":
        pi = create_composition_vector("equal")
    elif comp_choice == "3":
        pi = interactive_composition_input()
    else:
        print("Invalid choice. Using exemplar composition.")
        pi = create_composition_vector("exemplar")

    # 3. Calculate scaled Q matrix
    scaled_q = calculate_scaled_q(r, pi)

    # 4. Input sequences
    print("\n=== Sequence Input ===")
    print("Enter two DNA sequences of equal length (A, C, G, T only)")

    while True:
        seq1 = input("Sequence 1: ").upper()
        if all(c in "ACGT" for c in seq1):
            break
        print("Invalid sequence. Use only A, C, G, T.")

    while True:
        seq2 = input("Sequence 2: ").upper()
        if all(c in "ACGT" for c in seq2) and len(seq2) == len(seq1):
            break
        print("Invalid sequence or length mismatch. Use only A, C, G, T.")

    # 5. Calculate P matrices for different branch lengths
    print("\n=== Branch Length Selection ===")
    print("1. Use default branch lengths (0.02, 0.1, 0.2, 20.0)")
    print("2. Input custom branch lengths")
    print("3. Automatic optimization to find maximum likelihood")

    branch_choice = input("Choose an option (1-3): ")

    if branch_choice == "1":
        branch_lengths = [0.02, 0.1, 0.2, 20.0]
    elif branch_choice == "2":
        branch_lengths = []
        num_branches = int(input("How many branch lengths do you want to try? "))
        for i in range(num_branches):
            try:
                length = float(input(f"Branch length {i+1}: "))
                branch_lengths.append(length)
            except ValueError:
                print("Invalid input. Using 0.1 as default.")
                branch_lengths.append(0.1)
    else:
        branch_lengths = np.linspace(0.001, 1.0, 100)  # For optimization

    # 6. Calculate P matrices
    p_matrices = calculate_p_matrices(scaled_q, branch_lengths if branch_choice != "3" else [0.02, 0.1, 0.2])

    # 7. Calculate likelihoods
    if branch_choice == "3":
        print("\nOptimizing branch length to find maximum likelihood...")
        likelihoods = plot_likelihood_vs_branch_length(scaled_q, pi, seq1, seq2, branch_lengths)

        # Find and report maximum likelihood
        max_idx = np.argmax(likelihoods)
        max_likelihood = likelihoods[max_idx]
        optimal_branch_length = branch_lengths[max_idx]

        print(f"\nMaximum likelihood: {max_likelihood:.7f}")
        print(f"Optimal branch length: {optimal_branch_length:.5f} substitutions per site")

        # Calculate the P matrix for the optimal branch length
        optimal_p = linalg.expm(scaled_q * optimal_branch_length)

        # Explain the likelihood calculation for the optimal branch length
        explain_likelihood(optimal_p, pi, seq1, seq2, optimal_branch_length)
    else:
        # Calculate and explain likelihood for each branch length
        print("\n=== Likelihood Calculations ===")

        for branch_length in branch_lengths:
            print(f"\n--- Branch Length: {branch_length} ---")
            explain_likelihood(p_matrices[branch_length], pi, seq1, seq2, branch_length)

        # Offer to plot likelihood vs branch length
        plot_choice = input("\nWould you like to plot likelihood vs branch length? (y/n): ")
        if plot_choice.lower() == "y":
            fine_branch_lengths = np.linspace(0.001, 1.0, 100)
            plot_likelihood_vs_branch_length(scaled_q, pi, seq1, seq2, fine_branch_lengths)

    display_footer()

def run_tests():
    """Run unit tests to verify code correctness."""
    print("Running unit tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("All tests completed.")

def parse_fasta(fasta_content):
    """
    Parse FASTA format string into a dictionary of sequences.

    Parameters:
    ----------
    fasta_content : str
        FASTA format string

    Returns:
    -------
    dict
        Dictionary mapping sequence names to sequences
    """
    sequences = {}
    current_name = None
    current_seq = []

    for line in fasta_content.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith('>'):
            # Save the previous sequence if it exists
            if current_name is not None:
                sequences[current_name] = ''.join(current_seq)

            # Start a new sequence
            current_name = line[1:].strip()
            current_seq = []
        else:
            # Add to the current sequence
            current_seq.append(line)

    # Save the last sequence
    if current_name is not None and current_seq:
        sequences[current_name] = ''.join(current_seq)

    return sequences

def load_sequences_from_file():
    """
    Let user load sequences from a file or input manually.

    Returns:
    -------
    tuple
        Tuple containing (seq1, seq2)
    """
    print("\n=== Sequence Input ===")
    print("1. Enter sequences manually")
    print("2. Enter sequences in FASTA format")
    print("3. Use example sequences")

    choice = input("Choose an option (1-3): ")

    if choice == "1":
        # Manual input
        while True:
            seq1 = input("Sequence 1: ").upper()
            if all(c in "ACGT" for c in seq1):
                break
            print("Invalid sequence. Use only A, C, G, T.")

        while True:
            seq2 = input("Sequence 2: ").upper()
            if all(c in "ACGT" for c in seq2) and len(seq2) == len(seq1):
                break
            print("Invalid sequence or length mismatch. Use only A, C, G, T.")

    elif choice == "2":
        # FASTA format
        print("Enter sequences in FASTA format (enter an empty line to finish):")
        print("Example:")
        print(">Sequence1")
        print("ACGT")
        print(">Sequence2")
        print("ACTT")

        fasta_lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            fasta_lines.append(line)

        fasta_content = "\n".join(fasta_lines)
        sequences = parse_fasta(fasta_content)

        if len(sequences) < 2:
            print("Not enough sequences provided. Using example sequences.")
            return ("CCATGGATC", "CCGTGGATA")

        seq_names = list(sequences.keys())
        seq1 = sequences[seq_names[0]]
        seq2 = sequences[seq_names[1]]

        # Validate sequences
        if not all(c in "ACGT" for c in seq1) or not all(c in "ACGT" for c in seq2):
            print("Invalid characters in sequences. Using example sequences.")
            return ("CCATGGATC", "CCGTGGATA")

        if len(seq1) != len(seq2):
            print("Sequences must have the same length. Using example sequences.")
            return ("CCATGGATC", "CCGTGGATA")

    else:
        # Example sequences
        print("Using example sequences:")
        seq1 = "CCATGGATC"
        seq2 = "CCGTGGATA"
        print(f"Sequence 1: {seq1}")
        print(f"Sequence 2: {seq2}")

    return (seq1, seq2)

    # Define models to compare
    models = ["jukes_cantor", "kimura2p", "hky85", "gtr"]
    model_names = ["Jukes-Cantor", "Kimura 2-parameter", "HKY85", "GTR"]

    # Track results
    max_likelihoods = []
    optimal_branch_lengths = []

    # Compare each model
    plt.figure(figsize=(12, 8))

    for i, model in enumerate(models):
        print(f"\nAnalyzing with {model_names[i]} model...")

        # Create rate matrix and equal composition vector
        r = create_rate_matrix(model)
        pi = create_composition_vector("equal")

        # Calculate scaled Q matrix
        scaled_q = calculate_scaled_q(r, pi, explain=False)

        # Calculate likelihoods for a range of branch lengths
        branch_lengths = np.linspace(0.01, 1.0, 100)
        likelihoods = []

        for brlen in branch_lengths:
            p_mat = linalg.expm(scaled_q * brlen)
            like = calculate_likelihood(p_mat, pi, seq1, seq2)
            likelihoods.append(like)

        # Find maximum likelihood and optimal branch length
        max_idx = np.argmax(likelihoods)
        max_likelihood = likelihoods[max_idx]
        optimal_branch_length = branch_lengths[max_idx]

        max_likelihoods.append(max_likelihood)
        optimal_branch_lengths.append(optimal_branch_length)

        # Plot the results
        plt.plot(branch_lengths, likelihoods, label=model_names[i])
        plt.scatter([optimal_branch_length], [max_likelihood], marker='o')

        print(f"Maximum likelihood: {max_likelihood:.7f}")
        print(f"Optimal branch length: {optimal_branch_length:.5f}")

    # Finalize plot
    plt.xlabel('Branch Length (substitutions/site)')
    plt.ylabel('Likelihood')
    plt.title('Comparison of Evolutionary Models')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Create a summary bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, max_likelihoods)
    plt.xlabel('Evolutionary Model')
    plt.ylabel('Maximum Likelihood')
    plt.title('Maximum Likelihood by Model')

    # Add text labels on the bars
    for i, v in enumerate(max_likelihoods):
        plt.text(i, v+0.00001, f"{v:.7f}", ha='center', rotation=90)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\n=== Model Comparison Summary ===")
    print("Model".ljust(20) + "Maximum Likelihood".ljust(25) + "Optimal Branch Length")
    print("-" * 65)

    for i in range(len(models)):
        print(model_names[i].ljust(20) +
              f"{max_likelihoods[i]:.7f}".ljust(25) +
              f"{optimal_branch_lengths[i]:.5f}")

def run_tutorial():
    """Run a step-by-step tutorial on DNA likelihood calculation."""
    display_header()

    print("""
=== DNA Likelihood Calculation Tutorial ===

This tutorial will guide you through the process of calculating the likelihood
of DNA sequence evolution using different evolutionary models. We'll cover:

1. Rate matrices and their properties
2. Composition vectors and their interpretation
3. Q matrix calculation and scaling
4. P matrix derivation through matrix exponentiation
5. Likelihood calculation for sequences
6. Optimizing branch lengths
7. Comparing different evolutionary models

Let's begin with rate matrices...
""")

    input("Press Enter to continue...")

    # Step 1: Rate matrices
    print("""
Step 1: Rate Matrices
--------------------
A rate matrix represents the relative rates of substitution between nucleotides.
For instance, in the Jukes-Cantor model, all substitutions have the same rate.

The rate matrix is a 4x4 matrix where:
- Rows and columns correspond to A, C, G, T
- Off-diagonal elements represent rates of change
- Diagonal elements are not specified (will be set to make rows sum to zero)

Here's a comparison of different rate matrices:
""")

    # Display different rate matrices
    models = ["jukes_cantor", "kimura2p", "hky85", "gtr"]
    model_names = ["Jukes-Cantor", "Kimura 2-parameter", "HKY85", "GTR"]

    for i, model in enumerate(models):
        r = create_rate_matrix(model)
        print(f"\n{model_names[i]} Rate Matrix:")
        print(r)

        # For the first model, show a visualization
        if i == 0:
            plt.figure(figsize=(8, 6))
            bases = ['A', 'C', 'G', 'T']

            plt.imshow(r, cmap='viridis')
            plt.colorbar(label='Relative Rate')
            plt.title(f'{model_names[i]} Rate Matrix Visualization')

            # Add labels
            plt.xticks(np.arange(4), bases)
            plt.yticks(np.arange(4), bases)
            plt.xlabel('To')
            plt.ylabel('From')

            # Add text annotations
            for i in range(4):
                for j in range(4):
                    text_color = 'white' if r[i, j] > 0.5 else 'black'
                    plt.text(j, i, f'{r[i, j]:.1f}',
                            ha='center', va='center', color=text_color)

            plt.show()

    input("Press Enter to continue to the next step...")

    # Step 2: Composition vectors
    print("""
Step 2: Composition Vectors
-------------------------
A composition vector represents the equilibrium frequencies of nucleotides.
In other words, it tells us the expected long-term proportion of each nucleotide.

For example:
- Equal frequencies: [0.25, 0.25, 0.25, 0.25]
- GC-rich: [0.1, 0.4, 0.4, 0.1]
- AT-rich: [0.4, 0.1, 0.1, 0.4]

The composition vector affects the calculation of the Q matrix.
""")

    # Display different composition vectors
    print("Equal frequencies:")
    print(create_composition_vector("equal"))
    print("\Exemplar frequencies (used in this tutorial):")
    print(create_composition_vector("exemplar"))

    # Visualize composition vectors
    plt.figure(figsize=(10, 6))

    # Create a few composition vectors
    compositions = {
        "Equal": [0.25, 0.25, 0.25, 0.25],
        "Exemplar": [0.1, 0.4, 0.2, 0.3],
        "GC-rich": [0.1, 0.4, 0.4, 0.1],
        "AT-rich": [0.4, 0.1, 0.1, 0.4]
    }

    bases = ['A', 'C', 'G', 'T']

    # Plot each composition
    x = np.arange(len(bases))
    width = 0.2
    offsets = [-width*1.5, -width/2, width/2, width*1.5]

    for i, (name, comp) in enumerate(compositions.items()):
        plt.bar(x + offsets[i], comp, width, label=name)

    plt.xlabel('Nucleotide')
    plt.ylabel('Frequency')
    plt.title('Comparison of Different Composition Vectors')
    plt.xticks(x, bases)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    input("Press Enter to continue to the next step...")

    # Step 3: Q matrix calculation
    print("""
Step 3: Q Matrix Calculation
--------------------------
The Q matrix combines the rate matrix and composition vector.
It represents the instantaneous rates of change between nucleotides.

The calculation steps are:
1. Multiply the rate matrix by the composition vector
2. Set diagonal elements to make rows sum to zero
3. Scale the matrix to give an average of 1 substitution per site

Let's illustrate these steps with the Jukes-Cantor model and equal frequencies:
""")

    # Create a JC model with equal frequencies
    r_jc = create_rate_matrix("jukes_cantor")
    pi_equal = create_composition_vector("equal")

    # Calculate scaled Q matrix with explanation
    scaled_q_jc = calculate_scaled_q(r_jc, pi_equal, explain=True)

    input("Press Enter to continue to the next step...")

    # Step 4: P matrix calculation
    print("""
Step 4: P Matrix Calculation
--------------------------
The P matrix represents the probabilities of change over a given branch length.
It is calculated using matrix exponentiation: P(v) = e^(Qv)

Where:
- Q is the scaled Q matrix
- v is the branch length in substitutions per site
- e^ is the matrix exponential function

As the branch length increases:
- At v = 0, P is the identity matrix (no changes)
- At small v, P is close to the identity matrix (few changes)
- At large v, P approaches the stationary distribution (many changes)

Let's examine P matrices for different branch lengths:
""")

    # Calculate P matrices for different branch lengths
    branch_lengths = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]
    p_matrices = calculate_p_matrices(scaled_q_jc, branch_lengths, explain=True)

    input("Press Enter to continue to the next step...")

    # Step 5: Likelihood calculation
    print("""
Step 5: Likelihood Calculation
---------------------------
The likelihood of evolving from one sequence to another is the product of:
- The probability of observing each initial nucleotide (from the composition vector)
- The probability of each nucleotide changing (or not) to the corresponding nucleotide in the second sequence

For example, for sequences "ACGT" and "ACTT", the likelihood calculation would be:
L = P(A) * P(A→A) * P(C) * P(C→C) * P(G) * P(G→T) * P(T) * P(T→T)

Let's calculate the likelihood for a simple example:
""")

    # Define example sequences
    seq1 = "ACGT"
    seq2 = "ACTT"

    # Calculate likelihood for a few branch lengths
    for branch_length in [0.1, 0.5]:
        print(f"\nBranch length: {branch_length}")
        explain_likelihood(p_matrices[branch_length], pi_equal, seq1, seq2, branch_length)

    input("Press Enter to continue to the next step...")

    # Step 6: Optimizing branch lengths
    print("""
Step 6: Optimizing Branch Lengths
------------------------------
To find the maximum likelihood estimate of the branch length,
we calculate the likelihood for many different branch lengths
and find the one that gives the highest likelihood.

This is equivalent to finding the branch length that best explains the observed data.

Let's find the optimal branch length for our example:
""")

    # Plot likelihood vs branch length
    branch_lengths_fine = np.linspace(0.001, 1.0, 100)
    plot_likelihood_vs_branch_length(scaled_q_jc, pi_equal, seq1, seq2, branch_lengths_fine)

    input("Press Enter to continue to the final step...")

    # Step 7: Comparing models
    print("""
Step 7: Comparing Evolutionary Models
----------------------------------
Different evolutionary models make different assumptions about the substitution process.
We can compare models by calculating the maximum likelihood under each model.

However, more complex models (with more parameters) will generally fit better.
To account for this, we can use criteria like AIC (Akaike Information Criterion)
that penalize model complexity: AIC = 2k - 2ln(L)

Let's compare different models on a longer sequence pair:
""")

    # Define longer sequences
    seq1_long = "CCATGGATCTA"
    seq2_long = "CCGTGGATCCA"

    # Compare models
    models = ["jukes_cantor", "kimura2p"]
    model_names = ["Jukes-Cantor", "Kimura 2-parameter"]

    # Track results
    max_likelihoods = []
    optimal_branch_lengths = []

    # Compare each model
    plt.figure(figsize=(12, 8))

    for i, model in enumerate(models):
        print(f"\nAnalyzing with {model_names[i]} model...")

        # Create rate matrix
        r = create_rate_matrix(model)
        pi = create_composition_vector("equal")

        # Calculate scaled Q matrix
        scaled_q = calculate_scaled_q(r, pi, explain=False)

        # Calculate likelihoods for a range of branch lengths
        branch_lengths = np.linspace(0.01, 1.0, 100)
        likelihoods = []

        for brlen in branch_lengths:
            p_mat = linalg.expm(scaled_q * brlen)
            like = calculate_likelihood(p_mat, pi, seq1_long, seq2_long)
            likelihoods.append(like)

        # Find maximum likelihood and optimal branch length
        max_idx = np.argmax(likelihoods)
        max_likelihood = likelihoods[max_idx]
        optimal_branch_length = branch_lengths[max_idx]

        max_likelihoods.append(max_likelihood)
        optimal_branch_lengths.append(optimal_branch_length)

        # Plot the results
        plt.plot(branch_lengths, likelihoods, label=model_names[i])
        plt.scatter([optimal_branch_length], [max_likelihood], marker='o')

        print(f"Maximum likelihood: {max_likelihood:.7f}")
        print(f"Optimal branch length: {optimal_branch_length:.5f}")

    # Finalize plot
    plt.xlabel('Branch Length (substitutions/site)')
    plt.ylabel('Likelihood')
    plt.title('Comparison of Evolutionary Models')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate AIC
    print("\nCalculating AIC for model comparison:")

    param_counts = {"jukes_cantor": 1, "kimura2p": 2}

    for i, model in enumerate(models):
        k = param_counts[model]
        log_likelihood = np.log(max_likelihoods[i])
        aic = 2 * k - 2 * log_likelihood

        print(f"{model_names[i]}:")
        print(f"  Parameters: {k}")
        print(f"  Log-likelihood: {log_likelihood:.6f}")
        print(f"  AIC: {aic:.6f}")

    print("""
Tutorial Complete!
---------------
You've now learned how to:
1. Understand rate matrices and composition vectors
2. Calculate Q and P matrices
3. Compute the likelihood of DNA sequence evolution
4. Find the optimal branch length
5. Compare different evolutionary models

Feel free to explore the other features of this program!
""")

    display_footer()

def run_bootstrap_analysis():
    """
    Perform bootstrap analysis to estimate confidence intervals.

    Bootstrap resampling allows us to estimate the uncertainty in our
    parameter estimates by creating multiple resampled datasets.
    """
    print("\n=== Bootstrap Analysis ===")

    # Load sequences
    seq1, seq2 = load_sequences_from_file()

    if len(seq1) < 10:
        print("Bootstrap analysis requires sequences of at least 10 nucleotides.")
        print(f"Current sequences are only {len(seq1)} nucleotides long.")
        print("Using example longer sequences instead.")
        seq1 = "ACGTACGTACGTACGTACGTACGTACGT"
        seq2 = "ACTTACGCACGTACTTACGTACGAACTT"

    # Choose model
    print("\nWhich evolutionary model would you like to use?")
    print("1. Jukes-Cantor")
    print("2. Kimura 2-parameter")
    print("3. HKY85")
    print("4. GTR")

    model_choice = input("Choose a model (1-4): ")

    if model_choice == "1":
        model = "jukes_cantor"
        model_name = "Jukes-Cantor"
    elif model_choice == "2":
        model = "kimura2p"
        model_name = "Kimura 2-parameter"
    elif model_choice == "3":
        model = "hky85"
        model_name = "HKY85"
    elif model_choice == "4":
        model = "gtr"
        model_name = "GTR"
    else:
        print("Invalid choice. Using Jukes-Cantor model.")
        model = "jukes_cantor"
        model_name = "Jukes-Cantor"

    # Choose number of bootstrap replicates
    try:
        n_bootstraps = int(input("\nNumber of bootstrap replicates (10-1000, default=100): "))
        if n_bootstraps < 10 or n_bootstraps > 1000:
            print("Invalid number. Using 100 replicates.")
            n_bootstraps = 100
    except ValueError:
        print("Invalid input. Using 100 replicates.")
        n_bootstraps = 100

    # Create rate matrix and composition vector
    r = create_rate_matrix(model)
    pi = create_composition_vector("equal")

    # Calculate scaled Q matrix
    scaled_q = calculate_scaled_q(r, pi, explain=False)

    # Find the optimal branch length for the original data
    print(f"\nAnalyzing original sequences with {model_name} model...")

    branch_lengths = np.linspace(0.01, 1.0, 100)
    likelihoods = []

    for brlen in branch_lengths:
        p_mat = linalg.expm(scaled_q * brlen)
        like = calculate_likelihood(p_mat, pi, seq1, seq2)
        likelihoods.append(like)

    max_idx = np.argmax(likelihoods)
    optimal_branch_length = branch_lengths[max_idx]

    print(f"Original optimal branch length: {optimal_branch_length:.5f}")

    # Create site pairs for resampling
    site_pairs = []
    for i in range(len(seq1)):
        site_pairs.append((seq1[i], seq2[i]))

    # Perform bootstrap
    print(f"\nPerforming bootstrap analysis with {n_bootstraps} replicates...")
    bootstrap_branch_lengths = []

    for i in range(n_bootstraps):
        if i % 10 == 0:
            print(f"Completed {i} bootstraps...")

        # Create bootstrap sample
        bootstrap_indices = np.random.choice(len(seq1), size=len(seq1), replace=True)
        bootstrap_site_pairs = [site_pairs[j] for j in bootstrap_indices]

        # Extract bootstrap sequences
        bootstrap_seq1 = ''.join([pair[0] for pair in bootstrap_site_pairs])
        bootstrap_seq2 = ''.join([pair[1] for pair in bootstrap_site_pairs])

        # Find optimal branch length
        bootstrap_likelihoods = []

        for brlen in branch_lengths:
            p_mat = linalg.expm(scaled_q * brlen)
            like = calculate_likelihood(p_mat, pi, bootstrap_seq1, bootstrap_seq2)
            bootstrap_likelihoods.append(like)

        bootstrap_max_idx = np.argmax(bootstrap_likelihoods)
        bootstrap_optimal_branch_length = branch_lengths[bootstrap_max_idx]

        bootstrap_branch_lengths.append(bootstrap_optimal_branch_length)

    # Calculate confidence intervals
    bootstrap_branch_lengths.sort()
    ci_lower = bootstrap_branch_lengths[int(0.025 * n_bootstraps)]
    ci_upper = bootstrap_branch_lengths[int(0.975 * n_bootstraps)]

    print("\nBootstrap analysis complete!")
    print(f"Original optimal branch length: {optimal_branch_length:.5f}")
    print(f"95% confidence interval: [{ci_lower:.5f}, {ci_upper:.5f}]")

    # Plot histogram of bootstrap results
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_branch_lengths, bins=30, alpha=0.7, color='skyblue')

    # Add lines for original estimate and confidence interval
    plt.axvline(optimal_branch_length, color='red', linestyle='-', linewidth=2, label='Original estimate')
    plt.axvline(ci_lower, color='green', linestyle='--', linewidth=2, label='95% CI lower')
    plt.axvline(ci_upper, color='green', linestyle='--', linewidth=2, label='95% CI upper')

    plt.xlabel('Branch Length (substitutions/site)')
    plt.ylabel('Frequency')
    plt.title(f'Bootstrap Distribution of Optimal Branch Length ({model_name} model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Provide interpretation
    print("\nInterpretation:")
    print("- The bootstrap analysis gives us an estimate of the uncertainty in our branch length estimate.")
    print("- The width of the confidence interval indicates the precision of our estimate.")
    print("- A narrow interval suggests high confidence in the estimated branch length.")
    print("- A wide interval suggests uncertainty, possibly due to limited data or model inadequacy.")

    # Calculate and display additional statistics
    bootstrap_mean = np.mean(bootstrap_branch_lengths)
    bootstrap_median = np.median(bootstrap_branch_lengths)
    bootstrap_std = np.std(bootstrap_branch_lengths)

    print("\nAdditional statistics:")
    print(f"Mean bootstrap estimate: {bootstrap_mean:.5f}")
    print(f"Median bootstrap estimate: {bootstrap_median:.5f}")
    print(f"Standard deviation: {bootstrap_std:.5f}")

    # Check for bias
    bias = bootstrap_mean - optimal_branch_length
    print(f"Bias (mean - original): {bias:.5f}")
    if abs(bias) > 0.05:
        print("Note: The bootstrap mean differs substantially from the original estimate,")
        print("which may indicate bias in the estimation procedure.")
    else:
        print("The bootstrap mean is close to the original estimate, suggesting low bias.")

def show_parameter_guide():
    """
    Display a guide to interpreting evolutionary model parameters.
    """
    print("\n=== Evolutionary Model Parameter Interpretation Guide ===")

    print("""
This guide explains how to interpret the parameters of different evolutionary models
and their biological significance.

1. Branch Length
--------------
The branch length (v) represents the expected number of substitutions per site.
It is a measure of evolutionary distance between sequences.

Interpretation:
- v = 0.01: Approximately 1% of sites have changed
- v = 0.1: Approximately 10% of sites have changed
- v = 1.0: On average, each site has changed once
- v > 5.0: Sequences are highly diverged, with multiple substitutions at many sites

Note: For highly diverged sequences, the actual number of observed differences
will be less than the branch length due to multiple substitutions at the same site.
""")


    # Create a plot showing relationship between observed differences and branch length
    plt.figure(figsize=(10, 6))

    # Branch lengths (actual number of substitutions per site)
    branch_lengths = np.linspace(0, 3, 100)

    # Under JC model, calculate observed proportion of differences
    # The formula for JC model: p_observed = 3/4 * (1 - e^(-4t/3))
    # where t is the branch length
    observed_diff = 0.75 * (1 - np.exp(-(4 * branch_lengths) / 3))

    # Plot both the actual substitutions (branch length) and the observed differences
    plt.plot(branch_lengths, branch_lengths, 'r-',
             label='Actual substitutions (branch length)')
    plt.plot(branch_lengths, observed_diff, 'b-',
             label='Observed differences (JC model)')

    # Add the JC correction formula
    plt.text(1.5, 0.5, r'$p_{obs} = \frac{3}{4}(1 - e^{-\frac{4t}{3}})$',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Add a note explaining saturation
    plt.text(2.1, 0.65, 'Multiple substitutions\ncause saturation',
             fontsize=10, ha='center', bbox=dict(facecolor='lightyellow', alpha=0.8))

    # Add an arrow pointing to the divergence
    plt.annotate('', xy=(2.5, 0.73), xytext=(2.5, 0.55),
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Evolutionary Distance (substitutions per site)')
    plt.ylabel('Proportion of Sites')
    plt.title('Relationship Between Actual Substitutions and Observed Differences')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 0.8)
    plt.xlim(0, 3)
    plt.show()

    print("""
    The above plot illustrates a critical concept in molecular evolution: the difference
    between the actual number of substitutions that have occurred (branch length) and
    the observed proportion of differences between sequences.

    Key points:
    - The red line (y=x) represents the actual number of substitutions per site.
    - The blue curve shows the observable proportion of differences under the JC model.
    - As evolutionary distance increases, multiple substitutions occur at the same sites.
    - These multiple hits cause "saturation" - we observe fewer differences than actually occurred.
    - The JC model (and other models) correct for this saturation effect.

    Example: At an evolutionary distance of 2.0 substitutions per site, we would only
    observe differences at about 69% of sites, not the 200% that would be expected
    if every substitution produced an observable difference.

    This is why we need mathematical models - to estimate the true evolutionary distance
    from the observed differences between sequences.
    """)



    input("Press Enter to continue...")

    print("""
2. Jukes-Cantor Model
------------------
The Jukes-Cantor model assumes all substitutions occur at the same rate.

Parameters:
- Branch length (v): The only free parameter

Biological significance:
- Simplest model of nucleotide substitution
- Assumes equal base frequencies (25% each)
- All substitutions equally likely
- Appropriate for closely related sequences with little compositional bias
""")

    # JC model visualization
    plt.figure(figsize=(8, 6))
    r_jc = create_rate_matrix("jukes_cantor")

    plt.imshow(r_jc, cmap='Blues')
    plt.colorbar(label='Relative Rate')
    plt.title('Jukes-Cantor Rate Matrix')
    plt.xticks(np.arange(4), ['A', 'C', 'G', 'T'])
    plt.yticks(np.arange(4), ['A', 'C', 'G', 'T'])
    plt.xlabel('To')
    plt.ylabel('From')

    for i in range(4):
        for j in range(4):
            text_color = 'white' if r_jc[i, j] > 0.5 else 'black'
            plt.text(j, i, f'{r_jc[i, j]:.1f}',
                    ha='center', va='center', color=text_color)

    plt.show()

    input("Press Enter to continue...")

    print("""
3. Kimura 2-Parameter Model
------------------------
The K2P model distinguishes between transitions (A↔G, C↔T) and transversions (A↔C, A↔T, G↔C, G↔T).

Parameters:
- Branch length (v): Overall rate of substitution
- Transition/transversion ratio (κ): Relative rate of transitions vs. transversions

Typical values (Note - your data is likely to differ):
- κ = 2-5: Transitions occur 2-5 times more frequently than transversions

Biological significance:
- Transitions are biochemically more likely than transversions
- Transitions involve purine-to-purine (A↔G) or pyrimidine-to-pyrimidine (C↔T) changes
- Appropriate for sequences with transition bias but equal base frequencies
""")

    # K2P model visualization
    plt.figure(figsize=(8, 6))
    r_k2p = create_rate_matrix("kimura2p")

    plt.imshow(r_k2p, cmap='Blues')
    plt.colorbar(label='Relative Rate')
    plt.title('Kimura 2-Parameter Rate Matrix')
    plt.xticks(np.arange(4), ['A', 'C', 'G', 'T'])
    plt.yticks(np.arange(4), ['A', 'C', 'G', 'T'])
    plt.xlabel('To')
    plt.ylabel('From')

    for i in range(4):
        for j in range(4):
            text_color = 'white' if r_k2p[i, j] > 1.0 else 'black'
            plt.text(j, i, f'{r_k2p[i, j]:.1f}',
                    ha='center', va='center', color=text_color)

    plt.show()

    input("Press Enter to continue...")

    print("""
4. HKY85 Model
-----------
The HKY85 model combines unequal base frequencies with the transition/transversion distinction.

Parameters:
- Branch length (v): Overall rate of substitution
- Transition/transversion ratio (κ): Relative rate of transitions vs. transversions
- Base frequencies (πA, πC, πG, πT): Equilibrium frequencies of each nucleotide

Biological significance:
- Accounts for both compositional bias and transition bias
- More realistic for most biological sequences
- Particularly useful for sequences with GC or AT bias
""")

    # HKY85 P matrix visualization for different branch lengths
    r_hky = create_rate_matrix("hky85")
    pi_unequal = create_composition_vector("exemplar")  # Example unequal composition
    scaled_q_hky = calculate_scaled_q(r_hky, pi_unequal, explain=False)

    # Calculate P matrices
    branch_lengths = [0.1, 1.0]

    plt.figure(figsize=(15, 7))
    for i, brlen in enumerate(branch_lengths):
        p_mat = linalg.expm(scaled_q_hky * brlen)

        plt.subplot(1, 2, i+1)
        plt.imshow(p_mat, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(label='Probability')
        plt.title(f'HKY85 P Matrix (v = {brlen})')
        plt.xticks(np.arange(4), ['A', 'C', 'G', 'T'])
        plt.yticks(np.arange(4), ['A', 'C', 'G', 'T'])
        plt.xlabel('To')
        plt.ylabel('From')

        for i in range(4):
            for j in range(4):
                text_color = 'white' if p_mat[i, j] > 0.7 else 'black'
                plt.text(j, i, f'{p_mat[i, j]:.2f}',
                        ha='center', va='center', color=text_color)

    plt.tight_layout()
    plt.show()

    input("Press Enter to continue...")

    print("""
5. General Time Reversible (GTR) Model
----------------------------------
The GTR model is the most general time-reversible model, allowing all substitution types
to have different rates.

Parameters:
- Branch length (v): Overall rate of substitution
- Relative substitution rates (rAC, rAG, rAT, rCG, rCT, rGT): Specific rates for each substitution type
- Base frequencies (πA, πC, πG, πT): Equilibrium frequencies of each nucleotide

Biological significance:
- Most parameter-rich time-reversible model, with 9 free parameters
- Captures all aspects of substitution bias
- Most realistic for divergent sequences
- May overfit for closely related sequences (simpler models might be better)
""")

    # Diagram showing model relationships
    plt.figure(figsize=(10, 8))

    # Define model positions
    models = {
        "JC": (0.5, 0.9, "Jukes-Cantor\n1 parameter"),
        "K2P": (0.5, 0.7, "Kimura 2-parameter\n2 parameters"),
        "F81": (0.25, 0.5, "Felsenstein 81\n4 parameters"),
        "HKY": (0.75, 0.5, "HKY85\n5 parameters"),
        "TN93": (0.5, 0.3, "Tamura-Nei 93\n6 parameters"),
        "GTR": (0.5, 0.1, "GTR\n9 parameters")
    }

    # Plot points for each model
    for model, (x, y, label) in models.items():
        plt.scatter(x, y, s=100, zorder=5)
        plt.annotate(label, (x, y), xytext=(0, 10),
                     textcoords='offset points', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    # Add connections
    plt.plot([models["JC"][0], models["K2P"][0]], [models["JC"][1], models["K2P"][1]], 'k-')
    plt.plot([models["JC"][0], models["F81"][0]], [models["JC"][1], models["F81"][1]], 'k-')
    plt.plot([models["K2P"][0], models["HKY"][0]], [models["K2P"][1], models["HKY"][1]], 'k-')
    plt.plot([models["F81"][0], models["HKY"][0]], [models["F81"][1], models["HKY"][1]], 'k-')
    plt.plot([models["HKY"][0], models["TN93"][0]], [models["HKY"][1], models["TN93"][1]], 'k-')
    plt.plot([models["TN93"][0], models["GTR"][0]], [models["TN93"][1], models["GTR"][1]], 'k-')

    plt.title('Relationships Between Evolutionary Models')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.show()

    input("Press Enter to continue...")

    print("""
6. Model Selection
--------------
When comparing models, we must balance goodness-of-fit with model complexity.
More complex models will generally fit better, but they may overfit the data.

Model selection criteria:
- Likelihood Ratio Test (LRT): For nested models, tests if additional parameters significantly improve fit
- Akaike Information Criterion (AIC): AIC = 2k - 2ln(L), where k is the number of parameters
- Bayesian Information Criterion (BIC): BIC = k*ln(n) - 2ln(L), where n is the number of sites

Guidelines:
- Lower AIC or BIC values indicate better models
- A difference in AIC of >2 is considered significant
- A difference in AIC of >10 is considered strong
- BIC penalizes additional parameters more heavily than AIC

Example interpretation:
- If simple models (JC, K2P) have similar AIC to complex models, prefer the simpler model
- If complex models have substantially lower AIC, they are capturing important aspects of evolution
""")

    # AIC/BIC example visualization
    plt.figure(figsize=(10, 6))

    models = ["JC", "K2P", "F81", "HKY", "TN93", "GTR"]
    parameters = [1, 2, 4, 5, 6, 9]

    # Hypothetical AIC values (for illustration)
    aic_values = [100, 90, 95, 85, 83, 84]
    bic_values = [102, 94, 103, 95, 96, 102]

    plt.plot(parameters, aic_values, 'b-o', label='AIC')
    plt.plot(parameters, bic_values, 'r-s', label='BIC')

    # Mark minimum values
    min_aic_idx = np.argmin(aic_values)
    min_bic_idx = np.argmin(bic_values)

    plt.scatter(parameters[min_aic_idx], aic_values[min_aic_idx], s=200,
                facecolors='none', edgecolors='blue', linewidths=2)
    plt.scatter(parameters[min_bic_idx], bic_values[min_bic_idx], s=200,
                facecolors='none', edgecolors='red', linewidths=2)

    plt.xticks(parameters)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Information Criterion Value')
    plt.title('Model Selection Using AIC and BIC (Hypothetical example)')
    plt.grid(True)
    plt.legend()

    # Add model names
    for i, model in enumerate(models):
        plt.text(parameters[i], aic_values[i]-5, model, ha='center')

    plt.show()

    print("""
7. Practical Considerations
-----------------------
When applying these models in practice, consider:

- Data quality: Ensure sequences are properly aligned
- Model assumptions: Check if your data meets the assumptions of the model
- Statistical significance: Use confidence intervals to assess uncertainty
- Biological plausibility: Consider what model makes sense for your organism/gene
- Computational limitations: More complex models require more computation

Remember that all models are simplifications of reality. The goal is to find
a model that captures the important aspects of evolution while remaining
statistically tractable.
""")

    input("Press Enter to return to the main menu...")

def main():
    """Main function to run the program."""
    print("DNA Likelihood Calculator")
    print("========================")
    print("1. Run example calculation with explanation")
    print("2. Run interactive mode")
    print("3. Run unit tests")
    print("4. Compare evolutionary models")
    print("5. Interactive tutorial")
    print("6. Bootstrap analysis")
    print("7. Parameter interpretation guide")

    choice = input("Enter your choice (1-7): ")

    if choice == "1":
        run_example()
    elif choice == "2":
        run_interactive()
    elif choice == "3":
        run_tests()
    elif choice == "4":
        compare_evolutionary_models()
    elif choice == "5":
        run_tutorial()
    elif choice == "6":
        run_bootstrap_analysis()
    elif choice == "7":
        show_parameter_guide()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
