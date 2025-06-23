import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import functools
import unittest
import os
import sys

# Set numpy print formatting
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def pause_for_user(message="Press Enter to continue..."):
    """Pause and wait for user input."""
    input(f"\n{message}")

def display_welcome():
    """Display welcome message and introduction."""
    clear_screen()
    print("="*70)
    print("   DNA LIKELIHOOD CALCULATOR - INTERACTIVE EDUCATIONAL TOOL")
    print("="*70)
    print("\nWelcome! This program will guide you step-by-step through the")
    print("mathematical process of calculating the likelihood of DNA sequence")
    print("evolution. We'll explore:")
    print("\n  • Rate matrices (how substitutions occur)")
    print("  • Composition vectors (nucleotide frequencies)")
    print("  • Q matrices (instantaneous rate matrices)")
    print("  • P matrices (probability matrices over time)")
    print("  • Likelihood calculations")
    print("  • Finding optimal branch lengths")
    print("\nEach step will be explained interactively!")
    print("="*70)
    pause_for_user()

def create_rate_matrix(matrix_type="exemplar"):
    """Create and return a rate matrix of specified type."""
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
        return np.array([0.0, 1.0, 2.0, 1.0,
                         1.0, 0.0, 1.0, 2.0,
                         2.0, 1.0, 0.0, 1.0,
                         1.0, 2.0, 1.0, 0.0]).reshape(4,4)
    elif matrix_type == "hky85":
        return np.array([0.0, 1.0, 2.0, 1.0,
                         1.0, 0.0, 1.0, 2.0,
                         2.0, 1.0, 0.0, 1.0,
                         1.0, 2.0, 1.0, 0.0]).reshape(4,4)
    elif matrix_type == "gtr":
        return np.array([0.0, 1.0, 2.0, 3.0,
                         1.0, 0.0, 4.0, 5.0,
                         2.0, 4.0, 0.0, 6.0,
                         3.0, 5.0, 6.0, 0.0]).reshape(4,4)
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

def create_composition_vector(vector_type="exemplar", custom=None):
    """Create and return a composition vector of specified type."""
    if custom is not None:
        if len(custom) != 4 or abs(sum(custom) - 1.0) > 1e-10:
            raise ValueError("Custom composition vector must have 4 elements and sum to 1")
        return np.array(custom)
    elif vector_type == "exemplar":
        return np.array([0.1, 0.4, 0.2, 0.3])
    elif vector_type == "equal":
        return np.array([0.25, 0.25, 0.25, 0.25])
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")

def show_matrix_with_explanation(matrix, matrix_name, bases=['A', 'C', 'G', 'T'], show_diagonal=True):
    """Display a matrix with visual representation and explanation."""
    print(f"\n{matrix_name}:")
    print("-" * 50)
    
    # Print header
    print("     ", end="")
    for base in bases:
        print(f"    {base}    ", end="")
    print("\n" + " " * 5 + "-" * 45)
    
    # Print matrix with row labels
    for i, base_from in enumerate(bases):
        print(f"  {base_from} |", end="")
        for j in range(4):
            if i == j and not show_diagonal:
                print(f"   ---   ", end="")
            else:
                print(f"  {matrix[i,j]:5.3f} ", end="")
        print()
    
    return matrix

def visualize_matrix_interactive(matrix, title, bases=['A', 'C', 'G', 'T']):
    """Create an interactive visualization of a matrix."""
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    im = plt.imshow(matrix, cmap='Blues', aspect='auto')
    plt.colorbar(im, label='Value')
    
    # Set ticks and labels
    plt.xticks(range(4), bases)
    plt.yticks(range(4), bases)
    plt.xlabel('To', fontsize=12)
    plt.ylabel('From', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text_color = 'white' if matrix[i, j] > matrix.max()/2 else 'black'
            plt.text(j, i, f'{matrix[i, j]:.3f}',
                    ha='center', va='center', color=text_color, fontsize=12)
    
    plt.tight_layout()
    plt.show()

def step1_choose_rate_matrix():
    """Step 1: Interactive selection and explanation of rate matrix."""
    clear_screen()
    print("="*70)
    print("STEP 1: CHOOSE A RATE MATRIX")
    print("="*70)
    print("\nA rate matrix defines the relative rates of substitution between")
    print("different nucleotides. Think of it as defining how 'easy' it is")
    print("for one nucleotide to change into another during evolution.")
    print("\nDifferent models make different assumptions:")
    
    print("\n1. Jukes-Cantor (JC)")
    print("   - Simplest model")
    print("   - All substitutions equally likely")
    print("   - Good for closely related sequences")
    
    print("\n2. Kimura 2-parameter (K2P)")
    print("   - Distinguishes transitions (A↔G, C↔T) from transversions")
    print("   - Transitions usually more common")
    print("   - More realistic than JC")
    
    print("\n3. HKY85")
    print("   - Like K2P but allows unequal base frequencies")
    print("   - Good for sequences with GC or AT bias")
    
    print("\n4. GTR (General Time Reversible)")
    print("   - Most general model")
    print("   - Each substitution type can have different rate")
    print("   - Most flexible but needs more data")
    
    print("\n5. Custom")
    print("   - Define your own rate matrix")
    
    while True:
        choice = input("\nChoose a model (1-5): ")
        
        if choice == "1":
            r = create_rate_matrix("jukes_cantor")
            model_name = "Jukes-Cantor"
            break
        elif choice == "2":
            r = create_rate_matrix("kimura2p")
            model_name = "Kimura 2-parameter"
            break
        elif choice == "3":
            r = create_rate_matrix("hky85")
            model_name = "HKY85"
            break
        elif choice == "4":
            r = create_rate_matrix("gtr")
            model_name = "GTR"
            break
        elif choice == "5":
            r = interactive_matrix_input()
            model_name = "Custom"
            break
        else:
            print("Invalid choice. Please choose 1-5.")
    
    print(f"\nYou selected: {model_name}")
    print("\nHere's your rate matrix:")
    show_matrix_with_explanation(r, "Rate Matrix R", show_diagonal=False)
    
    print("\nNotice:")
    print("- Diagonal values are 0 (no rate for staying the same)")
    print("- Off-diagonal values show relative substitution rates")
    
    if choice == "2":  # Kimura 2-parameter
        print("- Transitions (A↔G, C↔T) have rate 2.0")
        print("- Transversions have rate 1.0")
    
    pause_for_user("\nWould you like to see a visual representation? Press Enter...")
    visualize_matrix_interactive(r, f"{model_name} Rate Matrix")
    
    return r, model_name

def step2_choose_composition():
    """Step 2: Interactive selection of composition vector."""
    clear_screen()
    print("="*70)
    print("STEP 2: CHOOSE BASE COMPOSITION")
    print("="*70)
    print("\nThe composition vector (π) represents the equilibrium frequencies")
    print("of each nucleotide. This affects:")
    print("  • The probability of observing each nucleotide")
    print("  • The scaling of the rate matrix")
    print("  • The long-term behavior of the substitution process")
    
    print("\n1. Equal frequencies")
    print("   A=0.25, C=0.25, G=0.25, T=0.25")
    print("   - Assumes no compositional bias")
    print("   - Good default choice")
    
    print("\n2. GC-rich")
    print("   A=0.1, C=0.4, G=0.4, T=0.1")
    print("   - Common in some organisms")
    print("   - Thermal stability")
    
    print("\n3. AT-rich")
    print("   A=0.4, C=0.1, G=0.1, T=0.4")
    print("   - Common in some genomes")
    print("   - Easier to denature")
    
    print("\n4. Example composition")
    print("   A=0.1, C=0.4, G=0.2, T=0.3")
    print("   - Moderate GC bias")
    
    print("\n5. Custom")
    print("   - Define your own frequencies")
    
    while True:
        choice = input("\nChoose composition (1-5): ")
        
        if choice == "1":
            pi = create_composition_vector("equal")
            comp_name = "Equal frequencies"
            break
        elif choice == "2":
            pi = create_composition_vector(custom=[0.1, 0.4, 0.4, 0.1])
            comp_name = "GC-rich"
            break
        elif choice == "3":
            pi = create_composition_vector(custom=[0.4, 0.1, 0.1, 0.4])
            comp_name = "AT-rich"
            break
        elif choice == "4":
            pi = create_composition_vector("exemplar")
            comp_name = "Example composition"
            break
        elif choice == "5":
            pi = interactive_composition_input()
            comp_name = "Custom"
            break
        else:
            print("Invalid choice. Please choose 1-5.")
    
    print(f"\nYou selected: {comp_name}")
    print(f"\nComposition vector π:")
    print(f"  A: {pi[0]:.3f}")
    print(f"  C: {pi[1]:.3f}")
    print(f"  G: {pi[2]:.3f}")
    print(f"  T: {pi[3]:.3f}")
    print(f"  Sum: {pi.sum():.3f} (should be 1.0)")
    
    # Visualize composition
    plt.figure(figsize=(8, 6))
    bases = ['A', 'C', 'G', 'T']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = plt.bar(bases, pi, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, pi):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, max(pi) * 1.2)
    plt.xlabel('Nucleotide', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Base Composition: {comp_name}', fontsize=14, pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return pi, comp_name

def step3_calculate_q_matrix_interactive(r_matrix, pi_vector):
    """Step 3: Interactive Q matrix calculation with detailed explanation."""
    clear_screen()
    print("="*70)
    print("STEP 3: CALCULATE THE Q MATRIX")
    print("="*70)
    print("\nThe Q matrix is the instantaneous rate matrix. It combines:")
    print("  • The rate matrix R (relative rates)")
    print("  • The composition vector π (base frequencies)")
    print("\nWe'll calculate this step by step!")
    
    pause_for_user()
    
    # Step 3.1: Calculate unscaled Q
    print("\n--- Step 3.1: Calculate unscaled Q ---")
    print("\nFirst, we multiply each column of R by the corresponding π value:")
    print("unscaledQ[i,j] = R[i,j] × π[j]")
    print("\nThis gives us the rate of substitution from i to j,")
    print("weighted by the frequency of the target nucleotide j.")
    
    pause_for_user("\nPress Enter to see the calculation...")
    
    unscaled_q = r_matrix * pi_vector
    
    print("\nR matrix:")
    show_matrix_with_explanation(r_matrix, "R", show_diagonal=False)
    print(f"\nπ vector: {pi_vector}")
    print("\nUnscaled Q = R × π:")
    show_matrix_with_explanation(unscaled_q, "Unscaled Q", show_diagonal=False)
    
    pause_for_user()
    
    # Step 3.2: Make rows sum to zero
    print("\n--- Step 3.2: Make rows sum to zero ---")
    print("\nFor a valid rate matrix, each row must sum to zero.")
    print("This ensures probability is conserved.")
    print("\nWe set diagonal elements to the negative of the row sum:")
    
    row_sums = unscaled_q.sum(axis=1)
    print(f"\nRow sums: {row_sums}")
    
    for i in range(4):
        unscaled_q[i][i] = -row_sums[i]
    
    print("\nQ matrix with adjusted diagonals:")
    show_matrix_with_explanation(unscaled_q, "Q (rows sum to 0)", show_diagonal=True)
    
    # Verify rows sum to zero
    new_row_sums = unscaled_q.sum(axis=1)
    print(f"\nNew row sums: {new_row_sums} (all should be ~0)")
    
    # Show an example row sum calculation
    print("\nExample - Row 1 (A) sums to:")
    print(f"  {unscaled_q[0,0]:.3f} + {unscaled_q[0,1]:.3f} + {unscaled_q[0,2]:.3f} + {unscaled_q[0,3]:.3f} = {new_row_sums[0]:.6f}")
    
    pause_for_user()
    
    # Step 3.3: Scale the matrix
    print("\n--- Step 3.3: Scale the Q matrix ---")
    print("\nWe scale Q so that the average substitution rate is 1.")
    print("This makes branch lengths interpretable as 'substitutions per site'.")
    
    print("\nTo find the scaling factor:")
    print("1. Create diagonal matrix of π: diag(π)")
    print("2. Calculate B = diag(π) × Q")
    print("3. Sum the off-diagonal elements of B")
    print("4. This sum is our scaling factor")
    
    pause_for_user("\nPress Enter to see the calculation...")
    
    diag_pi = np.diag(pi_vector)
    print("\nDiagonal matrix of π:")
    show_matrix_with_explanation(diag_pi, "diag(π)", show_diagonal=True)
    
    b = np.dot(diag_pi, unscaled_q)
    print("\nB = diag(π) × Q:")
    show_matrix_with_explanation(b, "B matrix", show_diagonal=True)
    
    # Calculate sum of off-diagonals
    diag_sum = sum(np.diagonal(b))
    total_sum = np.sum(b)
    off_diag_sum = total_sum - diag_sum
    
    print(f"\nSum of diagonal elements: {diag_sum:.6f}")
    print(f"Sum of all elements: {total_sum:.6f}")
    print(f"Sum of off-diagonal elements: {off_diag_sum:.6f}")
    print(f"\nScaling factor = {off_diag_sum:.6f}")
    
    # Scale the Q matrix
    scaled_q = unscaled_q / off_diag_sum
    
    print("\nFinal scaled Q matrix:")
    show_matrix_with_explanation(scaled_q, "Scaled Q", show_diagonal=True)
    
    print("\nThis Q matrix is now ready to use in the equation:")
    print("P(t) = e^(Qt)")
    print("where t is the branch length in substitutions per site.")
    
    pause_for_user()
    
    return scaled_q

def step4_calculate_p_matrix_interactive(scaled_q):
    """Step 4: Interactive P matrix calculation."""
    clear_screen()
    print("="*70)
    print("STEP 4: CALCULATE THE P MATRIX")
    print("="*70)
    print("\nThe P matrix contains the probabilities of change over time.")
    print("It's calculated using matrix exponentiation:")
    print("\n  P(t) = e^(Qt)")
    print("\nwhere:")
    print("  • Q is our scaled rate matrix")
    print("  • t is the branch length (time)")
    print("  • e^() is the matrix exponential")
    
    print("\nLet's explore how P changes with branch length!")
    
    while True:
        print("\nChoose a branch length to explore:")
        print("1. Very short (t = 0.01) - almost no change")
        print("2. Short (t = 0.1) - few changes")
        print("3. Moderate (t = 0.5) - moderate evolution")
        print("4. Long (t = 1.0) - substantial evolution")
        print("5. Very long (t = 5.0) - extensive evolution")
        print("6. Custom branch length")
        print("7. Continue to next step")
        
        choice = input("\nYour choice (1-7): ")
        
        if choice == "7":
            break
            
        if choice == "1":
            t = 0.01
        elif choice == "2":
            t = 0.1
        elif choice == "3":
            t = 0.5
        elif choice == "4":
            t = 1.0
        elif choice == "5":
            t = 5.0
        elif choice == "6":
            try:
                t = float(input("Enter branch length: "))
                if t < 0:
                    print("Branch length must be positive!")
                    continue
            except ValueError:
                print("Invalid input!")
                continue
        else:
            print("Invalid choice!")
            continue
        
        # Calculate P matrix
        p_matrix = linalg.expm(scaled_q * t)
        
        print(f"\nP matrix for branch length t = {t}:")
        show_matrix_with_explanation(p_matrix, f"P({t})")
        
        print("\nInterpretation:")
        print("- Each row shows probabilities of change FROM that nucleotide")
        print("- Each column shows probabilities of change TO that nucleotide")
        print("- Diagonal elements show probability of NO change")
        
        # Show example
        bases = ['A', 'C', 'G', 'T']
        print(f"\nExample: P[A→G] = {p_matrix[0,2]:.4f}")
        print(f"This means: Probability that A changes to G = {p_matrix[0,2]:.4f}")
        
        # Calculate average probability of change
        avg_no_change = np.mean(np.diag(p_matrix))
        avg_change = 1 - avg_no_change
        print(f"\nAverage probability of change: {avg_change:.4f}")
        print(f"Average probability of no change: {avg_no_change:.4f}")
        
        # Visualize
        visualize_matrix_interactive(p_matrix, f"P Matrix (t = {t})")
        
        pause_for_user()
    
    # Return a default P matrix for t=0.1
    return linalg.expm(scaled_q * 0.1), 0.1

def step5_input_sequences():
    """Step 5: Input DNA sequences for analysis."""
    clear_screen()
    print("="*70)
    print("STEP 5: INPUT DNA SEQUENCES")
    print("="*70)
    print("\nNow let's input two DNA sequences to analyze.")
    print("The sequences must be:")
    print("  • The same length")
    print("  • Contain only A, C, G, T")
    print("  • Aligned (corresponding positions are homologous)")
    
    print("\n1. Use example sequences")
    print("2. Input your own sequences")
    print("3. Generate random sequences")
    
    choice = input("\nYour choice (1-3): ")
    
    if choice == "1":
        seq1 = "ACGTACGTAC"
        seq2 = "ACTTACGCAC"
        print(f"\nUsing example sequences:")
    elif choice == "2":
        while True:
            seq1 = input("\nEnter sequence 1: ").upper()
            if not all(c in "ACGT" for c in seq1):
                print("Invalid sequence! Use only A, C, G, T.")
                continue
            
            seq2 = input("Enter sequence 2: ").upper()
            if not all(c in "ACGT" for c in seq2):
                print("Invalid sequence! Use only A, C, G, T.")
                continue
                
            if len(seq1) != len(seq2):
                print(f"Sequences must be same length! ({len(seq1)} vs {len(seq2)})")
                continue
                
            break
    else:  # Generate random
        length = 10
        bases = ['A', 'C', 'G', 'T']
        seq1 = ''.join(np.random.choice(bases, length))
        seq2 = seq1  # Start with identical
        # Introduce some random changes
        n_changes = np.random.randint(1, length//2)
        positions = np.random.choice(length, n_changes, replace=False)
        seq2_list = list(seq2)
        for pos in positions:
            old_base = seq2_list[pos]
            new_base = np.random.choice([b for b in bases if b != old_base])
            seq2_list[pos] = new_base
        seq2 = ''.join(seq2_list)
        print(f"\nGenerated random sequences with {n_changes} differences:")
    
    # Display alignment
    print(f"\nSequence alignment:")
    print(f"Seq1: {' '.join(seq1)}")
    print(f"      {' '.join(['|' if seq1[i] == seq2[i] else 'x' for i in range(len(seq1))])}")
    print(f"Seq2: {' '.join(seq2)}")
    
    # Count differences
    differences = sum(1 for i in range(len(seq1)) if seq1[i] != seq2[i])
    print(f"\nLength: {len(seq1)} nucleotides")
    print(f"Differences: {differences} ({differences/len(seq1)*100:.1f}%)")
    
    pause_for_user()
    
    return seq1, seq2

def step6_calculate_likelihood_interactive(p_matrix, pi_vector, seq1, seq2, branch_length):
    """Step 6: Interactive likelihood calculation."""
    clear_screen()
    print("="*70)
    print("STEP 6: CALCULATE LIKELIHOOD")
    print("="*70)
    print(f"\nWe'll calculate the likelihood of evolving from")
    print(f"Seq1: {seq1}")
    print(f"to")
    print(f"Seq2: {seq2}")
    print(f"with branch length t = {branch_length}")
    
    print("\nThe likelihood formula is:")
    print("L = ∏ π(base1) × P(base1→base2)")
    print("\nFor each position, we multiply:")
    print("  • The frequency of the starting base")
    print("  • The probability of change to the ending base")
    
    pause_for_user("\nLet's calculate step by step...")
    
    likelihood = 1.0
    
    print("\nPosition-by-position calculation:")
    print("-" * 60)
    print("Pos | Base1→Base2 | π(base1) | P(change) | Contribution")
    print("-" * 60)
    
    for i in range(len(seq1)):
        base1 = seq1[i]
        base2 = seq2[i]
        idx1 = {'A':0, 'C':1, 'G':2, 'T':3}[base1]
        idx2 = {'A':0, 'C':1, 'G':2, 'T':3}[base2]
        
        pi_val = pi_vector[idx1]
        p_val = p_matrix[idx1, idx2]
        contribution = pi_val * p_val
        
        likelihood *= contribution
        
        change_symbol = "→" if base1 != base2 else "="
        print(f"{i+1:3} | {base1} {change_symbol} {base2}      | {pi_val:.4f}   | {p_val:.4f}    | {contribution:.6f}")
        
        if (i + 1) % 5 == 0 and i < len(seq1) - 1:
            pause_for_user(f"Current likelihood: {likelihood:.8e}. Press Enter...")
    
    print("-" * 60)
    print(f"\nFinal likelihood: L = {likelihood:.8e}")
    
    # Convert to log likelihood
    log_likelihood = np.log(likelihood)
    print(f"Log likelihood: ln(L) = {log_likelihood:.4f}")
    
    print("\nInterpretation:")
    if likelihood > 0.01:
        print("- Relatively high likelihood")
        print("- Sequences are similar and branch length is appropriate")
    elif likelihood > 0.0001:
        print("- Moderate likelihood")
        print("- Sequences have some differences or branch length is reasonable")
    else:
        print("- Low likelihood")
        print("- Either sequences are very different or branch length is inappropriate")
    
    pause_for_user()
    
    return likelihood

def step7_explore_branch_lengths(scaled_q, pi_vector, seq1, seq2):
    """Step 7: Explore how likelihood changes with branch length."""
    clear_screen()
    print("="*70)
    print("STEP 7: EXPLORE BRANCH LENGTHS")
    print("="*70)
    print("\nLet's see how likelihood changes with different branch lengths!")
    print("This will help us find the optimal branch length.")
    
    while True:
        print("\n1. Calculate likelihood for specific branch length")
        print("2. Plot likelihood curve")
        print("3. Find optimal branch length")
        print("4. Compare multiple branch lengths")
        print("5. Continue to next step")
        
        choice = input("\nYour choice (1-5): ")
        
        if choice == "5":
            break
            
        elif choice == "1":
            try:
                t = float(input("\nEnter branch length: "))
                if t < 0:
                    print("Branch length must be positive!")
                    continue
                    
                p_mat = linalg.expm(scaled_q * t)
                likelihood = calculate_likelihood_simple(p_mat, pi_vector, seq1, seq2)
                log_like = np.log(likelihood) if likelihood > 0 else float('-inf')
                
                print(f"\nBranch length: {t}")
                print(f"Likelihood: {likelihood:.8e}")
                print(f"Log likelihood: {log_like:.4f}")
                
            except ValueError:
                print("Invalid input!")
                
        elif choice == "2":
            print("\nGenerating likelihood curve...")
            branch_lengths = np.linspace(0.001, 2.0, 200)
            likelihoods = []
            
            for t in branch_lengths:
                p_mat = linalg.expm(scaled_q * t)
                like = calculate_likelihood_simple(p_mat, pi_vector, seq1, seq2)
                likelihoods.append(like)
            
            plt.figure(figsize=(10, 6))
            plt.plot(branch_lengths, likelihoods, 'b-', linewidth=2)
            plt.xlabel('Branch Length (substitutions/site)', fontsize=12)
            plt.ylabel('Likelihood', fontsize=12)
            plt.title('Likelihood vs Branch Length', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Mark maximum
            max_idx = np.argmax(likelihoods)
            max_t = branch_lengths[max_idx]
            max_like = likelihoods[max_idx]
            plt.scatter([max_t], [max_like], color='red', s=100, zorder=5)
            plt.annotate(f'Max at t={max_t:.3f}', 
                        xy=(max_t, max_like), 
                        xytext=(max_t+0.1, max_like),
                        arrowprops=dict(arrowstyle='->', color='red'))
            
            plt.show()
            
        elif choice == "3":
            print("\nFinding optimal branch length...")
            
            # Coarse search
            coarse_t = np.linspace(0.001, 2.0, 50)
            coarse_likes = []
            
            for t in coarse_t:
                p_mat = linalg.expm(scaled_q * t)
                like = calculate_likelihood_simple(p_mat, pi_vector, seq1, seq2)
                coarse_likes.append(like)
            
            # Find approximate maximum
            max_idx = np.argmax(coarse_likes)
            approx_optimal = coarse_t[max_idx]
            
            # Fine search around maximum
            fine_t = np.linspace(max(0.001, approx_optimal - 0.1), 
                                approx_optimal + 0.1, 100)
            fine_likes = []
            
            for t in fine_t:
                p_mat = linalg.expm(scaled_q * t)
                like = calculate_likelihood_simple(p_mat, pi_vector, seq1, seq2)
                fine_likes.append(like)
            
            # Find optimal
            max_idx = np.argmax(fine_likes)
            optimal_t = fine_t[max_idx]
            max_likelihood = fine_likes[max_idx]
            
            print(f"\nOptimal branch length: {optimal_t:.4f}")
            print(f"Maximum likelihood: {max_likelihood:.8e}")
            print(f"Log likelihood: {np.log(max_likelihood):.4f}")
            
            # Show observed vs expected differences
            observed_diff = sum(1 for i in range(len(seq1)) if seq1[i] != seq2[i])
            observed_prop = observed_diff / len(seq1)
            
            print(f"\nObserved differences: {observed_diff}/{len(seq1)} = {observed_prop:.3f}")
            print(f"This corresponds to ~{optimal_t:.3f} substitutions per site")
            print("(accounting for multiple substitutions at same sites)")
            
        elif choice == "4":
            branch_lengths = [0.01, 0.1, 0.5, 1.0, 2.0]
            
            print("\nComparing branch lengths:")
            print("-" * 50)
            print("Branch Length | Likelihood    | Log Likelihood")
            print("-" * 50)
            
            for t in branch_lengths:
                p_mat = linalg.expm(scaled_q * t)
                like = calculate_likelihood_simple(p_mat, pi_vector, seq1, seq2)
                log_like = np.log(like) if like > 0 else float('-inf')
                print(f"{t:13.2f} | {like:13.6e} | {log_like:14.4f}")
            
            print("-" * 50)
        
        pause_for_user()
    
    return

def calculate_likelihood_simple(p_matrix, pi_vector, seq1, seq2):
    """Simple likelihood calculation without display."""
    likelihood = 1.0
    
    for i in range(len(seq1)):
        idx1 = {'A':0, 'C':1, 'G':2, 'T':3}[seq1[i]]
        idx2 = {'A':0, 'C':1, 'G':2, 'T':3}[seq2[i]]
        likelihood *= pi_vector[idx1] * p_matrix[idx1][idx2]
    
    return likelihood

def interactive_matrix_input():
    """Allow user to input custom rate matrix values."""
    print("\nEnter rate matrix values (relative rates):")
    print("Note: Diagonal values will be set automatically")
    
    base_names = ['A', 'C', 'G', 'T']
    R = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(4):
            if i != j:
                while True:
                    try:
                        R[i, j] = float(input(f"Rate {base_names[i]}→{base_names[j]}: "))
                        if R[i, j] < 0:
                            print("Rates must be non-negative!")
                            continue
                        break
                    except ValueError:
                        print("Invalid input! Please enter a number.")
    
    return R

def interactive_composition_input():
    """Allow user to input custom composition vector values."""
    print("\nEnter base frequencies (must sum to 1.0):")
    base_names = ['A', 'C', 'G', 'T']
    
    while True:
        Pi = np.zeros(4)
        
        for i in range(4):
            while True:
                try:
                    Pi[i] = float(input(f"Frequency of {base_names[i]}: "))
                    if Pi[i] < 0 or Pi[i] > 1:
                        print("Frequencies must be between 0 and 1!")
                        continue
                    break
                except ValueError:
                    print("Invalid input! Please enter a number.")
        
        total = np.sum(Pi)
        if abs(total - 1.0) > 0.001:
            print(f"\nFrequencies sum to {total}, not 1.0!")
            normalize = input("Normalize to sum to 1.0? (y/n): ")
            if normalize.lower() == 'y':
                Pi = Pi / total
                print(f"Normalized frequencies: {Pi}")
                break
        else:
            break
    
    return Pi

def run_guided_tutorial():
    """Run the complete guided interactive tutorial."""
    display_welcome()
    
    # Step 1: Choose rate matrix
    r_matrix, model_name = step1_choose_rate_matrix()
    
    # Step 2: Choose composition
    pi_vector, comp_name = step2_choose_composition()
    
    # Step 3: Calculate Q matrix
    scaled_q = step3_calculate_q_matrix_interactive(r_matrix, pi_vector)
    
    # Step 4: Calculate P matrix
    p_matrix, branch_length = step4_calculate_p_matrix_interactive(scaled_q)
    
    # Step 5: Input sequences
    seq1, seq2 = step5_input_sequences()
    
    # Step 6: Calculate likelihood
    likelihood = step6_calculate_likelihood_interactive(p_matrix, pi_vector, seq1, seq2, branch_length)
    
    # Step 7: Explore branch lengths
    step7_explore_branch_lengths(scaled_q, pi_vector, seq1, seq2)
    
    # Summary
    clear_screen()
    print("="*70)
    print("TUTORIAL COMPLETE!")
    print("="*70)
    print("\nYou've learned how to:")
    print("✓ Choose and understand rate matrices")
    print("✓ Set base composition vectors")
    print("✓ Calculate Q matrices (instantaneous rates)")
    print("✓ Calculate P matrices (probabilities over time)")
    print("✓ Calculate likelihood of sequence evolution")
    print("✓ Find optimal branch lengths")
    
    print("\nKey concepts to remember:")
    print("• Rate matrices define relative substitution rates")
    print("• Q matrices combine rates with base frequencies")
    print("• P matrices give probabilities over evolutionary time")
    print("• Likelihood measures how well a model explains the data")
    print("• Maximum likelihood finds the best model parameters")
    
    print("\nThank you for using the DNA Likelihood Calculator!")
    pause_for_user("\nPress Enter to return to main menu...")

def quick_calculation_mode():
    """Quick mode for experienced users."""
    print("\nQuick Calculation Mode")
    print("="*50)
    
    # Quick selections
    print("\nSelect model: 1=JC, 2=K2P, 3=HKY85, 4=GTR")
    model_choice = input("Model (1-4): ")
    
    models = {"1": "jukes_cantor", "2": "kimura2p", "3": "hky85", "4": "gtr"}
    r = create_rate_matrix(models.get(model_choice, "jukes_cantor"))
    
    print("\nSelect composition: 1=Equal, 2=Custom")
    comp_choice = input("Composition (1-2): ")
    
    if comp_choice == "2":
        pi = interactive_composition_input()
    else:
        pi = create_composition_vector("equal")
    
    # Input sequences
    seq1 = input("\nSequence 1: ").upper()
    seq2 = input("Sequence 2: ").upper()
    
    # Calculate
    scaled_q = calculate_scaled_q_simple(r, pi)
    
    # Find optimal branch length
    print("\nFinding optimal branch length...")
    branch_lengths = np.linspace(0.001, 2.0, 100)
    likelihoods = []
    
    for t in branch_lengths:
        p_mat = linalg.expm(scaled_q * t)
        like = calculate_likelihood_simple(p_mat, pi, seq1, seq2)
        likelihoods.append(like)
    
    max_idx = np.argmax(likelihoods)
    optimal_t = branch_lengths[max_idx]
    max_likelihood = likelihoods[max_idx]
    
    print(f"\nResults:")
    print(f"Optimal branch length: {optimal_t:.4f}")
    print(f"Maximum likelihood: {max_likelihood:.8e}")
    print(f"Log likelihood: {np.log(max_likelihood):.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(branch_lengths, likelihoods, 'b-', linewidth=2)
    plt.scatter([optimal_t], [max_likelihood], color='red', s=100)
    plt.xlabel('Branch Length')
    plt.ylabel('Likelihood')
    plt.title('Likelihood vs Branch Length')
    plt.grid(True, alpha=0.3)
    plt.show()

def calculate_scaled_q_simple(r_matrix, pi_vector):
    """Simple Q matrix calculation without display."""
    unscaled_q = r_matrix * pi_vector
    row_sums = unscaled_q.sum(axis=1)
    
    for i in range(4):
        unscaled_q[i][i] = -row_sums[i]
    
    diag_pi = np.diag(pi_vector)
    b = np.dot(diag_pi, unscaled_q)
    
    diag_sum = sum(np.diagonal(b))
    off_diag_sum = np.sum(b) - diag_sum
    
    scaled_q = unscaled_q / off_diag_sum
    
    return scaled_q

def main():
    """Main program loop."""
    while True:
        clear_screen()
        print("="*70)
        print("   DNA LIKELIHOOD CALCULATOR - INTERACTIVE EDUCATIONAL TOOL")
        print("="*70)
        print("\n1. Guided Interactive Tutorial (Recommended for first time)")
        print("2. Quick Calculation Mode")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ")
        
        if choice == "1":
            run_guided_tutorial()
        elif choice == "2":
            quick_calculation_mode()
            pause_for_user("\nPress Enter to continue...")
        elif choice == "3":
            print("\nThank you for using the DNA Likelihood Calculator!")
            break
        else:
            print("Invalid choice. Please try again.")
            pause_for_user()

if __name__ == "__main__":
    main()