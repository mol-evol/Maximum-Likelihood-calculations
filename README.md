# DNA Likelihood Calculator

An interactive educational tool for understanding and calculating the likelihood of DNA sequence evolution using various substitution models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Interactive Tutorial Mode](#interactive-tutorial-mode)
  - [Standard Mode](#standard-mode)
  - [Quick Calculation Mode](#quick-calculation-mode)
- [Mathematical Background](#mathematical-background)
- [Evolutionary Models](#evolutionary-models)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

The DNA Likelihood Calculator is a Python-based educational tool designed to teach the mathematical principles behind evolutionary sequence analysis. It provides step-by-step demonstrations of how to calculate the likelihood of DNA sequence evolution, making complex phylogenetic concepts accessible to students and researchers.

### Key Educational Goals

- Understand rate matrices and how they model substitution processes
- Learn about base composition and its effects on evolution
- Master Q matrix calculations and scaling
- Explore probability matrices through time
- Calculate and interpret likelihood values
- Find optimal branch lengths using maximum likelihood

## Features

### Core Functionality

- **Interactive Learning**: Step-by-step explanations with pauses for understanding
- **Multiple Evolutionary Models**: Jukes-Cantor, Kimura 2-parameter, HKY85, and GTR
- **Visualizations**: Interactive plots and heatmaps for matrices and likelihood curves
- **Flexible Input**: Manual sequence entry, FASTA format, or random generation
- **Model Comparison**: Compare different models on the same data
- **Bootstrap Analysis**: Estimate confidence intervals for parameters
- **Educational Modes**: Tutorial, example, and quick calculation modes

### Technical Features

- Matrix caching for performance
- Progress indicators for long calculations
- Input validation and error handling
- Unit tests for code verification
- Extensible architecture for new models

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/dna-likelihood-calculator.git
cd dna-likelihood-calculator
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy scipy matplotlib
```

### Verify Installation

```bash
python showLikelihoodCalcs.py
```

You should see the main menu of the DNA Likelihood Calculator.

## Quick Start

### Running the Program

```bash
python showLikelihoodCalcs.py
```

This launches the main program with two options:
1. **Guided Interactive Tutorial** (Recommended for first-time users) - walks you through each step with detailed explanations
2. **Quick Calculation Mode** - for experienced users who want fast results

## Usage Guide

### Interactive Tutorial Mode

The interactive tutorial (Option 1 in the main menu) provides the most comprehensive learning experience:

1. **Welcome Screen**: Introduction to concepts
2. **Step 1 - Rate Matrix**: Choose and visualize substitution models
3. **Step 2 - Composition**: Set nucleotide frequencies
4. **Step 3 - Q Matrix**: Detailed calculation with substeps
5. **Step 4 - P Matrix**: Explore probabilities over time
6. **Step 5 - Sequences**: Input DNA sequences
7. **Step 6 - Likelihood**: Position-by-position calculation
8. **Step 7 - Optimization**: Find optimal branch lengths

Each step includes:
- Clear explanations of the mathematics
- Visual representations
- Interactive elements
- Verification of calculations

### Additional Features

Beyond the interactive tutorial, you can access the original features by modifying the main() function to call the original menu. The program includes:

#### 1. Example Calculation
Shows a complete workflow with pre-set parameters:
```python
# Uses exemplar rate matrix and composition
# Sequences: CCAT → CCGT
# Multiple branch lengths explored
```

#### 2. Interactive Mode
Customize all parameters:
- Choose rate matrix (or input custom)
- Set base composition
- Enter your sequences
- Select branch lengths to test

#### 3. Model Comparison
Compare different evolutionary models:
- Tests JC, K2P, HKY85, and GTR
- Finds optimal branch length for each
- Shows AIC values for model selection

#### 4. Bootstrap Analysis
Estimate parameter uncertainty:
- Resamples alignment columns
- Calculates confidence intervals
- Visualizes bootstrap distribution

### Quick Calculation Mode

For experienced users who want fast results:
- Select option 2 from the main menu
- Choose model and parameters quickly
- Get results without detailed explanations

## Mathematical Background

### The Likelihood Calculation Process

1. **Rate Matrix (R)**
   - Defines relative substitution rates
   - Off-diagonal elements only
   - Symmetric for reversible models

2. **Composition Vector (π)**
   - Equilibrium base frequencies
   - Must sum to 1.0
   - Affects substitution probabilities

3. **Q Matrix Calculation**
   ```
   Q_ij = R_ij × π_j  (for i ≠ j)
   Q_ii = -Σ(Q_ij)    (diagonal elements)
   ```
   Then scale so average rate = 1

4. **P Matrix Calculation**
   ```
   P(t) = e^(Qt)
   ```
   Where t is branch length in substitutions/site

5. **Likelihood Calculation**
   ```
   L = Π π(i) × P(i→j)
   ```
   Product over all sites in the alignment

### Key Concepts

- **Branch Length**: Expected substitutions per site
- **Saturation**: Multiple substitutions at same site
- **Maximum Likelihood**: Find parameters that maximize L
- **Log Likelihood**: Use ln(L) for numerical stability

## Evolutionary Models

### Jukes-Cantor (JC)
- Simplest model
- Equal base frequencies (0.25 each)
- All substitutions equally likely
- 1 parameter (branch length only)

### Kimura 2-Parameter (K2P)
- Distinguishes transitions from transversions
- Transitions (A↔G, C↔T) typically more frequent
- Equal base frequencies
- 2 parameters

### HKY85
- Combines unequal base frequencies with transition/transversion ratio
- More realistic for most sequences
- 5 parameters

### GTR (General Time Reversible)
- Most general reversible model
- Each substitution type has its own rate
- Unequal base frequencies
- 9 parameters

## Examples

### Example 1: Basic Likelihood Calculation

```python
from showLikelihoodCalcs import *

# Create rate matrix (Jukes-Cantor)
r = create_rate_matrix("jukes_cantor")

# Set equal base frequencies
pi = create_composition_vector("equal")

# Calculate Q matrix
q = calculate_scaled_q(r, pi, explain=False)

# Calculate P matrix for branch length 0.1
p = linalg.expm(q * 0.1)

# Calculate likelihood
seq1 = "ACGT"
seq2 = "ACTT"
likelihood = calculate_likelihood(p, pi, seq1, seq2)
print(f"Likelihood: {likelihood}")
```

### Example 2: Finding Optimal Branch Length

```python
# Define sequences
seq1 = "ACGTACGTAC"
seq2 = "ACTTACGCAC"

# Test range of branch lengths
branch_lengths = np.linspace(0.01, 1.0, 100)
likelihoods = []

for t in branch_lengths:
    p = linalg.expm(q * t)
    like = calculate_likelihood(p, pi, seq1, seq2)
    likelihoods.append(like)

# Find maximum
max_idx = np.argmax(likelihoods)
optimal_t = branch_lengths[max_idx]
print(f"Optimal branch length: {optimal_t:.4f}")
```

### Example 3: Model Comparison

```python
models = ["jukes_cantor", "kimura2p", "hky85"]
results = {}

for model in models:
    r = create_rate_matrix(model)
    q = calculate_scaled_q(r, pi, explain=False)
    
    # Find optimal branch length
    # ... (optimization code)
    
    results[model] = {
        'likelihood': max_likelihood,
        'branch_length': optimal_t,
        'AIC': calculate_aic(max_likelihood, param_count[model])
    }
```

## API Reference

### Core Functions

The main program (`showLikelihoodCalcs.py`) provides several key functions that can be imported and used programmatically:

#### `create_rate_matrix(matrix_type)`
Create a rate matrix of specified type.

**Parameters:**
- `matrix_type` (str): One of "jukes_cantor", "kimura2p", "hky85", "gtr", "exemplar"

**Returns:**
- `numpy.ndarray`: 4×4 rate matrix

#### `create_composition_vector(vector_type, custom=None)`
Create a base composition vector.

**Parameters:**
- `vector_type` (str): "equal" or "exemplar"
- `custom` (list, optional): Custom frequencies [A, C, G, T]

**Returns:**
- `numpy.ndarray`: Composition vector

#### `calculate_scaled_q_simple(r_matrix, pi_vector)`
Calculate scaled Q matrix from R and π without display.

**Parameters:**
- `r_matrix` (numpy.ndarray): Rate matrix
- `pi_vector` (numpy.ndarray): Composition vector

**Returns:**
- `numpy.ndarray`: Scaled Q matrix

#### `calculate_likelihood_simple(p_matrix, pi_vector, seq1, seq2)`
Calculate likelihood of sequence evolution.

**Parameters:**
- `p_matrix` (numpy.ndarray): Probability matrix
- `pi_vector` (numpy.ndarray): Composition vector
- `seq1` (str): First sequence
- `seq2` (str): Second sequence

**Returns:**
- `float`: Likelihood value

### Interactive Functions

- `step1_choose_rate_matrix()`: Interactive rate matrix selection
- `step2_choose_composition()`: Interactive composition vector selection
- `step3_calculate_q_matrix_interactive()`: Step-by-step Q matrix calculation
- `step4_calculate_p_matrix_interactive()`: Interactive P matrix exploration
- `step5_input_sequences()`: Sequence input interface
- `step6_calculate_likelihood_interactive()`: Detailed likelihood calculation
- `step7_explore_branch_lengths()`: Branch length optimization tools

## Troubleshooting

### Common Issues

#### ImportError: No module named 'numpy'
**Solution:** Install required packages:
```bash
pip install numpy scipy matplotlib
```

#### Memory Error on Large Datasets
**Solution:** This tool is designed for educational purposes with small datasets. For production phylogenetic analysis, use specialized software like IQ-TREE or RAxML.

#### Plots Not Displaying
**Solution:** 
- Ensure matplotlib backend is properly configured
- On remote systems, use `plt.savefig()` instead of `plt.show()`
- Check X11 forwarding for SSH connections

#### Negative Likelihood Values
**Issue:** Likelihood should always be positive.
**Check:**
- Sequence alignment is correct
- No invalid characters in sequences
- Rate matrix and composition vector are properly normalized

### Platform-Specific Notes

**macOS:**
- May need to install XCode command line tools
- Use `python3` instead of `python` if needed

**Windows:**
- Use `python` instead of `python3`
- Paths use backslashes: `venv\Scripts\activate`

**Linux:**
- May need to install python3-tk for GUI: `sudo apt-get install python3-tk`

## Contributing

We welcome contributions to improve the educational value of this tool!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -am 'Add new model'`)
6. Push to the branch (`git push origin feature/new-model`)
7. Create a Pull Request

### Contribution Guidelines

- Focus on educational clarity over performance
- Add documentation for new features
- Include examples in docstrings
- Follow existing code style
- Test on Python 3.6+

### Areas for Contribution

- Additional evolutionary models
- New visualization types
- Interactive web interface
- Additional statistical analyses
- Translation to other languages
- More example datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your teaching or research, please cite:

```bibtex
@software{dna_likelihood_calculator,
  author = {McInerney, James O.},
  title = {DNA Likelihood Calculator: An Educational Tool for Phylogenetic Analysis},
  year = {2024},
  url = {https://github.com/mol-evol/dna-likelihood-calculator}
}
```

## Acknowledgments

- Inspired by the need for clear, interactive teaching tools in molecular evolution
- Built with NumPy, SciPy, and Matplotlib
- Thanks to all contributors and testers

## Contact

For questions, suggestions, or bug reports, please open an issue on GitHub or contact the maintainers.

---

**Note:** This tool is designed for educational purposes. For production phylogenetic analyses, please use specialized software such as IQ-TREE, RAxML, PAML, or MEGA.