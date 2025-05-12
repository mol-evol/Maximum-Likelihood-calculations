# Maximum Likelihood Model Calculator

## Overview

The DNA Likelihood Calculator is an educational tool designed to teach the mathematical principles behind evolutionary sequence analysis. This Python-based application demonstrates how to calculate the likelihood of DNA sequence evolution using various evolutionary models, helping students understand the core mathematical foundations of modern phylogenetics.

![Program Header](https://img.shields.io/badge/Educational%20Tool-Evolutionary%20Analysis-blue) [![DOI](https://zenodo.org/badge/435476325.svg)](https://doi.org/10.5281/zenodo.15390364)  ![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![Version](https://img.shields.io/badge/version-1.0.0-orange.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Features

- **Interactive Learning**: Step-by-step explanations of each calculation stage
- **Multiple Evolutionary Models**: Implementation of Jukes-Cantor, Kimura 2-parameter, HKY85, and GTR models
- **Visual Representations**: Comprehensive visualizations of matrices, likelihood surfaces, and model comparisons
- **Statistical Analysis**: Bootstrap confidence intervals and model selection using AIC
- **Educational Components**: Detailed tutorials and parameter interpretation guides
- **Sequence Support**: Manual input or FASTA format for DNA sequences
- **Interactive Mode**: Customize parameters and see real-time results

## Installation

### Prerequisites

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/mol-evol/Maximum-likelihood-calculations.git
cd Maximum-likelihood-calculations

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib
```

## Usage

Run the main script to start the program:

```bash
python showLikelihoodCalcs.py
```

### Main Menu Options

1. **Run Example Calculation**: Demonstrates a complete likelihood calculation with detailed explanations
2. **Run Interactive Mode**: Input your own sequences and parameters to analyze
3. **Run Unit Tests**: Verify the correct functioning of the code
4. **Compare Evolutionary Models**: Compare different models on the same sequence data
5. **Interactive Tutorial**: A guided step-by-step tutorial explaining evolutionary models
6. **Bootstrap Analysis**: Estimate confidence intervals for branch lengths
7. **Parameter Interpretation Guide**: Explains the biological meaning of model parameters

## Educational Components

### Evolutionary Models

The program implements several nucleotide substitution models:

- **Jukes-Cantor**: The simplest model, assuming equal base frequencies and substitution rates
- **Kimura 2-parameter**: Distinguishes between transitions and transversions
- **HKY85**: Combines unequal base frequencies with transition/transversion bias
- **GTR (General Time Reversible)**: The most general model with separate rates for each substitution type

### Mathematical Foundation

The program demonstrates these key concepts:

1. **Rate Matrix Construction**: Building relative rate matrices (R) for different evolutionary models
2. **Composition Vectors**: Creating and interpreting base frequency vectors (π)
3. **Q Matrix Calculation**: Combining R and π to form the instantaneous rate matrix (Q)
4. **P Matrix Derivation**: Using matrix exponentiation to calculate probability matrices (P = e^Qv)
5. **Likelihood Calculation**: Computing the likelihood of sequence evolution given a model
6. **Optimization**: Finding the maximum likelihood branch length
7. **Model Selection**: Using AIC to compare models of different complexity

## Example Usage

### Basic Example

```python
# Create a rate matrix (Jukes-Cantor model)
r = create_rate_matrix("jukes_cantor")

# Create a composition vector (equal frequencies)
pi = create_composition_vector("equal")

# Calculate the scaled Q matrix
scaled_q = calculate_scaled_q(r, pi)

# Calculate P matrix for a branch length of 0.1
p_matrix = linalg.expm(scaled_q * 0.1)

# Calculate likelihood for two sequences
seq1 = "ACGT"
seq2 = "ACTT"
likelihood = calculate_likelihood(p_matrix, pi, seq1, seq2)

print(f"Likelihood: {likelihood:.7f}")
```

### Interactive Analysis

The interactive mode allows you to:

1. Select or input custom rate matrices
2. Choose or specify base frequency compositions
3. Enter your own DNA sequences
4. Compare different evolutionary models
5. Visualize likelihood surfaces
6. Perform bootstrap analysis for confidence intervals

## Visualizations

The program provides multiple visualizations to aid understanding:

- **Rate Matrix Heatmaps**: Visual representation of substitution rates
- **P Matrix Visualizations**: Probability matrices for different branch lengths
- **Likelihood Curves**: Plots of likelihood vs. branch length
- **Bootstrap Distributions**: Histograms of bootstrap replicates with confidence intervals
- **Model Comparisons**: Comparative plots of different evolutionary models
- **Parameter Effects**: Visualizations showing how parameter choices affect results


## Mathematical Background

### Likelihood Calculation

The likelihood of evolving from sequence 1 to sequence 2 is calculated as:

$$L(D|M) = \prod_{i=1}^{n} \pi_{s1_i} \cdot P_{s1_i,s2_i}(v)$$

Where:
- $\pi_{s1_i}$ is the frequency of nucleotide $s1_i$ in sequence 1
- $P_{s1_i,s2_i}(v)$ is the probability of nucleotide $s1_i$ changing to $s2_i$ over branch length $v$
- $n$ is the sequence length

### P Matrix Calculation

$$P(v) = e^{Qv}$$

Where:
- $Q$ is the scaled instantaneous rate matrix
- $v$ is the branch length in substitutions per site
- $e^{Qv}$ is the matrix exponential

## Educational Use Cases

This program is particularly useful for:

- **Undergraduate Courses**: Introduction to computational biology and molecular evolution
- **Graduate Courses**: Advanced phylogenetic methods and statistical modeling
- **Self-Study**: Learning the mathematical foundations of evolutionary analysis
- **Teaching**: Demonstrating complex concepts with interactive examples

## Contributing

Contributions to improve the educational value of this program are welcome:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -am 'Add some amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This program was inspired by the need for clear, educational examples of likelihood calculations in evolutionary biology
- Many thanks to the developers of NumPy, SciPy, and Matplotlib for their excellent scientific computing tools
- Special acknowledgment to the pioneering work of Joe Felsenstein, Motoo Kimura, and other evolutionary biologists who developed these models

## Citation

If you use this program in your teaching or research, please cite:


> **McInerney, J.O. (2025)**. Maximum Likelihood Model Calculator: An Educational Tool for Understanding Phylogenetic Analysis. GitHub repository, https://github.com/mol-evol/dna-likelihood-calculator. DOI: [10.5281/zenodo.15390364](https://doi.org/10.5281/zenodo.15390364)

*This program is designed for educational purposes and is not optimized for large-scale phylogenetic analyses. For production research, please consider specialized software like PAML, IQ-TREE, or RAxML.*
