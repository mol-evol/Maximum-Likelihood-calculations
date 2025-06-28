# DNA Likelihood Calculator

An interactive educational tool for understanding the mathematical principles of DNA sequence evolution and likelihood calculations in phylogenetics.

## Overview

This program provides a step-by-step, interactive tutorial that guides users through the process of calculating the likelihood of DNA sequence evolution. It's designed for students and researchers learning phylogenetic methods, offering both detailed educational mode and quick calculation options.

## Features

- **Interactive Tutorial Mode**: Step-by-step guidance through the entire likelihood calculation process
- **Quick Calculation Mode**: Streamlined interface for experienced users
- **Multiple Evolutionary Models**: Supports Jukes-Cantor, Kimura 2-parameter, HKY85, and GTR models
- **Visual Learning**: Matplotlib visualizations of matrices and likelihood curves
- **Custom Parameters**: Option to input custom rate matrices and base compositions
- **Real-time Calculations**: See how changing parameters affects results

## Installation

### Prerequisites

- Python 3.6 or higher
- NumPy
- SciPy  
- Matplotlib

### Setup

1. Clone or download this repository:
```bash
git clone https://github.com/mol-evol/Maximum-Likelihood-calculations
cd Maximum-Likelihood-calculations
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy scipy matplotlib
```

## Usage

Run the main program:
```bash
python showLikelihoodCalcs.py
```

You'll see two main options:

### 1. Guided Interactive Tutorial (Recommended)

This mode walks you through seven comprehensive steps:

**Step 1: Choose Rate Matrix**
- Select from predefined models (JC, K2P, HKY85, GTR)
- Option to input custom rate matrix
- Visual representation of your selected matrix

**Step 2: Choose Base Composition**
- Select equilibrium nucleotide frequencies
- Options: equal, GC-rich, AT-rich, or custom
- Bar chart visualization of composition

**Step 3: Calculate Q Matrix**
- Detailed substep-by-substep calculation
- Shows unscaled Q, row sum adjustment, and scaling
- Verifies that rows sum to zero

**Step 4: Calculate P Matrix**
- Explore probability matrices at different time points
- Interactive selection of branch lengths
- See how probabilities change over evolutionary time

**Step 5: Input DNA Sequences**
- Use example sequences
- Input your own aligned sequences
- Generate random sequences with specified differences

**Step 6: Calculate Likelihood**
- Position-by-position likelihood calculation
- Shows contribution of each site
- Converts to log-likelihood for interpretation

**Step 7: Explore Branch Lengths**
- Calculate likelihood for specific branch lengths
- Plot likelihood curves
- Find optimal branch length automatically
- Compare multiple branch lengths

### 2. Quick Calculation Mode

For experienced users:
- Rapid model and parameter selection
- Direct sequence input
- Automatic optimal branch length calculation
- Results with visualization

## Mathematical Background

The program implements the standard likelihood calculation for DNA sequence evolution:

1. **Rate Matrix (R)**: Defines relative substitution rates between nucleotides
2. **Composition Vector (π)**: Equilibrium base frequencies
3. **Q Matrix**: Instantaneous rate matrix calculated as Q = R × π, scaled for unit substitution rate
4. **P Matrix**: Transition probability matrix P(t) = e^(Qt)
5. **Likelihood**: L = ∏ π(i) × P(i→j) over all sites

## Evolutionary Models

### Jukes-Cantor (JC)
- Simplest model with equal substitution rates
- 1 parameter (branch length only)

### Kimura 2-Parameter (K2P)  
- Distinguishes transitions from transversions
- 2 parameters

### HKY85
- Unequal base frequencies + transition/transversion bias
- 5 parameters

### GTR (General Time Reversible)
- Most general model with all rates different
- 9 parameters

## Example Session

```python
# From the tutorial, you'll see calculations like:

Rate Matrix R:
--------------------------------------------------
         A        C        G        T    
     ---------------------------------------------
  A |   ---     1.000   2.000   1.000 
  C |  1.000    ---     1.000   2.000 
  G |  2.000   1.000    ---     1.000 
  T |  1.000   2.000   1.000    ---   

Composition vector π:
  A: 0.250
  C: 0.250
  G: 0.250
  T: 0.250

# The program then shows Q matrix calculation, P matrix for chosen branch length,
# and step-by-step likelihood calculation for your sequences
```

## Educational Value

This tool is designed for:
- **Learning**: Understand each step of likelihood calculation
- **Teaching**: Demonstrate phylogenetic concepts interactively  
- **Experimentation**: See how different models and parameters affect results
- **Verification**: Check hand calculations against computed results


## Tips for Users

1. Start with the Guided Tutorial to understand the process
2. Try different evolutionary models on the same sequences
3. Experiment with branch lengths to see likelihood curves
4. Use custom matrices to explore parameter effects
5. Compare results with published phylogenetic software

## Limitations

- Designed for educational purposes, not large-scale analyses
- Single site model (no rate heterogeneity across sites)
- No tree topology optimization (single branch only)
- For production analyses, use IQ-TREE, RAxML, or similar

## Troubleshooting

**Import errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Display issues**: If plots don't appear, check your matplotlib backend
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

**Memory errors**: This tool is for small educational examples. Large alignments need specialized software.

## Contributing

Contributions welcome! Areas for improvement:
- Additional evolutionary models
- Rate heterogeneity across sites
- Tree visualization
- More statistical tests
- Performance optimizations

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in teaching or research:

```
DNA Likelihood Calculator: An Interactive Educational Tool for Phylogenetics
https://github.com/[your-username]/dna-likelihood-calculator
```

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Remember**: This is an educational tool. For research-grade phylogenetic analyses, use established software packages like IQ-TREE, RAxML, MEGA, or PAML.