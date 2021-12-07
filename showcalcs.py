import numpy as np
from scipy import linalg
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


print ('''
  _     _  _         _  _  _                  _
 | |   (_)| |__ ___ | |(_)| |_   ___  ___  __| |
 | |__ | || / // -_)| || || ' \ / _ \/ _ \/ _` |
 |____||_||_\_\\___||_||_||_||_|\___/\___/\__,_|
   ___        _            _        _    _
  / __| __ _ | | __  _  _ | | __ _ | |_ (_) ___  _ _   ___
 | (__ / _` || |/ _|| || || |/ _` ||  _|| |/ _ \| ' \ (_-<
  \___|\__,_||_|\__| \_,_||_|\__,_| \__||_|\___/|_||_|/__/

\n\nThis script takes you through the calculations for combining a substitution process
and a composition vector in order to obtain a P matrix that can be used for estimating
the likelihood of evolutionary change from one DNA sequence to another
I provide a relative rate matrix and a composition vector and all calculations are derived from
this starting point. In practice, the \"maximum\" likelihood will be calculated through optimisation of
these matrices - the composition and the process matrices
''')
# set up the substitution process
# this is a relative rate matrix - there are no units
# The standard composition matrix for this exercise: 
R = np.array([0.0, 0.3, 0.4, 0.3, 0.3, 0.0, 0.3, 0.4, 0.4, 0.3, 0.0, 0.3, 0.3, 0.4, 0.3, 0.0]).reshape(4,4)

# A Jukes-Cantor style rate matrix - all substutitions have equal rates.
#R = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]).reshape(4,4)

#R = np.array([0.0, 1.0, 1.333, 1.0, 1.0, 0.0, 1.0, 1.333, 1.333, 1.0, 0.0, 1.0, 1.0, 1.333, 1.0, 0.0]).reshape(4,4)

print('''\nThis is R, the relative rate matrix.
The order of the columns is A, C, G, T
and the order of the rows is A, C, G, T:

R =''')
print(R)

# set up a base composition vector
# in practice, this vector would be calculated from the data, or estimated
Pi = np.array([0.1, 0.4, 0.2, 0.3])
#Pi = np.array([0.25, 0.25, 0.25, 0.25])
print('''\nThis is Pi, a composition vector and the order of the bases is A, C, G, T:
''')
print("Pi = ", Pi)


# get the first of the Q matrices,
# by multiplying the relative rate matrix
# by the composition vector

unscaledQ = R * Pi

# have a look at the first Q matrix

print("\n\nThis is unscaledQ (R multiplied by Pi)")
print(unscaledQ)
print ('''
While this matrix gives some idea of the relative evolvability
of each kind of substitution, it needs some scaling:
''')

# we will normalise the matrix in a few steps
# first we will make the rows sum to Zero

rowSums = unscaledQ.sum(axis=1)

print("These are the row sums of unscaledQ:")
print(rowSums)

# now get the negative of the rowSums in order to put on the diagonals
for i in range(4):
    unscaledQ[i][i] = -rowSums[i]

print("\nThe diagonal of the unscaled Q matrix now has negative of row sums (new unscaledQ):")
print(unscaledQ)
print("\nAs you can see, every row now sums to Zero\n")

# construct a diagonal matrix of the composition
print("This is a diagonal matrix of the composition vector (diagPi):")
diagPi = np.diag(Pi)
print(diagPi)

# Sum the off-diagonals
def sumOfOffDiags(a):
    diag_sum = sum(np.diagonal(a))
    off_diag_sum = np.sum(a) - diag_sum
    return off_diag_sum

# carry out matrix multiplication
b = np.dot(diagPi,unscaledQ)

print("\nThis is b (b = unscaledQ * diagPi):")
print(b)
print("\nWe just constructed this matrix temporarily in order to get a scaling factor")

thisSumOfOffDiags = sumOfOffDiags(b)

print('''
This is the sum of the off diagonals of b,
and this sum can be used as a scaling factor
''')
print("\t", f'{thisSumOfOffDiags:.3f}')

scaledQ = unscaledQ / thisSumOfOffDiags

print('''\nThe appropriately scaled Q matrix
is derived by dividing the unscaledQ matrix
by the sum of the off diagonals of the b matrix
i.e. scaledQ = unscaledQ / thisSumOfOffDiags\n
''')
print(scaledQ)

print('''
We use this scaledQ matrix in the equation: P(v) = e^Qv

The code is:
    P(BranchLengthInSubsPerSite) = linalg.expm(scaledQ * BranchLengthInSubsPerSite)''')
P = linalg.expm(scaledQ * 0.02)
print("\n")

print("This is Q multiplied by v (Qv), when v=0.02)")
QbyV = scaledQ * 0.02
print(QbyV)
print('''
e, also known as Euler's number, is a mathematical constant approximately equal to 2.71828

Using the formula:
    P(ν)=e^Qν (i.e. we raise e to the power of Qv)")

This is P(v), for a branch length (v) of 0.02 substitutions per site.
''')
print(P, "\n")

P001 = linalg.expm(scaledQ * 0.1)
print("This is P(v), for a branch length (v) of 0.1 substitutions per site.")
print(P001)

print("\n")
Ppoint2 = linalg.expm(scaledQ * 0.2)
print("This is P(v), for a branch length (v) of 0.2 substitutions per site.")
print(Ppoint2)

print("\n")
P20 = linalg.expm(scaledQ * 20)
print("Finally, this is P(v), for a branch length (v) of 20 substitutions per site.")
print(P20)
print("\n")
# for brln in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0]:
#    print ("branch length", brln)
#    print (linalg.expm(brln * bigQ))
#    print

# for i in range(4):
#    for j in range(4):
#        print(P[i][j])
print('''
  _      _ _        _ _ _                     _
 | |    (_) |      | (_) |                   | |
 | |     _| | _____| |_| |__   ___   ___   __| |
 | |    | | |/ / _ \ | | '_ \ / _ \ / _ \ / _` |
 | |____| |   <  __/ | | | | | (_) | (_) | (_| |
 |______|_|_|\_\___|_|_|_| |_|\___/ \___/ \__,_|
         __
        / _|
   ___ | |_    __ _ _ __
  / _ \|  _|  / _` | '_ \\
 | (_) | |   | (_| | | | |
  \___/|_| _ _\__,_|_| |_|                      _
     /\   | (_)                                | |
    /  \  | |_  __ _ _ __  _ __ ___   ___ _ __ | |_
   / /\ \ | | |/ _` | '_ \| '_ ` _ \ / _ \ '_ \| __|
  / ____ \| | | (_| | | | | | | | | |  __/ | | | |_
 /_/    \_\_|_|\__, |_| |_|_| |_| |_|\___|_| |_|\__|
                __/ |
               |___/
''')
print('''
Consider this two sequence alignment.
We will calculate the likelikood of evolving from Gene1 to Gene2,
using the P matrix above, with different branch lengths.

  Gene1 C C A T
        | | + |
  Gene2 C C G T
  ''')

print('''
Starting with Gene1, we will calculate the likelihood of evolving to Gene2
on a 'tree' with a branch length of 0.02.
To remind ourselves what the P matrix for this short branch length is
here it is:
''')
print(P)
print('''
The likelihood of evolving from Gene1 to Gene2 on tree with a
branch length of 0.02, with our specified base composition vector is:
''')
print("Likelihood (Data | Model) = ProbA * Pa->a * ProbC * Pc->c * ProbG * Pg->c * ProbT * Pt->t")

print("i.e. \n\nL(D|M) = ", Pi[1], "*", f'{P[1][1]:.3f}',"*", Pi[1], "*", f'{P[1][1]:.3f}', "*", Pi[0], "*", f'{P[0][2]:.3f}', "*", Pi[3], "*", f'{P[3][3]:.3f}')


print("\nThat is to say, the likelihood is the product of the probabilities of")
print("observing a particular nucleotide multiplied by the probability of change")
print("In this case, the likelihood of the data, given the model is:\n")
like = Pi[1]*P[1][1] * Pi[1] * P[1][1] * Pi[0] * P[0][2] * Pi[3] * P[3][3]
print("L(D) = P(D|M) =", f'{like:.7f}')
print('''model:
    Alignment:
      Gene1 A C G T
            | | + |
      Gene2 A C C T
    Branch length:
      0.02 substitutions per site")
    Base composition:''')
print("      ", Pi)

print("But what about a longer branch length, say a branch length of 0.1")

print('''The P matrix for a branch length of 0.1 is:''')
print(P001)
print('''The likelihood of evolving from Gene1 to Gene2 on tree with a
branch length of 0.1 substitutions per site, with our specified substitution process is:
Likelihood (Data | Model) = ProbA * Pa->a(0.1) * ProbC * Pc->c(0.1) * ProbG * Pg->c(0.1) * ProbT * Pt->t(0.1)''')
print("\ni.e.:", Pi[1], "*", f'{P001[1][1]:.3f}',"*", Pi[1], "*", f'{P001[1][1]:.3f}', "*", Pi[0], "*", f'{P001[0][2]:.3f}', "*", Pi[3], "*", f'{P001[3][3]:.3f}')

print('''Which works out at:''')
like001 = Pi[1]*P001[1][1] * Pi[1] * P001[1][1] * Pi[0] * P001[0][2] * Pi[3] * P001[3][3]
print("L(D) = P(D|M) =", f'{like001:.7f}')
print("Now we can examine the  change in likelihood for different branch lengths:")

for pwr in np.arange(0.3, 0.375, 0.001):
    PMat = linalg.expm(scaledQ * pwr)
    likeTest = Pi[1]*PMat[1][1] * Pi[1] * PMat[1][1] * Pi[0] * PMat[0][2] * Pi[3] * PMat[3][3]
    print("branch length =", f'{pwr:,.5f}', "subs/site; Likelihood =", f'{likeTest:.7f}')
