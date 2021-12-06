import numpy as np
from scipy import linalg

# set up the substitution process
# this is a relative rate matrix - there are no units
bigR = np.array([0.0, 0.3, 0.4, 0.3, 0.3, 0.0, 0.3, 0.4, 0.4, 0.3, 0.0, 0.3, 0.3, 0.4, 0.3, 0.0]).reshape(4,4)

# set up a base composition vector
# in practice, this vector would be calculated from the data, or estimated

myPi = np.array([0.1, 0.4, 0.2, 0.3])

# check that everything is OK

print("\nThis is bigR:")
print(bigR)

# Output
# This is bigR:
# [[0.  0.3 0.4 0.3]
#  [0.3 0.  0.3 0.4]
#  [0.4 0.3 0.  0.3]
#  [0.3 0.4 0.3 0. ]]


print("\nThis is myPi:")
print(myPi)

# Output
# This is myPi:
# [0.1 0.4 0.2 0.3]


# get the first of the Q matrices,
# by multiplying the relative rate matrix
# by the composition vector

unscaledBigQ = bigR * myPi

# have a look at the first Q matrix

print("\nThis is unscaledBigQ:")
print(unscaledBigQ)

# Output
# This is unscaledBigQ:
# [[0.00 0.12 0.08 0.09]
#  [0.03 0.00 0.06 0.12]
#  [0.04 0.12 0.00 0.09]
#  [0.03 0.16 0.06 0.00]]


# we will normalise the matrix in a few steps
# first we will make the rows sum to Zero

rowSums = unscaledBigQ.sum(axis=1)

print("\nThese are the rowSums:")
print(rowSums)

# Output:
# These are the rowSums:
# [0.29 0.21 0.25 0.25]



for i in range(4):
    unscaledBigQ[i][i] = -rowSums[i]

print("\nThis is the diagonal of the matrix fixed:")
print(unscaledBigQ)

# construct a diagonal matrix of the composition

bigPi = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0,0.0, 0.0, 0.0, 0.3 ]).reshape(4,4)

print("\nThis is the diagonal matrix of the composition:")
print(bigPi)

# make a handy function
def sumOfOffDiags(a):
    dm = a.shape[0]
    assert a.shape[1] == dm
    sm = 0.0
    for i in range(dm):
        for j in range(dm):
            if j is not i:
                sm += a[i][j]
    return sm

# carry out matrix multiplication

b = np.dot(bigPi,unscaledBigQ)

print("\nThis is b:")
print(b)

thisSumOfOffDiags = sumOfOffDiags(b)

print("\nThis is the sum of the off diagonals of b")
print(thisSumOfOffDiags)

bigQ = unscaledBigQ / thisSumOfOffDiags

print("\nThis is the appropriately scaled bigQ")
print("It has been derived by dividing the unscaledBigQ by the sum of the off diagonals")
print(bigQ)




# Now, we can get a precise P-matrix for a branch length of 0.02,
# by P(ν)=e^Qν

bigP = linalg.expm(0.02 * bigQ)

print("This is bigP, for a branch length of 0.02 substitutions per site.")
print(bigP)

for brln in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]:
    print ("branch length" brln)
    print (linalg.expm(brln * bigQ))


# for pwr in [2, 3, 10, 100, 200, 800]:
#     print ("power", pwr)
#    print(np.linalg.matrix_power(bigP, pwr))  # only for integral powers
#    print
