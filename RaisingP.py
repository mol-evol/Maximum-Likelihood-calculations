import numpy as np
from scipy import linalg
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})





print("A quick note about P matrices - raise them to a large power and the composition pops out:")
for pwr in [2, 10, 100, 800]:
    print ("power", pwr)
    print(np.linalg.matrix_power(bigP, pwr))  # only for integral powers
    print

bigPpoint02 = linalg.expm(bigQ * 0.02)
bigPpoint2 = linalg.expm(bigQ * 0.2)
bigP1 = linalg.expm(bigQ * 1)
bigP2 = linalg.expm(bigQ * 2)
bigP20 = linalg.expm(bigQ * 20)
print("\nThis is bigP, for a branch length of 0.2 substitutions per site.")
print(bigPpoint2)
print("\nThis is bigP, for a branch length of 2 substitutions per site.")
print(bigP2)
print("\nThis is bigP, for a branch length of 20 substitutions per site.")
print(bigP20)

import matplotlib.pyplot as plt
w = 4
h = 3
d = 70
plt.figure(figsize=(w, h), dpi=d)
color_map = plt.imshow(bigP20)
color_map.set_cmap("Blues_r")
plt.colorbar()
plt.savefig("out.png")
