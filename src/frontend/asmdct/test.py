import numpy as np
from scipy.fftpack import dct

a = np.asarray([[1 for _ in range(8)] for _ in range(8)])

for i in range(8):
    for j in range(8):

        a[i,j] = 8
        
print(a)
print(dct(dct(a.T).T))
