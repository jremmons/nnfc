import mpmath
import sys

mpmath.mp.dps = 100000

x = mpmath.mpf(sys.argv[1])
y = mpmath.mpf(sys.argv[2])

print(int(mpmath.log(y - x) / mpmath.log(2)))
