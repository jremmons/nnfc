import mpmath

from mpmath import mpf
from decimal import Decimal 

mpmath.mp.dps = 100000
digits = mpf
#digits = Decimal
print('binary digits of precision', mpmath.mp.prec)

import sys

model = {
    'A' : (digits(0), digits(3)/digits(8)),
    'B' : (digits(3)/digits(8), digits(7)/digits(8)),
    '$' : (digits(7)/digits(8), digits(1)),
}

with open('text.txt', 'r') as f:
    data = f.read().strip()
    print(data)

    data += '$'
    
    high = digits(1)
    low = digits(0)

    for char in data:
        m = model[char]

        r = high - low
        high = low + r * m[1]
        low = low + r * m[0]

    print(high)
    num = 0
    for i, digit in enumerate(sys.argv[1]):
        
        if digit == '1':
            num += digits(1) / digits(2**(i+1))

    print(num)          
    print(low)   
    print('high > low', high > low)
    print('high > num', high > num)
    print('low <= num', low <= num)

    sequence = ''
    high = digits(1)
    low = digits(0)
    while True:
        r = high - low
        key = None
        for k in model.keys():

            sym_low = low + r * model[k][0]
            sym_high = low + r * model[k][1]
            
            if num >= sym_low and num < sym_high:
                high = sym_high
                low = sym_low
                key = k
                break

        sequence += key
        if key == '$':
            print()
            print('detected $. done!')
            break
            
        sys.stdout.write(key)

    print('input == output', data == sequence)
