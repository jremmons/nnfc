#!/usr/bin/env python3

import random
import math

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

DISTRIBUTION = [random.randint(1, 1000) for _ in range(len(ALPHABET))]
SEQUENCE_LENGTH = 2**20

assert len(ALPHABET) == len(DISTRIBUTION)
pdf = list(map(lambda x: x/sum(DISTRIBUTION), DISTRIBUTION))

# compute entropy 
entropy = 0
for p in pdf:
    entropy += -(p * math.log2(p))

print('entropy (bits / symbol)', entropy)
print('compressed size lower bound (bytes):', int(SEQUENCE_LENGTH * entropy/8) + 1)
        
cdf = []
cumsum = 0
for p in pdf:
    cumsum += p
    cdf.append(cumsum)

assert len(cdf) == len(ALPHABET)    
def get_symbol():
    sample = random.random()

    for i in range(len(cdf)):
        if sample < cdf[i]:
            return ALPHABET[i]

            
sequence = ''
for _ in range(SEQUENCE_LENGTH):
    sequence += get_symbol()

open('text.txt', 'w').write(sequence)
