#!/usr/bin/env python3

import random

ALPHABET = 'AB'
DISTRIBUTION = [11000, 999]
SEQUENCE_LENGTH = 2**20

assert len(ALPHABET) == len(DISTRIBUTION)
pdf = list(map(lambda x: x/sum(DISTRIBUTION), DISTRIBUTION))

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
