#!/usr/bin/env python3
"""mapper.py"""

import sys

count = 0
sum_val = 0
sum_squares = 0

for line in sys.stdin:
    line = line.strip('\t')
    line = line.split(',')
    try:
    	value = int(line[1])
    except (ValueError, IndexError) as error:
        continue

    count += 1
    sum_val += value
    sum_squares += value ** 2

price_m = sum_val / count 
price_v = sum_squares / count - price_m ** 2
    
print(count, price_m, price_v)
