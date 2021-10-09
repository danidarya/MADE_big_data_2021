#!/usr/bin/env python3
"""reducer.py"""

import sys

price_mean = 0
price_var = 0
count = 0

for line in sys.stdin:
    line = line.strip()
    cnt, price_m, price_v = line.split(" ")
    
    try:
        price_m, price_v =  float(price_m), float(price_v)
        cnt = int(cnt)
    except ValueError:
        continue

    price_var = (price_v * cnt + price_var * count) / (cnt + count) + \
                cnt * count * ((price_m - price_mean) / (cnt + count))**2
    price_mean = (price_m * cnt + price_mean * count) / (cnt + count)
    
    count += cnt

print('mapreduce mean price is ', price_mean)
print('numpy mean price is 152.7206871868289')
print('------------------------------------')
print('mapreduce price variance is ', price_var)
print('numpy price variance is 57672.84569843359')
