import csv
import os
import random
from fingerSnapUtil import *

f = open('./data.csv', 'r', encoding='utf8')
rdr = csv.reader(f)

for line in rdr:
    if line[1] == '9':
        if random.randrange(1, 11) < 8:
            continue
    line = [strip(line[0]), line[1]]
    t = open('./target.csv', 'a', newline='', encoding='utf8')
    wr = csv.writer(t)
    wr.writerow(line)
    t.close()

f.close()
