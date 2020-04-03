import pandas as pd
from collections import defaultdict
import csv

# {0 , 0~20, 20~40, 40~60, 60~80, 80~100, 100~120, 120~140}
mybin = defaultdict(int)
rows = []
filename = 'mealAmountData1.csv'

with open(filename, 'r') as file:
    csvreader = csv.reader(file,delimiter=',')

    for row in csvreader:
        s = int(','.join(row))
        rows.append(s)

# slice out first 50 rows
rows = rows[:50]
print(rows)

# categorize into mybin
for i in rows:
    if i == 0:
        mybin["0"] += 1
    elif i > 0 and i <= 20:
        mybin["0~20"] += 1
    elif i > 20 and i <= 40:
        mybin["20~40"] += 1
    elif i > 40 and i <= 60:
        mybin["40~60"] += 1
    elif i > 60 and i <= 80:
        mybin["60~80"] += 1
    elif i > 80 and i <= 100:
        mybin["80~100"] += 1
    elif i > 100 and i <= 120:
        mybin["100~120"] += 1
    elif i > 120 and i <= 140:
        mybin["120~140"] += 1
print(mybin)