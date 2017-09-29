from collections import Counter
from itertools import chain
from string import punctuation
import glob
import os
import string
import re

def countInFile(filename):
    with open(filename, "r", encoding = 'utf8') as f:
        for line in f:
            nobr = re.sub(r'<br>', ' ', line)
            no_punct = ''.join(c for c in nobr if c not in string.punctuation)
            linewords = no_punct.lower().split()
        return Counter(linewords)

dir = os.path.dirname(__file__)
file_list = glob.glob(os.path.join(dir, 'reviews/pos/*'))
file_list.extend(glob.glob(os.path.join(dir, 'reviews/neg/*')))

count = Counter()
for f in file_list:
    count += countInFile(f)

with open("50_common.txt", "w") as f:
    mostcom = count.most_common(50)
    for el in mostcom:
        f.write("'")
        f.write(el[0])
        f.write("', ")
