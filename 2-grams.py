from collections import defaultdict
import string

s = "This; is a different,   thing"
l = s.split(" ")
print l
l = [x for x in l if x]
print l
d = defaultdict(int)

for i in range(0, len(l) - 1):
    d[l[i]+ " " +l[i+1]] += 1
print d