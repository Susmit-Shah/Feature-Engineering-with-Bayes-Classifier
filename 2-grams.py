from collections import defaultdict
import string, re

st = "This; is a different,   thing"
s = re.sub('[^\w ]', "", st)
print s
l = s.split(" ")
print l
l = [x for x in l if x]
print l
d = defaultdict(int)

for i in range(0, len(l) - 3):
    d[l[i]+ " " +l[i+1] + " " + l[i+2] + " " + l[i+3]] += 1
print d