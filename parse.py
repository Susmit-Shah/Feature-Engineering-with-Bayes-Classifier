fh = open('lines.txt', 'r')
all = fh.read()

lines = all.strip('\n').split("\n")

for line in lines:
    each_line = line.split('###')
    poem_line = each_line[0]
    author = each_line[1]
    poem = poem_line.strip(' ,').split(',')
    print str(poem) + " :: " + author
