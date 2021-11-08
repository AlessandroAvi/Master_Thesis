
import numpy as np


with open('file.csv') as csvDataFile:

    # read file as csv file 
    csvReader = csv.reader(csvDataFile)

    # for every row, print the row
    for row in csvReader:
        print(row)


print(len(lines))

print(len(lines[0]))

OL_weights = np.zeros((len(lines[0]), len(lines)))


print(lines[0][4])

"""
for i in range(0, len(lines)):
    for j in range(0, len(lines[0])):

        OL_weights[] = 
"""