import numpy as np
import re

def read_file(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        rows, columns = [int(s) for s in line.split(' ') if s.isdigit()]

        matrix = np.empty([rows + 1, columns])
        columnIndex = 0

        while columnIndex < columns:
            line = f.readline()
            for number in map(int, re.findall('\d+', line)):
                matrix[0][columnIndex] = number
                columnIndex += 1

        rowIndex = 1
        for line in f:
            setSize = int(line)
            counter = 0
            while counter < setSize:
                line = f.readline()
                for number in map(int, re.findall('\d+', line)):
                    matrix[rowIndex][number - 1] = 1
                    counter += 1

            rowIndex += 1

        return matrix
