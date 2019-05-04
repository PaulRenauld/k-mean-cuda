

import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: plot_csv.py <csv-file-to-plot> <column-as-y>")

    y_col = int(sys.argv[2])
    x_buf = []
    y_buf = []
    with open(sys.argv[1]) as file:
        lines = file.readlines()
        for line in lines[1:]:
            line = "".join(line.split()).split(',')
            if line[0].replace('.', '').isnumeric():
                x_buf.append(float(line[0]))
                y_buf.append(float(line[y_col]))
        plt.plot(x_buf, y_buf, 'bo-')

        header = "".join(lines[0].split()).split(',')
        plt.xlabel(header[0])
        plt.ylabel(header[y_col])
        plt.show()
