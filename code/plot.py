#!/usr/bin/env python3

import random
import sys

import matplotlib.pyplot as plt


def rand_color():
  return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

if __name__ == '__main__':
  if len(sys.argv) != 2 or sys.argv[1][0] == '-':
    print("Usage: plot.py <file-to-plot>")

  width = 1000
  height = 1000
  current_color = '#0000FF'
  x_buf = []
  y_buf = []
  with open(sys.argv[1]) as file:
    for line in file.readlines():
      line = "".join(line.split()).split(',')
      if line[0].replace('.', '').isnumeric():
        x_buf.append(float(line[0]))
        y_buf.append(float(line[1]))
      else:
        if line[0] == 'height':
          height = int(line[1])
        elif line[0] == 'width':
          width = int(line[1])
        elif line[0] == 'C':
          if x_buf:
            plt.plot(x_buf, y_buf, ',', color=current_color)
            x_buf = []
            y_buf = []
          current_color = rand_color()
          plt.plot([float(line[1])], [float(line[2])],
                '^', color=current_color)

  if x_buf:
    plt.plot(x_buf, y_buf, ',', color=current_color)
    x_buf = []
    y_buf = []

  plt.axis([0, width, 0, height])
  plt.title('Figure 2: K-mean clustering for the optimal k value')
  plt.show()
