#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
  if len(sys.argv) != 2 or sys.argv[1][0] == '-':
    print("Usage: plot.py <file-to-plot>")

  width = 1000
  height = 1000
  current_color = 0
  color = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
  with open(sys.argv[1]) as file:
    for line in file.readlines():
      line = "".join(line.split()).split(',')
      if line[0].replace('.', '').isnumeric():
        plt.plot([float(line[0])], [float(line[1])],
                 color[current_color] + '.')
      else:
        if line[0] == 'height':
          height = int(line[1])
        elif line[0] == 'width':
          width = int(line[1])
        elif line[0] == 'C':
          current_color = (current_color + 1) % len(color)
          plt.plot([float(line[1])], [float(line[2])],
                   color[current_color] + '^')

  plt.axis([0, width, 0, height])
  plt.show()
