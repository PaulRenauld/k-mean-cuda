#!/usr/bin/env python

import argparse
import random
from math import sqrt, isclose


def main():
  parser = argparse.ArgumentParser("Partition the data set into k clusters", conflict_handler="resolve")
  parser.add_argument("-o", "--output-file", type=str,
                      help="Name of the file to store the generated result", default="results")
  parser.add_argument("-i", "--input-file", type=str,
                      help="Name of the file to read the graph", default="random_points")
  parser.add_argument("-k", "--cluster-count", type=int, default=5,
                      help="Specify the number of cluster")
  parser.add_argument("-w", "--write-interval", type=int, default=100,
                      help="Number of iteration between displaying the information and writing the file")

  args = parser.parse_args()

  dataset, dataset_argument = read_input(args)
  clusters = select_clusters_positions(dataset, args.cluster_count)
  cluster_index, clusters = main_loop(clusters, dataset)

  write_file_output(args.output_file, dataset, clusters, cluster_index, dataset_argument)
  print('done')


def main_loop(clusters, dataset):
  changed = True
  cluster_index = []
  it = 0
  while changed:
    clusters, cluster_index, changed = update_clusters(clusters, dataset)
    it += 1

    print('iteration ' + str(it) + ' done')
  return cluster_index, clusters


def select_clusters_positions(dataset, cluster_count):
  positions = set()
  while len(positions) < cluster_count:
    positions.add(random.randrange(len(dataset)))
  return [dataset[i] for i in positions]


def update_clusters(clusters, dataset):
  changed = False
  k = len(clusters)
  new_cluster_position = [point(0, 0) for _ in range(k)]
  new_cluster_counts = [0] * k
  new_cluster_index = []

  for p in dataset:
    distances = [p.distance_to(c) for c in clusters]
    closest = distances.index(min(distances))
    new_cluster_position[closest] += p
    new_cluster_counts[closest] += 1
    new_cluster_index.append(closest)

  for i in range(k):
    new_cluster_position[i].div(new_cluster_counts[i])

  for (n, o) in zip(new_cluster_position, clusters):
    if n != o:
      changed = True

  return new_cluster_position, new_cluster_index, changed


# Read and Write in the files
def read_input(args):
  dataset_argument = []
  dataset = []
  with open(args.input_file) as file:
    for line in file.readlines():
      line = "".join(line.split()).split(',')
      if line[0].replace('.', '').isnumeric():
        dataset.append(point(float(line[0]), float(line[1])))
      elif 'C' not in line[0]:
        dataset_argument.append(line)
  return dataset, dataset_argument


def write_file_output(file_name, dataset, clusters, cluster_index, dataset_argument):
  with open(file_name, 'w') as file:
    for arg in dataset_argument:
      file.write(','.join(arg) + '\n')

    points = list(zip(dataset, cluster_index))
    points.sort(key=lambda tup: tup[1])

    index = 0
    for i, clust in enumerate(clusters):
      file.write('C,%f,%f\n' % (clust.x, clust.y))
      while index < len(points) and points[index][1] == i:
        file.write('%f,%f\n' % (points[index][0].x, points[index][0].y))
        index += 1


class point:
  def __init__(self, x, y):
    self.x = float(x)
    self.y = float(y)

  def __add__(self, other):
    return point(self.x + other.x, self.y + other.y)

  def div(self, count):
    self.x /= count
    self.y /= count

  def distance_to(self, other):
    return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

  def __eq__(self, other):
    return isclose(self.x, other.x) and isclose(self.y, other.y)


if __name__ == '__main__':
  main()
