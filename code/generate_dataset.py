#!/usr/bin/env python

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np


def generate_cluster(data_points, max_width, max_height, n: int, variance):
    center_x = random.random() * max_width
    center_y = random.random() * max_height

    xs = np.random.normal(center_x, variance, n)
    ys = np.random.normal(center_y, variance, n)

    for x, y in zip(xs, ys):
        if 0 <= x < max_width and 0 <= y < max_height:
            data_points.append((x, y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate a dataset with cluster", conflict_handler="resolve")
    parser.add_argument("-o", "--output-file", type=str,
                        help="Name of the file to store the generated graph", default="random_points")
    parser.add_argument("-w", "--width", type=float,
                        help="width of the generated schema", default=1000)
    parser.add_argument("-h", "--height", type=float,
                        help="height of the generated schema", default=1000)
    parser.add_argument("-k", "--cluster-count", type=int, default=5,
                        help="Specify the number of cluster")
    parser.add_argument("-n", "--point-count", type=int, default=1000,
                        help="specify the number of data points")
    parser.add_argument("-v", "--variance", type=int, default=10,
                        help="specify the variance around the cluster centers")
    parser.add_argument("-r", "--randomness", type=int, default=20, choices=range(0, 101), metavar="[0-100]",
                        help="specify the percentage of points that are not in a cluster")

    args = parser.parse_args()

    sample_per_cluster = int((100 - args.randomness) / 100 * args.point_count / args.cluster_count)
    data_points = []
    for _ in range(args.cluster_count):
        generate_cluster(data_points, args.width, args.height,
                         sample_per_cluster, args.variance)

    print("not random points: " + str(len(data_points)))
    while len(data_points) < args.point_count:
        x = random.random() * args.width
        y = random.random() * args.height
        data_points.append((x, y))

    with open(args.output_file, 'w') as file:
        file.write("width, %d\n" % args.width)
        file.write("height, %d\n" % args.height)
        file.write("dim, 2\n")
        file.write("point-count, %d\n" % args.point_count)
        for (x, y) in data_points:
            file.write("%f, %f\n" % (x, y))

    xs = [x for (x, _) in data_points]
    ys = [y for (_, y) in data_points]

    plt.plot(xs, ys, 'b,')
    plt.axis([0, args.width, 0, args.height])
    plt.show()

    print(args)
