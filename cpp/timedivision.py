import os
import matplotlib.pyplot as plt
import numpy as np


def calc():
	test_k = [10, 25, 50, 100, 200, 500, 1000]
	for _ in range(5):
		for k in test_k:
			print(k)
			create = '../code/generate_dataset.py -r 2 -n 1000000 -k ' + str(int(k * 1.1))
			os.system(create)
			run = './kmean-par -k ' + str(k)
			os.system(run)


def reduce():
	with open("timedivision.csv") as file:
		lines = file.readlines()
		header = lines[0]
		count = {}
		attr = {}
    	for line in lines[1:]:
    		line = "".join(line.split()).split(',')
    		k = int(line[0])
    		if k in count:
    			count[k] += 1
    			a = attr[k]
    			new_a = [x + int(y) for (x, y) in zip(a, line[1:])]
    			attr[k] = new_a
    		else:
    			count[k] = 1
    			attr[k] = [int(x) for x in line[1:]]

    	with open("reduced.csv", 'w') as out:
    		out.write(header)
    		for k in count:
    			a = attr[k]
    			c = count[k]
    			a = [str(x / c) for x in a]
    			out.write(str(k) + ", " + ",".join(a) + '\n')

def get_col(data, i):
    return np.array([int(x[i]) for x in data])

def plot():
    with open("reduced.csv") as file:
        lines = file.readlines()
        header = lines[0]
        data = ["".join(line.split()).split(',') for line in lines[1:]]
        k = get_col(data, 0)
        x = range(len(k))
        init = get_col(data, 1)
        acc = get_col(data, 2)
        pos = get_col(data, 3)
        clust = get_col(data, 4)

        plt.bar(x, init, color='r', align='center')
        plt.bar(x, acc, color='b', bottom=init, align='center')
        plt.bar(x, pos, color='y', bottom=init + acc, align='center')
        plt.bar(x, clust, color='g', bottom=init + acc + pos, align='center')

        plt.xticks(x, k)
        plt.xlabel("number of cluster", fontsize=16)
        plt.ylabel("execution time (ms)", fontsize=16)
        plt.legend(['Initialization', 'Add the points in each cluster', 'Divide by the number of point', 'Cluster assignment'], loc=2)

        plt.show()

plot();
