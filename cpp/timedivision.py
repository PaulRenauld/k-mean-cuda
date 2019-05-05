import os
import matplotlib.pyplot as plt

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

def plot():
	with open("reduced.csv") as file:

reduce();
