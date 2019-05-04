import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

time_gpu = {}
time_cpu = {}

generate = './generate_dataset.py'
par = './../cpp/kmean-par'
seq = './../cpp/kmean-seq'
file_input = 'points_'

for k in range(100, 510, 100):
    time_cpu[k] = {}
    time_gpu[k] = {}
    for n in range(100000, 1000001, 200000):
        filename = file_input
        subprocess.call(['python3', 'generate_dataset.py', '-n', str(n), '-k', str(k), '-o', filename, '-r', '10', '-v', '20'])
        output = subprocess.check_output([seq, '-o', 'res', '-i', filename, '-k', str(k)]).decode('utf8').split('\n')
        time_c = int(output[-2])
        output = subprocess.check_output([par, '-o', 'res', '-i', filename, '-k', str(k)]).decode('utf8').split('\n')
        time_p = int(output[-2])
        time_cpu[k][n] = time_c
        time_gpu[k][n] = time_p
        print(time_c)
        print(time_p)

# Scale n
x1, y1 = zip(*sorted(time_gpu[500].items()))
x2, y2 = zip(*sorted(time_cpu[500].items()))
fig, ax = plt.subplots()

print(len(time_gpu))
print(len(time_gpu[500]))

ax.plot(x1, y1)
ax.plot(x2, y2)
ax.set(xlabel='Number of data points', ylabel='Execution time')
plt.show()

# Scale k
fig, ax = plt.subplots()
new = [(k, v[500000]) for k, v in time_gpu.items()]
x, y = zip(*sorted(new))
ax.set(xlabel='Number of clusters', ylabel='Execution time')
ax.plot(x, y)
plt.show()

# heat map
m = np.zeros((6, 6))
for k, v in sorted(time_gpu.items()):
	for k2, v2 in sorted(v.items()):
		m[k / 100][k2 / 100000] = (1.0 * time_cpu[k][k2]) / v2
		#m.append(((k, k2), v2 / (1.0 * time_cpu[k][k2])))
print(m[1:,1:])
plt.imshow(m[1:,1:], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(0, 5), range(100, 510, 100))
plt.yticks(range(0, 5), range(100000, 1000001, 200000))
plt.show()

# K discovery
subprocess.call(['python3', 'generate_dataset.py', '-n', '500000', '-k', '50', '-o', file_input, '-r', '10', '-v', '20'])
time_c = int(subprocess.check_output([seq, '-o', 'res', '-i', file_input, '-a', '-m', '5', '-M', '100', '-s', '2'])[0])
time_p = int(subprocess.check_output([par, '-o', 'res', '-i', file_input, '-a', '-m', '5', '-M', '100', '-s', '2'])[0])

print('speedup : ' + str(time_p / (1.0 * time_c)))