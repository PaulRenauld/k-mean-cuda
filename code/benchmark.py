import matplotlib
import subprocess
import matplotlib.pyplot as plt

time_gpu = []
time_cpu = []

generate = './generate_dataset.py'
par = './../cpp/kmean-par'
seq = './../cpp/kmean-seq'
file_input = 'points_'

for k in range(2, 51, 2):
    time_cpu[k] = {}
    time_gpu[k] = {}
    for n in range(100_000, 1_000_000, 100_000):
        filename = file_input
        subprocess.call(['python', 'generate_dataset.py', '-n', str(n), '-k', str(k), '-o', filename, '-r', '10', '-v', '20'])
        time_c = int(subprocess.check_output([seq, '-o', 'res', '-i', filename, '-k', str(k)]))
        time_p = int(subprocess.check_output([par, '-o', 'res', '-i', filename, '-k', str(k)]))
        time_cpu[k][n] = time_c
        time_gpu[k][n] = time_p

# Scale n
x = range(100_000, 1_000_000, 100_000)
y1 = time_gpu[50]
y2 = time_cpu[50]

fig, ax = plt.subplots()
ax.plot(x, [y1, y2])
plt.show()
