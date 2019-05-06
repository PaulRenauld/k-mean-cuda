import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    f1 = "../results/n100000k50.csv"
    f2 = "../results/n100000k50a.csv"
    y_col = 1
    x1_buf = []
    x2_buf = []
    y1_buf = []
    y2_buf = []
    with open(f1) as file1:
        with open(f2) as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()
            for line in lines1[1:]:
                line = "".join(line.split()).split(',')
                if line[0].replace('.', '').isnumeric():
                    x1_buf.append(float(line[0]))
                    y1_buf.append(float(line[y_col]))
            for line in lines2[1:]:
                line = "".join(line.split()).split(',')
                if line[0].replace('.', '').isnumeric():
                    x2_buf.append(float(line[0]))
                    y2_buf.append(float(line[y_col]))

            header = "".join(lines1[0].split()).split(',')
            plt.xlabel(header[0])
            plt.ylabel(header[y_col])
            plt.show()

            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('k parameter')
            ax1.set_ylabel('original silhouette', color=color)
            ax1.plot(x1_buf, y1_buf, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('approximation', color=color)  # we already handled the x-label with ax1
            ax2.plot(x2_buf, y2_buf, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
