#!/usr/bin/env python
import matplotlib.pyplot as plt

def main(output: str):
    output = output.replace("silhouette with k=", '').replace(' ', '').strip().split()
    values = [(int(line[0]), float(line[1])) for line in [l.split(':') for l in output]]
    xs = [x for (x, _) in values]
    ys = [y for (_, y) in values]
    plt.plot(xs, ys, 'bo-')
    plt.title('Figure 1: Silhouette value for a dataset of 10000 points')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Silhouette value')
    plt.show()


if __name__ == '__main__':
    output = """
    silhouette with k=2: 0.346227
silhouette with k=3: 0.470481
silhouette with k=4: 0.417261
silhouette with k=5: 0.409835
silhouette with k=6: 0.456681
silhouette with k=7: 0.448866
silhouette with k=8: 0.472404
silhouette with k=9: 0.427348
silhouette with k=10: 0.416935
silhouette with k=11: 0.475667
silhouette with k=12: 0.476308
silhouette with k=13: 0.464758
silhouette with k=14: 0.470616
silhouette with k=15: 0.497664
silhouette with k=16: 0.486602
silhouette with k=17: 0.493796
silhouette with k=18: 0.493715
silhouette with k=19: 0.491342
silhouette with k=20: 0.502917
silhouette with k=21: 0.517567
silhouette with k=22: 0.507803
silhouette with k=23: 0.496292
silhouette with k=24: 0.48864
silhouette with k=25: 0.49081
silhouette with k=26: 0.482454
silhouette with k=27: 0.476064
silhouette with k=28: 0.472313
silhouette with k=29: 0.483676
silhouette with k=30: 0.480141
silhouette with k=31: 0.477628
silhouette with k=32: 0.469208
silhouette with k=33: 0.468222
silhouette with k=34: 0.473008
silhouette with k=35: 0.473488
silhouette with k=36: 0.462661
silhouette with k=37: 0.458961
silhouette with k=38: 0.460186
silhouette with k=39: 0.460809
silhouette with k=40: 0.448713
silhouette with k=41: 0.444547
silhouette with k=42: 0.43751
silhouette with k=43: 0.432112
silhouette with k=44: 0.419957
silhouette with k=45: 0.415264
silhouette with k=46: 0.414678
silhouette with k=47: 0.415762
silhouette with k=48: 0.410349
silhouette with k=49: 0.40355
silhouette with k=50: 0.396805
silhouette with k=51: 0.397055
silhouette with k=52: 0.39367
silhouette with k=53: 0.394573
silhouette with k=54: 0.382966
silhouette with k=55: 0.375492
silhouette with k=56: 0.373873
silhouette with k=57: 0.363668
silhouette with k=58: 0.373386
silhouette with k=59: 0.370099
silhouette with k=60: 0.367258
silhouette with k=61: 0.367424
silhouette with k=62: 0.362858
silhouette with k=63: 0.366425
silhouette with k=64: 0.360038
silhouette with k=65: 0.356889
silhouette with k=66: 0.354824
silhouette with k=67: 0.356824
silhouette with k=68: 0.351318
silhouette with k=69: 0.352354
silhouette with k=70: 0.347736
silhouette with k=71: 0.348015
silhouette with k=72: 0.345109
silhouette with k=73: 0.355291
silhouette with k=74: 0.353053
silhouette with k=75: 0.353703
silhouette with k=76: 0.348955
silhouette with k=77: 0.343844
silhouette with k=78: 0.339369
silhouette with k=79: 0.338756
silhouette with k=80: 0.339398
silhouette with k=81: 0.339631
silhouette with k=82: 0.339216
silhouette with k=83: 0.340668
silhouette with k=84: 0.341916
silhouette with k=85: 0.343993
silhouette with k=86: 0.34329
silhouette with k=87: 0.344199
silhouette with k=88: 0.341271
silhouette with k=89: 0.337846
silhouette with k=90: 0.340286
silhouette with k=91: 0.340816
silhouette with k=92: 0.339894
silhouette with k=93: 0.33923
silhouette with k=94: 0.338258
silhouette with k=95: 0.337707
silhouette with k=96: 0.335981
silhouette with k=97: 0.336288
silhouette with k=98: 0.336502
silhouette with k=99: 0.335928
"""
    main(output)

