import matplotlib
from matplotlib import colors
import pandas as pd
import random
import numpy as np
from matplotlib import pyplot as plt


def calculate_cost():
    sum = 0
    for j in range(1, int(np.size(data.values) / np.size(data.values[0]))):
        for i in range(1, C):
            sum += pow(u[j][i], m) * pow(distance(data.values[j], v[i]), 2)
    return sum


def colored_plot():
    color_indices = []
    for i in range(int(np.size(data.values) / np.size(data.values[0]))):
        color_indices.append(np.argmax(u[i]))
    color = ['yellow', 'green', 'orange']
    colormap = matplotlib.colors.ListedColormap(color)
    plt.scatter(data.values[:, 0], data.values[:, 1], c=color_indices, cmap=colormap)
    color = ['black']
    colormap = matplotlib.colors.ListedColormap(color)
    plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], c=[0, 0, 0], cmap=colormap)
    plt.show()


def centroids_plot():
    plt.xlabel('C')
    plt.ylabel('Cost')
    plt.plot(centroid_list, cost, color='blue')
    plt.show()


def distance(a, b):
    res = 0
    for i in range(np.size(a)):
        res += pow(a[i] - b[i], 2)
    return np.sqrt(res)


def generate_centers():
    for i in range(C):
        if data_set_to_read == 4:
            temp_x = random.random() * 10 - 5
            temp_y = random.random() * 10 - 5
            temp_z = random.random() * 10 - 5
            v.append([temp_x, temp_y, temp_z])
        elif data_set_to_read == 2:
            temp_x = random.random() * 10 - 5
            temp_y = random.random() * 10 - 5
            temp_z = random.random() * 10 - 5
            temp_k = random.random() * 10 - 5
            v.append([temp_x, temp_y, temp_z, temp_k])
        else:
            temp_x = random.random() * 10 - 5
            temp_y = random.random() * 10 - 5
            v.append([temp_x, temp_y])


def find_cluster_dependency():
    for k in range(int(np.size(data.values) / np.size(data.values[0]))):
        temp_u = []
        flag = True
        for i in range(C):
            sum = 0
            for j in range(1, C):
                if distance(data.values[k], v[j]) != 0 and distance(data.values[k], v[i]) != 0:
                    sum += pow(distance(data.values[k], v[i]) / distance(data.values[k], v[j]), 2/(m - 1))
                else:
                    temp_u = []
                    for s in range(C):
                        if s == i:
                            temp_u.append(1.0)
                        else:
                            temp_u.append(0.0)
                    u.append(temp_u)
                    flag = False
                    break
            if flag:
                sum = 1 / sum
                temp_u.append(sum)
            else:
                break
        if flag:
            u.append(temp_u)


def update_centers():
    for i in range(C):
        temp_x = 0
        temp_y = 0
        temp_z = 0
        temp_k = 0
        temp1 = 0
        for k in range(int(np.size(data.values) / np.size(data.values[0]))):
            temp_x += pow(u[k][i], m) * data.values[k][0]
            temp_y += pow(u[k][i], m) * data.values[k][1]
            if np.size(data.values[k]) > 2:
                temp_z += pow(u[k][i], m) * data.values[k][2]
                if np.size(data.values[k]) > 3:
                    temp_k += pow(u[k][i], m) * data.values[k][3]
            temp1 += pow(u[k][i], m)
        v[i][0] = temp_x / temp1
        v[i][1] = temp_y / temp1
        if np.size(data.values[i]) > 2:
            v[i][2] = temp_z / temp1
            if np.size(data.values[i]) > 3:
                v[i][3] = temp_k / temp1


m = 3
C = 3
cost = []
centroid_list = []
v = []
u = []
data_set_to_read = int(input("Enter dataset to read: "))
data = pd.read_csv("data{}.csv".format(data_set_to_read))
#for C in range(2, 4):
generate_centers()
for z in range(100):
    u = []
    find_cluster_dependency()
    update_centers()
    #cost.append(calculate_cost())
    #centroid_list.append(C)
# centroids_plot()
colored_plot()
