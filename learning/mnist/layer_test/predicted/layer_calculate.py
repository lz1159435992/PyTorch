import numpy as np
labels = np.loadtxt('labels.txt')
all1_predicted_10 = np.loadtxt('all1/predicted_10.txt')
fc1_predicted_10 = np.loadtxt('fc1/predicted_10.txt')
fc2_predicted_10 = np.loadtxt('fc2/predicted_10.txt')
fc3_predicted_10 = np.loadtxt('fc3/predicted_10.txt')
output_predicted_10 = np.loadtxt('output/predicted_10.txt')
def fc1_layer(labels, all1_predicted_10, fc1_predicted_10):
    x = 0
    t = 0
    f = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            a = a + all1_predicted_10[x][j]
            b = b + fc1_predicted_10[x][j]
        a = all1_predicted_10[x][int(i)]/a
        b = fc1_predicted_10[x][int(i)]/b
        if a - b >= 0:
            t = t + 1
        else :
            f = f + 1
        x = x + 1
    print(t,f)
def fc2_layer(labels, fc1_predicted_10, fc2_predicted_10):
    x = 0
    t = 0
    f = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            a = a + fc1_predicted_10[x][j]
            b = b + fc2_predicted_10[x][j]
        a = fc1_predicted_10[x][int(i)]/a
        b = fc2_predicted_10[x][int(i)]/b
        if a - b >= 0:
            t = t + 1
        else :
            f = f + 1
        x = x + 1
    print(t,f)
def fc3_layer(labels, fc2_predicted_10, fc3_predicted_10):
    x = 0
    t = 0
    f = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            a = a + fc2_predicted_10[x][j]
            b = b + fc3_predicted_10[x][j]
        a = fc2_predicted_10[x][int(i)]/a
        b = fc3_predicted_10[x][int(i)]/b
        if a - b >= 0:
            t = t + 1
        else :
            f = f + 1
        x = x + 1
    print(t,f)
#fc1_layer(labels,all1_predicted_10,fc1_predicted_10)
#fc2_layer(labels,fc1_predicted_10,fc2_predicted_10)
fc3_layer(labels,fc2_predicted_10,fc3_predicted_10)
