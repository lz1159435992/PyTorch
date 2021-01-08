import numpy as np
import math

labels = np.loadtxt('labels.txt')
all1_predicted_10 = np.loadtxt('all1/predicted_10.txt')
fc1_predicted_10 = np.loadtxt('fc1/predicted_10.txt')
fc2_predicted_10 = np.loadtxt('fc2/predicted_10.txt')
fc3_predicted_10 = np.loadtxt('fc3/predicted_10.txt')
output_predicted_10 = np.loadtxt('output/predicted_10.txt')
new_layer_predicted_10 = np.loadtxt('new_layer/predicted_10.txt')


def fc1_layer(labels, all1_predicted_10, fc1_predicted_10):
    x = 0
    t = 0
    f = 0
    m = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            try:
                c = math.exp(all1_predicted_10[x][j])
            except OverflowError:
                c = float('inf')
            try:
                d = math.exp(fc1_predicted_10[x][j])
            except OverflowError:
                d = float('inf')
            # a = a + math.exp(all1_predicted_10[x][j])
            a = a + c
            # b = b + math.exp(fc1_predicted_10[x][j])
            b = b + d
        try:
            c = math.exp(all1_predicted_10[x][int(i)])
        except OverflowError:
            c = float('inf')
        try:
            d = math.exp(fc1_predicted_10[x][int(i)])
        except OverflowError:
            d = float('inf')
        a = c / a
        b = d / b
        if b - a > 0:
            t = t + 1
        elif b - a == 0:
            m = m + 1
        else:
            f = f + 1
        x = x + 1
    print('fc1')
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)


def fc2_layer(labels, fc1_predicted_10, fc2_predicted_10):
    x = 0
    t = 0
    f = 0
    m = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            try:
                c = math.exp(fc1_predicted_10[x][j])
            except OverflowError:
                c = float('inf')
            try:
                d = math.exp(fc2_predicted_10[x][j])
            except OverflowError:
                d = float('inf')
            # a = a + math.exp(fc1_predicted_10[x][j])
            # b = b + math.exp(fc2_predicted_10[x][j])
            a = a + c
            b = b + d
        try:
            c = math.exp(fc1_predicted_10[x][int(i)])
        except OverflowError:
            c = float('inf')
        try:
            d = math.exp(fc2_predicted_10[x][int(i)])
        except OverflowError:
            d = float('inf')
        # a = math.exp(fc1_predicted_10[x][int(i)]) / a
        # b = math.exp(fc2_predicted_10[x][int(i)]) / b
        a = c / a
        b = d / b
        if b - a > 0:
            t = t + 1
        elif b - a == 0:
            m = m + 1
        else:
            f = f + 1
        x = x + 1
    print('fc2')
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)


def fc3_layer(labels, fc2_predicted_10, fc3_predicted_10):
    x = 0
    t = 0
    f = 0
    m = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            a = a + math.exp(fc2_predicted_10[x][j])
            b = b + math.exp(fc3_predicted_10[x][j])
        a = math.exp(fc2_predicted_10[x][int(i)]) / a
        b = math.exp(fc3_predicted_10[x][int(i)]) / b
        if b - a > 0:
            t = t + 1
        elif b - a == 0:
            m = m + 1
        else:
            f = f + 1
        x = x + 1
    print('fc3')
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)


def output_layer(labels, fc3_predicted_10, output_predicted_10):
    x = 0
    t = 0
    f = 0
    for i in labels:
        a = 0
        b = 0
        m = 0
        for j in range(10):
            a = a + math.exp(fc3_predicted_10[x][j])
            b = b + math.exp(output_predicted_10[x][j])
        # 计算概率的变化
        a = math.exp(fc3_predicted_10[x][int(i)]) / a
        b = math.exp(output_predicted_10[x][int(i)]) / b
        if b - a > 0:
            t = t + 1
        elif b - a == 0:
            m = m + 1
        else:
            f = f + 1
        x = x + 1
    print('output')
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)


def new_layer(labels, before, after):
    x = 0
    t = 0
    f = 0
    for i in labels:
        a = 0
        b = 0
        m = 0
        for j in range(10):
            try:
                c = math.exp(before[x][j])
            except OverflowError:
                c = float('inf')
            # a = a + math.exp(before[x][j])
            a = a + c
            b = b + math.exp(after[x][j])
        # 计算概率的变化
        try:
            c = math.exp(before[x][int(i)])
        except OverflowError:
            c = float('inf')
        a = c / a
        b = math.exp(after[x][int(i)]) / b
        if b - a > 0:
            t = t + 1
        elif b - a == 0:
            m = m + 1
        else:
            f = f + 1
        x = x + 1
    print('新增节点')
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)


fc1_layer(labels, all1_predicted_10, fc1_predicted_10)
new_layer(labels, fc2_predicted_10, new_layer_predicted_10)
fc2_layer(labels, fc1_predicted_10, fc2_predicted_10)
fc3_layer(labels, new_layer_predicted_10, fc3_predicted_10)
output_layer(labels, fc3_predicted_10, output_predicted_10)
