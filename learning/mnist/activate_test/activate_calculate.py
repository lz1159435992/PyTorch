import numpy as np
import math

labels = np.loadtxt('predicted/labels.txt')
zero_predicted_10 = np.loadtxt('predicted/zero_predicted_10.txt')
first_predicted_10 = np.loadtxt('predicted/first_predicted_10.txt')
second_predicted_10 = np.loadtxt('predicted/second_predicted_10.txt')
predicted_10 = np.loadtxt('predicted/predicted_10.txt')
def first_function(labels, zero_predicted_10, first_predicted_10):
    x = 0
    t = 0
    f = 0
    m = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            try:
                c = math.exp(zero_predicted_10[x][j])
            except OverflowError:
                c = float('inf')
            try:
                d = math.exp(first_predicted_10[x][j])
            except OverflowError:
                d = float('inf')
            # a = a + math.exp(all1_predicted_10[x][j])
            a = a + c
            # b = b + math.exp(fc1_predicted_10[x][j])
            b = b + d
        try:
            c = math.exp(zero_predicted_10[x][int(i)])
        except OverflowError:
            c = float('inf')
        try:
            d = math.exp(first_predicted_10[x][int(i)])
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
    print('第一个激活函数')
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)
def second_function(labels, zero_predicted_10, second_predicted_10):
    x = 0
    t = 0
    f = 0
    m = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            try:
                c = math.exp(zero_predicted_10[x][j])
            except OverflowError:
                c = float('inf')
            try:
                d = math.exp(second_predicted_10[x][j])
            except OverflowError:
                d = float('inf')
            # a = a + math.exp(all1_predicted_10[x][j])
            a = a + c
            # b = b + math.exp(fc1_predicted_10[x][j])
            b = b + d
        try:
            c = math.exp(zero_predicted_10[x][int(i)])
        except OverflowError:
            c = float('inf')
        try:
            d = math.exp(second_predicted_10[x][int(i)])
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
    print('第二个激活函数')
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)
first_function(labels,predicted_10,first_predicted_10)
second_function(labels,predicted_10,second_predicted_10)