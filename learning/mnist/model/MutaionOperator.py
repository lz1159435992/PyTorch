# import pickle as pickle
#
# f = open('model_paras.pkl', 'rb+')
# info = pickle.load(f)
# print
# info
import os
import torch
import copy
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import math
from collections import OrderedDict
import random
import re


# class MutaionOperator(object):
#     def __init__(self, ration, model, acc_tolerant=0.90, verbose=True, test=True, test_data_laoder=None):
#         self.ration = ration
#         self.model = model
#         self.verbose = verbose
#         self.test_data_laoder = test_data_laoder
#
#     def ns(self, skip=10):
#         unique_neurons = 0
#         mutation_model = self.model
#         for idx_layer, param in mutation_model.named_parameters():
#             shape = param.size()#参数矩阵的形状
#             dim = len(shape)  # 定轴参数大小 即是几维的矩阵
#             unique_neurons_layer = shape[0]#矩阵的列数，即是前一层的输出
#             # skip the bias
#             if dim > 1 and unique_neurons_layer >= skip:
#                 import math
#                 temp = unique_neurons_layer * self.ration
#                 num_mutated = math.floor(temp) if temp > 2. else math.ceil(temp)
#                 mutated_neurons = np.random.choice(unique_neurons_layer,
#                                                    int(num_mutated), replace=False)
#                 m = mutated_neurons
#
#                 switch = copy.copy(mutated_neurons)
#                 np.random.shuffle(switch)
#
#                 param.data[mutated_neurons] = param.data[switch]
#                 if self.verbose:
#                     print(">>:mutated neurons in {0} layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
#
#                                                                            unique_neurons_layer))
#
#         return mutation_model
def change_weight_(filepath1,filepath2):
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
    # 循环创建文件
    if not os.path.exists(filepath2+'/'):  # 若不存在路径则创建
        os.makedirs(filepath2+'/')
    m = 5
    list = [1, 3, 5, 7]
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 1:
                        list_ = []
                        a = random.randint(0, 499)
                        b = random.randint(0, 783)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 499)
                            b = random.randint(0, 783)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                filepath2+'/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                filepath2+'/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        # if not os.path.isfile(filepath2+'/mutant_' + str(i) + '_' + str(a) + '_' + str(
                        #                b) + '/model_paras.pkl'):  # 无文件时创建
                        #     # os.mknod(filepath2+'/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                        #     #            b) + '/model_paras.pkl')
                        #     fd = open(filepath2+'/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/model_paras.pkl', mode="w", encoding="utf-8")
                        #     fd.close()
                        torch.save(net.state_dict(),
                                   filepath2+'/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
            i = 0
            net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 3:
                        print(i)
                        list_ = []
                        a = random.randint(0, 499)
                        b = random.randint(0, 499)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 499)
                            b = random.randint(0, 499)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        torch.save(net.state_dict(),
                                   filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
            i = 0
            net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 5:
                        list_ = []
                        a = random.randint(0, 499)
                        b = random.randint(0, 499)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 499)
                            b = random.randint(0, 499)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        torch.save(net.state_dict(),
                                   filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
            i = 0
            net.load_state_dict(torch.load(filepath1 + 'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 7:
                        list_ = []
                        a = random.randint(0, 9)
                        b = random.randint(0, 499)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 9)
                            b = random.randint(0, 499)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                filepath2 + 'mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        torch.save(net.state_dict(),
                                   filepath2 + '/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
def Weight_Shuffing_(filepath1,filepath2):
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
    # 循环创建文件
    if not os.path.exists(filepath2):  # 若不存在路径则创建
        os.makedirs(filepath2)
    dict = {1: [500,784], 3: [500,500], 5: [500,500], 7: [10,500]}
    list = [1, 3, 5]
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    a = random.randint(0, dict[i][0]-1)
                    b = random.randint(0, dict[i][1]-1)
                    x = copy.deepcopy(param[a][b].detach().numpy())
                    print(x)
                elif i == j + 2:
                    c = random.randint(0, dict[i][0]-1)
                    y = copy.deepcopy(param[c][a].detach().numpy())
                    print(y)
            i = 0
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    param[a][b] = torch.FloatTensor(y)
                    print(param[a][b])
                elif i == j + 2:
                    param[c][a] = torch.FloatTensor(x)
                    print(param[c][a])
                    if not os.path.exists(
                            filepath2+'mutant_' + str(i-2) + '_' +str(a)+'_'+ str(b) + '_' + str(i) + '_'+ str(c) + '_' + str(a) + '/'):  # 若不存在路径则创建
                        os.makedirs(filepath2+'mutant_' + str(i-2) + '_' +str(a)+'_'+ str(b) + '_' + str(i) + '_'+ str(c) + '_' + str(a) + '/')
                        torch.save(net.state_dict(),filepath2+'mutant_' + str(i-2) + '_' +str(a)+'_'+ str(b) + '_' + str(i) + '_'+ str(c) + '_' + str(a) + '/model_paras.pkl')
def Neuron_Effect_Blocking_(filepath1,filepath2):
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    # 循环创建文件
    if not os.path.exists(filepath2):  # 若不存在路径则创建
        os.makedirs(filepath2)
    dict = {1: [500, 784], 3: [500, 500], 5: [500, 500], 7: [10, 500]}
    list = [5]
    list_ = []
    flag = False
    p = 0
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    x = random.randint(0, dict[i][0]-1)
                    while not flag:
                        if [i,x] not in list_:
                            list_.append([i,x])
                            a = copy.deepcopy(np.zeros(param[x].detach().numpy().shape))
                            param[x] = torch.from_numpy(a)
                            flag = True
                            if j == 7:
                                p = p + 1
                            if j==7 and p>=10:
                                flag = True
                        else:
                            x = random.randint(0, dict[i][0] - 1)
                    flag = False
                elif i == j + 1:
                    a = copy.deepcopy(np.zeros(param[x].detach().numpy().shape))
                    param[x] = torch.FloatTensor(a)
                    #print(param[x])
                    if not os.path.exists(
                            filepath2+'/mutant_' + str(i-1) + '_' +str(x)+ '/'):  # 若不存在路径则创建
                        os.makedirs( filepath2+'/mutant_' + str(i-1) + '_' +str(x)+  '/')

                    torch.save(net.state_dict(), filepath2+'/mutant_' + str(i-1) + '_' +str(x)+  '/model_paras.pkl')
def Neuron_Switch_(filepath1,filepath2):
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    # 循环创建文件
    if not os.path.exists(filepath2):  # 若不存在路径则创建
        os.makedirs(filepath2)
    dict = {1: [500, 784], 3: [500, 500], 5: [500, 500], 7: [10, 500]}
    # list = [1, 3, 5, 7]
    list = [7]
    list_ = []
    flag = False
    p = 0
    #每层生成多少个变异体
    for k in range(100):
        #循环层数次数（list个数） 生成4个变异体
        for j in list:
            i = 0
            net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    x = random.randint(0, dict[i][0] - 1)
                    y = random.randint(0, dict[i][0] - 1)
                    while not flag:
                        if x == y:
                            x = random.randint(0, dict[i][0] - 1)
                            y = random.randint(0, dict[i][0] - 1)
                        elif [i,x,y] not in list_ and [i,y,x] not in list_:
                            list_.append([i,x,y])
                            list_.append([i, y, x])
                            a = copy.deepcopy(param[x].detach().numpy())
                            b = copy.deepcopy(param[y].detach().numpy())
                            param[x] = torch.from_numpy(b)
                            param[y] = torch.from_numpy(a)
                            flag = True
                            if j == 7:
                                p = p + 1
                            if j==7 and p>=45:
                                flag = True
                        else:
                            x = random.randint(0, dict[i][0] - 1)
                            y = random.randint(0, dict[i][0] - 1)
                    flag = False
                elif i == j + 1:
                    a = copy.deepcopy(param[x].detach().numpy())
                    b = copy.deepcopy(param[y].detach().numpy())
                    param[x] = torch.from_numpy(b)
                    param[y] = torch.from_numpy(a)
                    # print(param[x])
                    if not os.path.exists(
                            filepath2+'mutant_' + str(i - 1) + '_' + str(
                                x) + '_' + str(y) +'/'):  # 若不存在路径则创建
                        os.makedirs(
                            filepath2+'mutant_' + str(i - 1) + '_' + str(x) + '_' + str(y) +'/')

                    torch.save(net.state_dict(),
                               filepath2+'mutant_' + str(i - 1) + '_' + str(
                                   x) + '_' + str(y) +'/model_paras.pkl')
#修改循环次数
def Neuron_Activation_Inverse_(filepath1,filepath2):
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    # 循环创建文件
    if not os.path.exists(filepath2):  # 若不存在路径则创建
        os.makedirs(filepath2)
    dict = {1: [500, 784], 3: [500, 500], 5: [500, 500], 7: [10, 500]}
    list = [1, 3, 5, 7]
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load(filepath1+'model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    #if k == 0:
                        #resultlist1 = random.sample(range(0, dict[i][0] - 1), 100)
                    x = random.randint(0, dict[i][0] - 1)
                    a = copy.deepcopy(param[x].detach().numpy())
                    l = 0
                    for k in a:
                        a[l] = -k
                        l = l + 1
                    param[x] = torch.from_numpy(a)
                elif i == j + 1:
                    a = copy.deepcopy(param[x].detach().numpy())
                    param[x] = torch.FloatTensor([-a])
                    # print(param[x])
                    if not os.path.exists(
                            filepath2+'mutant_' + str(i - 1) + '_' + str(
                                x) + '/'):  # 若不存在路径则创建
                        os.makedirs(
                            filepath2+'mutant_' + str(i - 1) + '_' + str(x) + '/')

                    torch.save(net.state_dict(),
                               filepath2+'mutant_' + str(i - 1) + '_' + str(
                                   x) + '/model_paras.pkl')

def change_weight():
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    net.load_state_dict(torch.load('model_paras.pkl'))
    # 循环创建文件
    if not os.path.exists('/home/adv/jupyter_workspace/fault_localization/weight/'):  # 若不存在路径则创建
        os.makedirs('/home/adv/jupyter_workspace/fault_localization/weight/')
    m = 5
    list = [1, 3, 5, 7]
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 1:
                        list_ = []
                        a = random.randint(0, 499)
                        b = random.randint(0, 783)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 499)
                            b = random.randint(0, 783)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        torch.save(net.state_dict(),
                                   '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 3:
                        print(i)
                        list_ = []
                        a = random.randint(0, 499)
                        b = random.randint(0, 499)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 499)
                            b = random.randint(0, 499)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        torch.save(net.state_dict(),
                                   '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 5:
                        list_ = []
                        a = random.randint(0, 499)
                        b = random.randint(0, 499)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 499)
                            b = random.randint(0, 499)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        torch.save(net.state_dict(),
                                   '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    if i == 7:
                        list_ = []
                        a = random.randint(0, 9)
                        b = random.randint(0, 499)
                        if [a, b] not in list_:
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        else:
                            a = random.randint(0, 9)
                            b = random.randint(0, 499)
                            #param[a][b] = param[a][b] * random.uniform(-100, 100)
                            param[a][b] = param[a][b] * m
                            list_.append([a, b])
                        if not os.path.exists(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                        b) + '/'):  # 若不存在路径则创建
                            os.makedirs(
                                '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(b) + '/')
                        torch.save(net.state_dict(),
                                   '/home/adv/jupyter_workspace/fault_localization/weight/mutant_' + str(i) + '_' + str(a) + '_' + str(
                                       b) + '/model_paras.pkl')
def Weight_Shuffing():
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    net.load_state_dict(torch.load('model_paras.pkl'))
    # 循环创建文件
    if not os.path.exists('/home/adv/jupyter_workspace/fault_localization/weight_shuffing/'):  # 若不存在路径则创建
        os.makedirs('/home/adv/jupyter_workspace/fault_localization/weight_shuffing/')
    dict = {1: [500,784], 3: [500,500], 5: [500,500], 7: [10,500]}
    list = [1, 3, 5]
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    a = random.randint(0, dict[i][0]-1)
                    b = random.randint(0, dict[i][1]-1)
                    x = copy.deepcopy(param[a][b].detach().numpy())
                    print(x)
                elif i == j + 2:
                    c = random.randint(0, dict[i][0]-1)
                    y = copy.deepcopy(param[c][a].detach().numpy())
                    print(y)
            i = 0
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    param[a][b] = torch.FloatTensor(y)
                    print(param[a][b])
                elif i == j + 2:
                    param[c][a] = torch.FloatTensor(x)
                    print(param[c][a])
                    if not os.path.exists(
                            '/home/adv/jupyter_workspace/fault_localization/weight_shuffing/mutant_' + str(i-2) + '_' +str(a)+'_'+ str(b) + '_' + str(i) + '_'+ str(c) + '_' + str(a) + '/'):  # 若不存在路径则创建
                        os.makedirs('/home/adv/jupyter_workspace/fault_localization/weight_shuffing/mutant_' + str(i-2) + '_' +str(a)+'_'+ str(b) + '_' + str(i) + '_'+ str(c) + '_' + str(a) + '/')
                        torch.save(net.state_dict(),'/home/adv/jupyter_workspace/fault_localization/weight_shuffing/mutant_' + str(i-2) + '_' +str(a)+'_'+ str(b) + '_' + str(i) + '_'+ str(c) + '_' + str(a) + '/model_paras.pkl')
def Neuron_Effect_Blocking():
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    # 循环创建文件
    if not os.path.exists('E:/fault_localization/Neuron_Effect_Blocking/'):  # 若不存在路径则创建
        os.makedirs('E:/fault_localization/Neuron_Effect_Blocking/')
    dict = {1: [500, 784], 3: [500, 500], 5: [500, 500], 7: [10, 500]}
    list = [5]
    list_ = []
    flag = False
    p = 0
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    x = random.randint(0, dict[i][0]-1)
                    while not flag:
                        if [i,x] not in list_:
                            list_.append([i,x])
                            a = copy.deepcopy(np.zeros(param[x].detach().numpy().shape))
                            param[x] = torch.from_numpy(a)
                            flag = True
                            if j == 7:
                                p = p + 1
                            if j==7 and p>=10:
                                flag = True
                        else:
                            x = random.randint(0, dict[i][0] - 1)
                    flag = False
                elif i == j + 1:
                    a = copy.deepcopy(np.zeros(param[x].detach().numpy().shape))
                    param[x] = torch.FloatTensor(a)
                    #print(param[x])
                    if not os.path.exists(
                            'E:/fault_localization/Neuron_Effect_Blocking/mutant_' + str(i-1) + '_' +str(x)+ '/'):  # 若不存在路径则创建
                        os.makedirs('E:/fault_localization/Neuron_Effect_Blocking/mutant_' + str(i-1) + '_' +str(x)+  '/')

                    torch.save(net.state_dict(),'E:/fault_localization/Neuron_Effect_Blocking/mutant_' + str(i-1) + '_' +str(x)+  '/model_paras.pkl')
def Neuron_Switch():
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    # 循环创建文件
    if not os.path.exists('/home/adv/jupyter_workspace/fault_localization/Neuron_Switch/'):  # 若不存在路径则创建
        os.makedirs('/home/adv/jupyter_workspace/fault_localization/Neuron_Switch/')
    dict = {1: [500, 784], 3: [500, 500], 5: [500, 500], 7: [10, 500]}
    # list = [1, 3, 5, 7]
    list = [7]
    list_ = []
    flag = False
    p = 0
    #每层生成多少个变异体
    for k in range(100):
        #循环层数次数（list个数） 生成4个变异体
        for j in list:
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    x = random.randint(0, dict[i][0] - 1)
                    y = random.randint(0, dict[i][0] - 1)
                    while not flag:
                        if x == y:
                            x = random.randint(0, dict[i][0] - 1)
                            y = random.randint(0, dict[i][0] - 1)
                        elif [i,x,y] not in list_ and [i,y,x] not in list_:
                            list_.append([i,x,y])
                            list_.append([i, y, x])
                            a = copy.deepcopy(param[x].detach().numpy())
                            b = copy.deepcopy(param[y].detach().numpy())
                            param[x] = torch.from_numpy(b)
                            param[y] = torch.from_numpy(a)
                            flag = True
                            if j == 7:
                                p = p + 1
                            if j==7 and p>=45:
                                flag = True
                        else:
                            x = random.randint(0, dict[i][0] - 1)
                            y = random.randint(0, dict[i][0] - 1)
                    flag = False
                elif i == j + 1:
                    a = copy.deepcopy(param[x].detach().numpy())
                    b = copy.deepcopy(param[y].detach().numpy())
                    param[x] = torch.from_numpy(b)
                    param[y] = torch.from_numpy(a)
                    # print(param[x])
                    if not os.path.exists(
                            '/home/adv/jupyter_workspace/fault_localization/Neuron_Switch/mutant_' + str(i - 1) + '_' + str(
                                x) + '_' + str(y) +'/'):  # 若不存在路径则创建
                        os.makedirs(
                            '/home/adv/jupyter_workspace/fault_localization/Neuron_Switch/mutant_' + str(i - 1) + '_' + str(x) + '_' + str(y) +'/')

                    torch.save(net.state_dict(),
                               '/home/adv/jupyter_workspace/fault_localization/Neuron_Switch/mutant_' + str(i - 1) + '_' + str(
                                   x) + '_' + str(y) +'/model_paras.pkl')
#修改循环次数
def Neuron_Activation_Inverse():
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    # 循环创建文件
    if not os.path.exists('/home/adv/jupyter_workspace/fault_localization/Neuron_Activation_Inverse/'):  # 若不存在路径则创建
        os.makedirs('/home/adv/jupyter_workspace/fault_localization/Neuron_Activation_Inverse/')
    dict = {1: [500, 784], 3: [500, 500], 5: [500, 500], 7: [10, 500]}
    list = [1, 3, 5, 7]
    for k in range(100):
        for j in list:
            i = 0
            net.load_state_dict(torch.load('model_paras.pkl'))
            for name, param in net.named_parameters():
                i = i + 1
                if i == j:
                    #if k == 0:
                        #resultlist1 = random.sample(range(0, dict[i][0] - 1), 100)
                    x = random.randint(0, dict[i][0] - 1)
                    a = copy.deepcopy(param[x].detach().numpy())
                    l = 0
                    for k in a:
                        a[l] = -k
                        l = l + 1
                    param[x] = torch.from_numpy(a)
                elif i == j + 1:
                    a = copy.deepcopy(param[x].detach().numpy())
                    param[x] = torch.FloatTensor([-a])
                    # print(param[x])
                    if not os.path.exists(
                            '/home/adv/jupyter_workspace/fault_localization/Neuron_Activation_Inverse/mutant_' + str(i - 1) + '_' + str(
                                x) + '/'):  # 若不存在路径则创建
                        os.makedirs(
                            '/home/adv/jupyter_workspace/fault_localization/Neuron_Activation_Inverse/mutant_' + str(i - 1) + '_' + str(x) + '/')

                    torch.save(net.state_dict(),
                               '/home/adv/jupyter_workspace/fault_localization/Neuron_Activation_Inverse/mutant_' + str(i - 1) + '_' + str(
                                   x) + '/model_paras.pkl')


if __name__ == '__main__':
    Neuron_Activation_Inverse()
    Neuron_Switch()
    Weight_Shuffing()
    Neuron_Effect_Blocking()
    change_weight()
    # 交换同层的随机两个神经元
    # x = random.randint(0, 10)
    # y = random.randint(0, 10)
    # print(x, y)
    # for name, param in net.named_parameters():
    #     i = i + 1
    #     if i == 7:
    #         a = copy.deepcopy(param[x].detach().numpy())
    #         b = copy.deepcopy(param[y].detach().numpy())
    #         param[x] = torch.from_numpy(b)
    #         param[y] = torch.from_numpy(a)
    #         print('******')
    #     elif i == 8:
    #         a = copy.deepcopy(param[x].detach().numpy())
    #         b = copy.deepcopy(param[y].detach().numpy())
    #         param[x] = torch.from_numpy(b)
    #         param[y] = torch.from_numpy(a)
    #逆转神经元激活状态
    # x = random.randint(0, 9)
    # print(x)
    # for name, param in net.named_parameters():
    #     i = i + 1
    #     if i == 7:
    #         a = copy.deepcopy(param[x].detach().numpy())
    #         j = 0
    #         for m in a:
    #             a[j] = -m
    #             j = j + 1
    #         param[x] = torch.from_numpy(a)
    #         print('******')
    #     elif i == 8:
    #         a = copy.deepcopy(param[x].detach().numpy())
    #         print(a)
    #         param[x] = torch.FloatTensor([-a])
    #         print(param[x])
    #阻断神经元向下一层的传播,不是将连接权重变为0，而是将输出变为0
    # x = random.randint(0, 9)
    # print(x)
    # for name, param in net.named_parameters():
    #     i = i + 1
    #     if i == 7:
    #         a = copy.deepcopy(np.zeros(param[x].detach().numpy().shape))
    #         a = a + 0.01
    #         param[x] = torch.from_numpy(a)
    #         print('******')
    #     elif i == 8:
    #         a = copy.deepcopy(np.zeros(param[x].detach().numpy().shape))
    #         param[x] = torch.FloatTensor(a)
    #         print(param[x])
    # operator = MutaionOperator(ration=0.1, model=net)
    # operator.ns()
    # torch.save(net.state_dict(), 'mutants/model_paras.pkl')

