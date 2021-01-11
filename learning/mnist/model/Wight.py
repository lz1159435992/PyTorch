import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import os
import numpy as np
from collections import OrderedDict
def get_weight():
    net = torch.nn.Sequential(
    OrderedDict([
        ("fc1", torch.nn.Linear(784, 500)),
        ("fc2", torch.nn.Linear(500, 500)),
        ("fc3", torch.nn.Linear(500, 500)),
        ("relu", torch.nn.ReLU()),
        ("output", torch.nn.Linear(500, 10)),
    ])
    )
    net.load_state_dict(torch.load('mutants/model_paras.pkl'))
    for name,param in net.named_parameters():
        #将第一个权重变为10
        # i= i+1
        # if i == 1:
        #     param[0][0] = 10
        print(name, param)
        a = param.detach().numpy()
        np.savetxt('mutants/weight/'+name+'.txt', a)
        print(param.size())

def get_weight_(filepath):
    net = torch.nn.Sequential(
    OrderedDict([
        ("fc1", torch.nn.Linear(784, 500)),
        ("fc2", torch.nn.Linear(500, 500)),
        ("fc3", torch.nn.Linear(500, 500)),
        ("relu", torch.nn.ReLU()),
        ("output", torch.nn.Linear(500, 10)),
    ])
    )
    net.load_state_dict(torch.load(filepath+'/model_paras.pkl'))
    if not os.path.exists(filepath+'/weight/'):  # 若不存在路径则创建
        os.makedirs(filepath+'/weight/')
    for name,param in net.named_parameters():
        #将第一个权重变为10
        # i= i+1
        # if i == 1:
        #     param[0][0] = 10
        #print(name, param)
        a = param.detach().numpy()
        np.savetxt(filepath+'/weight/'+name+'.txt', a)
        print(param.size())
#dirpath为根目录，filenames为文件列表
def get_proportion():
    for dirpath, dirnames, filenames in os.walk('mutants/weight'):
        for filename in filenames:
            if 'weight' in filename:
                a = np.loadtxt(dirpath+'/'+filename)
                for i in a:
                    x = 0
                    sum = 0
                    for j in i:
                        sum = sum + j
                    k = 0#判断是否到了最后一个
                    for j in i:
                        k = k + 1
                        if k == len(i):
                            j = 1 - x#防止比例不为1
                        else:
                            j = j / sum
                        x = x + j
                    print(x)
                np.savetxt(dirpath+'_proportion/proportion.'+filename, a)
def get_proportion_(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'/weight'):
        for filename in filenames:
            if 'weight' in filename:
                a = np.loadtxt(dirpath+'/'+filename)
                for i in a:
                    x = 0
                    sum = 0
                    for j in i:
                        sum = sum + j
                    k = 0#判断是否到了最后一个
                    for j in i:
                        #print(j.shape)
                        k = k + 1
                        if k == len(i):
                            j = 1 - x#防止比例不为1
                        else:
                            #解决分母为0的情况
                            if sum == 0:
                                j = 0
                            else:
                                j = j / sum
                        x = x + j
                if not os.path.exists(dirpath+'_proportion/'):  # 若不存在路径则创建
                    os.makedirs(dirpath+'_proportion/')
                np.savetxt(dirpath+'_proportion/proportion.'+filename, a)
# get_weight()
# get_proportion()
# a = np.loadtxt('weight/proportion.fc1.weight.txt')
# for i in a:
#     sum = 0
#     for j in i:
#         sum = sum + j
#         print(sum)
# print(sum)
# for i in range(0,500):
#     sum = 0
#     for j in a[i]:
#         sum = sum + j
#     for j in a[i]:
#         j = j/sum
# print(a)
# while i < a[0].size()
