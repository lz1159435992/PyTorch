import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch
import numpy as np
from collections import OrderedDict
import os
import math
import random
def processing(net,filepath,m):
    train_set = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    train_loader = dataloader.DataLoader(
        dataset=train_set,
        batch_size=100,
        shuffle=False,
    )

    test_set = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    test_loader = dataloader.DataLoader(
        dataset=test_set,
        batch_size=100,
        shuffle=False,
    )
    correct = 0
    total = 0
    a = torch.empty(2, 3)  # 创建一个tensor对象来存储预测结果
    b = torch.empty(2, 3)
    c = torch.empty(2, 3)
    i = 0
    for images, labels in test_loader:
        i = i + 1
        images = images.reshape(-1, 28 * 28)
        # print(images)
        output = net(images)
        _, predicted = torch.max(output, 1)
        if i == 1:
            a = predicted
            b = labels
            c = output
        else:
            a = torch.cat([a, predicted], dim=0)
            b = torch.cat([b, labels], dim=0)
            c = torch.cat([c, output], dim=0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 获取预测标签和实际标签
    # np.savetxt('predicted/all1/labels.txt',b)
    # np.savetxt('predicted/all1/predicted.txt', a)
    if not os.path.exists(filepath + '\\output'+str(m)+'\\'):
        os.makedirs(filepath + '\\output'+str(m)+'\\')
    np.savetxt(filepath + '/output'+str(m)+'/predicted_10.txt', c.detach().numpy())
    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
def Layer_Deactivation():
    if not os.path.exists('D:\\fault_localization\\Layer_Deactivation'):
        os.makedirs('D:\\fault_localization\\Layer_Deactivation')
    if not os.path.exists('D:\\fault_localization\\Layer_Deactivation\\del_fc2'):
        os.makedirs('D:\\fault_localization\\Layer_Deactivation\\del_fc2')
    filepath = 'D:\\fault_localization\\Layer_Deactivation\\del_fc2'
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            #("fc2", torch.nn.Linear(500, 500)),
            # ("new_layer", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ]))
    a = torch.load('model_paras.pkl')
    # a['new_layer.weight'] = a['fc3.weight']
    # a['new_layer.bias'] = a['fc3.bias']
    del a['fc2.weight']
    del a['fc2.bias']
    net.load_state_dict(a)
    z = 0
    x = 0
    for name,param in net.named_parameters():
        x = x + 1
        # a = np.ones(param.shape)
        # param = a
        if x%2 == 1:
            i = 0
            for m in param:
                j = 0
                for n in m:
                    param[i][j] = 0.01
                    j = j + 1
                i = i + 1
        else:
            i = 0
            for m in param:
                param[i] = 0.01
                i = i + 1
    processing(net,filepath,z)
    z = z + 1
    net.load_state_dict(a)
    x = 0
    #添加第一层
    for name,param in net.named_parameters():
        x = x + 1
        # a = np.ones(param.shape)
        # param = a
        if x == 1 or x == 2:
            print(param)
        elif x == 3 or x==5 or x == 7 or x == 9:
            i = 0
            for m in param:
                j = 0
                for n in m:
                    param[i][j] = 0.01
                    j = j + 1
                i = i + 1
        else:
            i = 0
            for m in param:
                param[i] = 0.01
                i = i + 1
    processing(net,filepath,z)
    z = z + 1
    net.load_state_dict(a)
    x = 0
    # print(param,param.shape)
    # 将第二层的权重和偏置添加进去，其实实现的时候就是对第一第二层的权重和偏置不做操作
    for name,param in net.named_parameters():
        x = x + 1
        if x==1 or x == 2 or x == 3 or x == 4:
            print(param)
        elif x==5 or x == 7 or x == 9:
            i = 0
            for m in param:
                j = 0
                for n in m:
                    param[i][j] = 0.01
                    j = j + 1
                i = i + 1
        else:
            i = 0
            for m in param:
                param[i] = 0.01
                i = i + 1
    processing(net, filepath, z)
    z = z + 1
    #第三层不做操作
    net.load_state_dict(a)
    processing(net,filepath,z)
    #删除第三层
    if not os.path.exists('D:\\fault_localization\\Layer_Deactivation\\del_fc3'):
        os.makedirs('D:\\fault_localization\\Layer_Deactivation\\del_fc3')
    filepath = 'D:\\fault_localization\\Layer_Deactivation\\del_fc3'
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            #("fc2", torch.nn.Linear(500, 500)),
            # ("new_layer", torch.nn.Linear(500, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ]))
    a = torch.load('model_paras.pkl')
    # a['new_layer.weight'] = a['fc3.weight']
    # a['new_layer.bias'] = a['fc3.bias']
    del a['fc3.weight']
    del a['fc3.bias']
    net.load_state_dict(a)
    z = 0
    x = 0
    for name,param in net.named_parameters():
        x = x + 1
        # a = np.ones(param.shape)
        # param = a
        if x%2 == 1:
            i = 0
            for m in param:
                j = 0
                for n in m:
                    param[i][j] = 0.01
                    j = j + 1
                i = i + 1
        else:
            i = 0
            for m in param:
                param[i] = 0.01
                i = i + 1
    processing(net,filepath,z)
    z = z + 1
    net.load_state_dict(a)
    x = 0
    #添加第一层
    for name,param in net.named_parameters():
        x = x + 1
        # a = np.ones(param.shape)
        # param = a
        if x == 1 or x == 2:
            print(param)
        elif x == 3 or x==5 or x == 7 or x == 9:
            i = 0
            for m in param:
                j = 0
                for n in m:
                    param[i][j] = 0.01
                    j = j + 1
                i = i + 1
        else:
            i = 0
            for m in param:
                param[i] = 0.01
                i = i + 1
    processing(net,filepath,z)
    z = z + 1
    net.load_state_dict(a)
    x = 0
    # print(param,param.shape)
    # 将第二层的权重和偏置添加进去，其实实现的时候就是对第一第二层的权重和偏置不做操作
    for name,param in net.named_parameters():
        x = x + 1
        if x==1 or x == 2 or x == 3 or x == 4:
            print(param)
        elif x==5 or x == 7 or x == 9:
            i = 0
            for m in param:
                j = 0
                for n in m:
                    param[i][j] = 0.01
                    j = j + 1
                i = i + 1
        else:
            i = 0
            for m in param:
                param[i] = 0.01
                i = i + 1
    processing(net, filepath, z)
    z = z + 1
    #第三层不做操作
    net.load_state_dict(a)
    processing(net,filepath,z)
#计算单个层的怀疑度
def calculate(labelpath,firstpath,secondpath):
    labels = np.loadtxt(labelpath)
    first = np.loadtxt(firstpath)
    second = np.loadtxt(secondpath)
    x = 0
    t = 0
    f = 0
    m = 0
    for i in labels:
        a = 0
        b = 0
        for j in range(10):
            try:
                c = math.exp(first[x][j])
            except OverflowError:
                c = float('inf')
            try:
                d = math.exp(second[x][j])
            except OverflowError:
                d = float('inf')
            # a = a + math.exp(all1_predicted_10[x][j])
            a = a + c
            # b = b + math.exp(fc1_predicted_10[x][j])
            b = b + d
        try:
            c = math.exp(first[x][int(i)])
        except OverflowError:
            c = float('inf')
        try:
            d = math.exp(second[x][int(i)])
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
    print(t, m, f)
    score = f ** 3 / (t + f)
    print('怀疑度', score)
    return score
def layer_calculate(filepath):
    list = []
    for dirpath, dirnames, filenames in os.walk(filepath + '\\'):
        if filenames:
            if 'predicted' in filenames[0]:
                list.append(dirpath+'\\'+filenames[0])
                print(dirpath, filenames)
    for i in range(len(list)-1):
        a = calculate(filepath+'\\labels.txt',list[i],list[i+1])
    print(list)
if __name__ == '__main__':
    #Layer_Deactivation()
    layer_calculate('D:\\fault_localization\Layer_Deactivation\\del_fc3')
    # 将第三层的权重和偏置添加进去，其实实现的时候就是对第一第二第三层的权重和偏置不做操作
    # for name,param in net.named_parameters():
    #     x = x + 1
    #     if x == 1 or x == 2 or x == 3 or x == 4 or x==5 or x==6:
    #         print(param)
    #     elif x == 7 or x == 9:
    #         i = 0
    #         for m in param:
    #             j = 0
    #             for n in m:
    #                 param[i][j] = 0.01
    #                 j = j + 1
    #             i = i + 1
    #     else:
    #         i = 0
    #         for m in param:
    #             param[i] = 0.01
    #             i = i + 1
    # 加入第四层
    # for name,param in net.named_parameters():
    #     x = x + 1
    #     if x == 1 or x == 2 or x == 3 or x == 4 or x==5 or x==6 or x == 7 or x == 8:
    #         print(param)
    #     elif x == 9:
    #         i = 0
    #         for m in param:
    #             j = 0
    #             for n in m:
    #                 param[i][j] = 0.01
    #                 j = j + 1
    #             i = i + 1
    #     else:
    #         i = 0
    #         for m in param:
    #             param[i] = 0.01
    #             i = i + 1
# net = torch.nn.Sequential(
#         OrderedDict([
#             ("fc1", torch.nn.Linear(784, 500)),
#             ("fc2", torch.nn.Linear(500, 500)),
#             #("new_layer", torch.nn.Linear(500, 500)),
#             ("fc3", torch.nn.Linear(500, 500)),
#             ("relu", torch.nn.ReLU()),
#             ("output", torch.nn.Linear(500, 10)),
#         ])
#     )
# a = torch.load('model_paras.pkl')
# #a['new_layer.weight'] = a['fc3.weight']
# #a['new_layer.bias'] = a['fc3.bias']
# # del a['fc2.weight']
# # del a['fc2.bias']
# net.load_state_dict(a)
# x = 0
# print('**********************************************************************************************************************')
#将权重和偏置都设为0.01
# for name,param in net.named_parameters():
#     x = x + 1
#     # a = np.ones(param.shape)
#     # param = a
#     if x%2 == 1:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 0.01
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 0.01
#             i = i + 1
#将第一层的权重和偏置添加进去，其实实现的时候就是对第一层的权重和偏置不做操作
# for name,param in net.named_parameters():
#     x = x + 1
#     # a = np.ones(param.shape)
#     # param = a
#     if x == 1 or x == 2:
#         print(param)
#     elif x == 3 or x==5 or x == 7 or x == 9:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 0.01
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 0.01
#             i = i + 1
    # print(param,param.shape)
#将第二层的权重和偏置添加进去，其实实现的时候就是对第一第二层的权重和偏置不做操作
# for name,param in net.named_parameters():
#     x = x + 1
#     if x==1 or x == 2 or x == 3 or x == 4:
#         print(param)
#     elif x==5 or x == 7 or x == 9:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 0.01
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 0.01
#             i = i + 1
#将第三层的权重和偏置添加进去，其实实现的时候就是对第一第二第三层的权重和偏置不做操作
# for name,param in net.named_parameters():
#     x = x + 1
#     if x == 1 or x == 2 or x == 3 or x == 4 or x==5 or x==6:
#         print(param)
#     elif x == 7 or x == 9:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 0.01
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 0.01
#             i = i + 1
#加入第四层
# for name,param in net.named_parameters():
#     x = x + 1
#     if x == 1 or x == 2 or x == 3 or x == 4 or x==5 or x==6 or x == 7 or x == 8:
#         print(param)
#     elif x == 9:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 0.01
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 0.01
#             i = i + 1
#原始的缺陷语句
# for name,param in net.named_parameters():
#     x = x + 1
#     if x == 1:
#         # 加入缺陷
#         param[100][100] = 50
#     elif x == 2:
#         param[100] = 50
# print(net.state_dict())
# torch.save(net.state_dict(), 'paras_mutant/model_paras.pkl')