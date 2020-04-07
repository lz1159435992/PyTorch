import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch
import numpy as np
from collections import OrderedDict
import random
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
x = 0
print('**********************************************************************************************************************')
#将权重和偏置都设为1
# for name,param in net.named_parameters():
#     x = x + 1
#     # a = np.ones(param.shape)
#     # param = a
#     if x%2 == 1:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 1
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 1
#             i = i + 1
#将第一层的权重和偏置添加进去，其实实现的时候就是对第一层的权重和偏置不做操作
# for name,param in net.named_parameters():
#     x = x + 1
#     # a = np.ones(param.shape)
#     # param = a
#     if x == 1 or x == 2:
#         print(param)
#     elif x == 3 or x==5 or x == 7:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 1
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 1
#             i = i + 1
    # print(param,param.shape)
#将第二层的权重和偏置添加进去，其实实现的时候就是对第一第二层的权重和偏置不做操作
# for name,param in net.named_parameters():
#     x = x + 1
#     # a = np.ones(param.shape)
#     # param = a
#     if x == 1 or x == 2 or x == 3 or x == 4:
#         print(param)
#     elif x==5 or x == 7:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 1
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 1
#             i = i + 1
#将第三层的权重和偏置添加进去，其实实现的时候就是对第一第二第三层的权重和偏置不做操作
# for name,param in net.named_parameters():
#     x = x + 1
#     # a = np.ones(param.shape)
#     # param = a
#     if x == 1 or x == 2 or x == 3 or x == 4 or x==5 or x==6:
#         print(param)
#     elif x == 7:
#         i = 0
#         for m in param:
#             j = 0
#             for n in m:
#                 param[i][j] = 1
#                 j = j + 1
#             i = i + 1
#     else:
#         i = 0
#         for m in param:
#             param[i] = 1
#             i = i + 1
print(net.state_dict())
torch.save(net.state_dict(), 'paras_mutant/model_paras.pkl')