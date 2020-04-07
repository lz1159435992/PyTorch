import torch
import torch.nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import copy
net = torch.nn.Sequential(
OrderedDict([
    ("hidden",torch.nn.Linear(2, 3)),
    ("relu", torch.nn.ReLU()),
    ("out",torch.nn.Linear(3, 2)),
])
)
net.load_state_dict(torch.load('net_paras2.pkl'))
#print(net)
i = 0
#将第一层第二个神经元第一个权重与第二层第一个神经元第二个权重其中一个权重交换
for name, param in net.named_parameters():
    i = i + 1
    if i == 1:
        a = copy.deepcopy(param[1][0].detach().numpy())
    elif i == 3:
        b = copy.deepcopy(param[0][1].detach().numpy())
i = 0
for name, param in net.named_parameters():
    i = i + 1
    if i == 1:
        param[1][0] = torch.FloatTensor(b)
    elif i == 3:
        param[0][1] = torch.FloatTensor(a)
# # 将第一层第二个神经元的输出反转
# for name, param in net.named_parameters():
#     i = i + 1
#     if i == 1:
#         a = copy.deepcopy(param[1][0].detach().numpy())
#         b = copy.deepcopy(param[1][1].detach().numpy())
#         param[1] = torch.FloatTensor([-a, -b])
#         print('******')
#     elif i == 2:
#         a = copy.deepcopy(param[1].detach().numpy())
#         param[1] = torch.FloatTensor([-a])
# 交换第一层的前两个神经元
# for name, param in net.named_parameters():
#     i = i + 1
#     if i == 1:
#         a = copy.deepcopy(param[0].detach().numpy())
#         b = copy.deepcopy(param[1].detach().numpy())
#         param[0] = torch.from_numpy(b)
#         param[1] = torch.from_numpy(a)
#         print('******')
#     elif i == 2:
#         a = copy.deepcopy(param[0].detach().numpy())
#         b = copy.deepcopy(param[1].detach().numpy())
#         param[0] = torch.from_numpy(b)
#         param[1] = torch.from_numpy(a)
i = 0
for name,param in net.named_parameters():
    #将第一个权重变为10
    # i= i+1
    # if i == 1:
    #     param[0][0] = 10
    print(name, param)
# n_data = torch.ones(100, 2)
# x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
# y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
# x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
# y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
# x9 = x[0:1,:]
# x8 = x[100:101,:]
# x00 = torch.cat((x9,x8),0)
# npx0 = x00.detach().numpy()
# np.savetxt('npx0.txt',npx0)
x000 = np.loadtxt('metamorphic/x.txt')
x000 = torch.from_numpy(x000)
x000 = torch.tensor(x000, dtype=torch.float32)
print(x000)
print('******************')
print(net.hidden(x000))
print('******************')
print(net.relu(net.hidden(x000)))
print('******************')
y0 = net.out(net.relu(net.hidden(x000)))
# y0 = net.out(net.hidden(x000))
print(net.out(net.relu(net.hidden(x000))))
print('******************')
prediction = torch.max(F.softmax(y0), 1)[1]
pred_y = prediction.data.numpy().squeeze()
print(prediction)