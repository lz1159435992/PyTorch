import torch.nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
net = torch.nn.Sequential(
OrderedDict([
    ("hidden",torch.nn.Linear(2, 3)),
    ("relu", torch.nn.ReLU()),
    ("out",torch.nn.Linear(3, 2)),
])
)
net.load_state_dict(torch.load('net_paras2.pkl'))
for name,param in net.named_parameters():
    print(name,param)
# i = 0
# n_data = torch.ones(100, 2)
# x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
# y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
# x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
# y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
# npx = x.detach().numpy()
# npy = y.detach().numpy()
# np.savetxt('x.txt', npx)
# np.savetxt('y.txt', npy)
# x9 = x[0:1,:]
# x8 = x[100:101,:]
# x00 = torch.cat((x9,x8),0)
# npx0 = x00.detach().numpy()
# np.savetxt('npx0.txt',npx0)
# x000 = np.loadtxt('npx0.txt')
x = np.loadtxt('x.txt')
x = torch.from_numpy(x)
x = torch.tensor(x, dtype=torch.float32)
# x000 = torch.from_numpy(x000)
# x000 = torch.tensor(x000, dtype=torch.float32)
print(x)
print('******************')
print(net.hidden(x))
print('******************')
print(net.relu(net.hidden(x)))
print('******************')
y0 = net.out(net.relu(net.hidden(x)))
print(net.out(net.relu(net.hidden(x))))
print('******************')
prediction = torch.max(F.softmax(y0), 1)[1]
pred_y = prediction.data.numpy().squeeze()
print(prediction)
np.savetxt('prediction.txt', prediction);