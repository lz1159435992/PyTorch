import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import numpy as np
from collections import OrderedDict
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
# for name,param in net.named_parameters():
#     #将第一个权重变为10
#     # i= i+1
#     # if i == 1:
#     #     param[0][0] = 10
#     print(name, param)
#     print(param.size())
# print(net)
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
a = torch.empty(2, 3)#创建一个tensor对象来存储预测结果
b = torch.empty(2, 3)
i = 0
for images, labels in test_loader:
    i=i+1
    images = images.reshape(-1, 28*28)
    #print(images)
    z = net.fc1(images)
    x = net.fc2(z)
    c = net.fc3(x)
    v = net.relu(c)
    r = net.output(v)
    print(z.shape)
    #print (net.fc1(images))
    #print(net.fc1(images))
    output = net(images)
    #print(output)
    _, predicted = torch.max(output, 1)
    #print(predicted)
    print('****************')
    if i == 1:
        a = predicted
        b = labels
    else:
        a = torch.cat([a, predicted], dim=0)
        b = torch.cat([b, labels], dim=0)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    #获取预测标签和实际标签
np.savetxt('mutants/predicted/labels.txt',b)
np.savetxt('mutants/predicted/predicted.txt', a)
print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))