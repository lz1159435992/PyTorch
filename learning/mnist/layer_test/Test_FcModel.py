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
net.load_state_dict(torch.load('paras_mutant/model_paras.pkl'))
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
c = torch.empty(2, 3)
i = 0
for images, labels in test_loader:
    i=i+1
    images = images.reshape(-1, 28*28)
    #print(images)
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
    #获取预测标签和实际标签
np.savetxt('predicted/output/labels.txt',b)
np.savetxt('predicted/output/predicted.txt', a)
np.savetxt('predicted/output/predicted_10.txt',c.detach().numpy())
print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))