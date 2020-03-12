import torch
import torch.nn as nn
import classifier
import torchvision.models as models
# net = models.resnet18(pretrained=False)
# pthfile = r'C:\Users\Zheng\.torch\models\resnet18-5c106cde.pth'
# net.load_state_dict(torch.load(pthfile))
# print(net)
model = torch.load('net.pkl')
print(model)
for param in model.parameters():
    for p in param:
        print(p)
for name in model.state_dict():
    print(name)
#Then  I konw that the name of target layer is '1.weight'

#schemem1(recommended)

#scheme2
params = list(model.named_parameters())#get the index by debuging
#print(params[2][0])#name
#print(params[2][1].data)#data
#print('111111111')
#scheme3
params = {}#change the tpye of 'generator' into dict
for name,param in model.named_parameters():
    #print(name)
    params[name] = param.detach().cpu().numpy()
    #print(params['0.weight'])
#print('11111111')
#scheme4
for layer in model.modules():
    if(isinstance(layer,nn.Conv3d)):
        print(layer.weight)

#打印每一层的参数名和参数值
#schemem1(recommended)
for name,param in model.named_parameters():
    print(name,param)

#scheme2
for name in model.state_dict():
    print(name)
    print(model.state_dict()[name])

# pthfile = r'C:\Users\Zheng\PycharmProjects\PyTorch\learning\cnn1.pkl'
# # net = torch.load(pthfile)
# # print(net)
# EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
# BATCH_SIZE = 50
# LR = 0.001
# net = nn.Sequential(
# nn.Conv2d(
#                 in_channels=1,              # input height
#                 out_channels=16,            # n_filters
#                 kernel_size=5,              # filter size
#                 stride=1,                   # filter movement/step
#                 padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
#             ),                              # output shape (16, 28, 28)
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(kernel_size=2),    # choose max
# nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(2),
# nn.Linear(32 * 7 * 7, 10),
# )
# net.load_state_dict(torch.load(pthfile))
# print(net)