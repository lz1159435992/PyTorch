from learning.mnist.model import Wight
import os
from learning.mnist.model import Get_difference
from learning.mnist.model import Score
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import numpy as np
from learning.mnist.model import excel
import math
from collections import OrderedDict
from sklearn import preprocessing
def Normalize(data):
 shape = data.shape
 data = data.reshape(-1, 1)
 m = np.mean(data)
 mx = data.max()
 mn = data.min()
 data = preprocessing.minmax_scale(data, feature_range=(-1,1))
 return data.reshape(shape)
def get_failed_data_(filepath):
    a = np.loadtxt('D:/fault_localization/origin_predicted/predicted.txt')
    # a = np.loadtxt(filepath + '/predicted/labels.txt')
    b = np.loadtxt(filepath + '/predicted/predicted.txt')
    flag = {}
    i = 0
    j = 0
    while i < a.size:
        if a[i]!=b[i]:
            flag[i] = b[i]
            j = j + 1
        i=i+1
    print(j)
    #不一致的结果的位置为key，值为错误的分类标签
    return flag
#计算错误分类个数
def calculate_failed_data(filepath):
    failed_data = {}
    flag = get_failed_data_(filepath)
    for key, value in flag.items():
        if value not in failed_data:
            failed_data[value] = 1
        else:
            failed_data[value] = failed_data[value] + 1
    #各个分类结果中，错误的个数
    return failed_data
def calculate_passed_data(filepath):
    a = np.loadtxt(filepath + '/predicted/predicted.txt')
    j = 0
    failed_data = get_failed_data_(filepath)
    passed_data = {}
    for i in a:
        j = j + 1
        #j是位置
        if j not in failed_data:
            if i not in passed_data:
                passed_data[i] = 1
            else:
                passed_data[i] = passed_data[i]+1
    #各个分类结果中，正确的个数
    return passed_data
def calculate_new(filepath):
    passed_data = calculate_passed_data(filepath)
    failed_data = calculate_failed_data(filepath)
    output = np.loadtxt(
        filepath + '/weight_proportion/proportion.output.weight.txt')
    fc3 = np.loadtxt(
        filepath + '/weight_proportion/proportion.fc3.weight.txt')
    fc2 = np.loadtxt(
        filepath + '/weight_proportion/proportion.fc2.weight.txt')
    fc1 = np.loadtxt(
        filepath + '/weight_proportion/proportion.fc1.weight.txt')
    output_passed = np.zeros(output.shape)
    output_failed = np.zeros(output.shape)
    for key, value in passed_data.items():
        x = 0
        for i in output:
            y = 0
            for j in i:
                if j * passed_data[key]>=0 :
                    output_passed[x][y] = output_passed[x][y] + j * passed_data[key]
                else:
                    output_failed[x][y] = output_failed[x][y] - j * passed_data[key]
                y = y + 1
            x = x + 1
    for key, value in failed_data.items():
        x = 0
        for i in output:
            y = 0
            for j in i:
                if j * failed_data[key]>=0 :
                    output_failed[x][y] = output_failed[x][y] + j * failed_data[key]
                else:
                    output_passed[x][y] = output_passed[x][y] - j * failed_data[key]
                y = y + 1
            x = x + 1
    fc3_passed = np.zeros(fc3.shape)
    fc3_failed = np.zeros(fc3.shape)
    x = 0
    for i in fc3:
        y = 0
        for j in i:
            for k in output_passed:
                if k[x] * j>=0:
                    fc3_passed[x][y] = fc3_passed[x][y] + k[x] * j
                else:
                    fc3_failed[x][y] = fc3_failed[x][y] - k[x] * j
            for k in output_failed:
                if k[x] * j>=0:
                    fc3_failed[x][y] = fc3_failed[x][y] + k[x] * j
                else:
                    fc3_passed[x][y] = fc3_passed[x][y] - k[x] * j
            y = y + 1
        x = x + 1
    fc2_passed = np.zeros(fc2.shape)
    fc2_failed = np.zeros(fc2.shape)
    x = 0
    for i in fc2:
        y = 0
        for j in i:
            for k in fc3_passed:
                if k[x] * j >= 0:
                    fc2_passed[x][y] = fc2_passed[x][y] + k[x] * j
                else:
                    fc2_failed[x][y] = fc2_failed[x][y] - k[x] * j
            for k in fc3_failed:
                if k[x] * j >= 0:
                    fc2_failed[x][y] = fc2_failed[x][y] + k[x] * j
                else:
                    fc2_passed[x][y] = fc2_passed[x][y] - k[x] * j
            y = y + 1
        x = x + 1
    fc1_passed = np.zeros(fc1.shape)
    fc1_failed = np.zeros(fc1.shape)
    x = 0
    for i in fc1:
        y = 0
        for j in i:
            for k in fc2_passed:
                if k[x] * j >= 0:
                    fc1_passed[x][y] = fc1_passed[x][y] + k[x] * j
                else:
                    fc1_failed[x][y] = fc1_failed[x][y] - k[x] * j
            for k in fc2_failed:
                if k[x] * j >= 0:
                    fc1_failed[x][y] = fc1_failed[x][y] + k[x] * j
                else:
                    fc1_passed[x][y] = fc1_passed[x][y] - k[x] * j
            y = y + 1
        x = x + 1
    if not os.path.exists(filepath + '/passed/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/passed/')
    if not os.path.exists(filepath + '/failed/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/failed/')
    np.savetxt(filepath +'/passed/output_passed.txt', output_passed)
    np.savetxt(filepath +'/passed/fc3_passed.txt', fc3_passed)
    np.savetxt(filepath +'/passed/fc2_passed.txt', fc2_passed)
    np.savetxt(filepath +'/passed/fc1_passed.txt', fc1_passed)

    np.savetxt(filepath +'/failed/output_failed.txt', output_failed)
    np.savetxt(filepath +'/failed/fc3_failed.txt', fc3_failed)
    np.savetxt(filepath +'/failed/fc2_failed.txt', fc2_failed)
    np.savetxt(filepath +'/failed/fc1_failed.txt', fc1_failed)
def calculate_(filepath):
    passed_data = calculate_passed_data(filepath)
    failed_data = calculate_failed_data(filepath)
    output = np.loadtxt(
        filepath +'/weight_proportion/proportion.output.weight.txt')
    fc3 = np.loadtxt(
        filepath +'/weight_proportion/proportion.fc3.weight.txt')
    fc2 = np.loadtxt(
        filepath +'/weight_proportion/proportion.fc2.weight.txt')
    fc1 = np.loadtxt(
        filepath +'/weight_proportion/proportion.fc1.weight.txt')
    output_passed = np.zeros(output.shape)
    for key, value in passed_data.items():
        x = 0
        for i in output:
            y = 0
            for j in i:
                output_passed[x][y] = output_passed[x][y] + j * passed_data[key]
                y = y + 1
            x = x + 1
    fc3_passed = np.zeros(fc3.shape)
    x = 0
    for i in fc3:
        y = 0
        for j in i:
            for k in output_passed:
                fc3_passed[x][y] = fc3_passed[x][y]+k[x]*j
            y = y + 1
        x = x + 1
    fc2_passed = np.zeros(fc2.shape)
    x = 0
    for i in fc2:
        y = 0
        for j in i:
            for k in fc3_passed:
                fc2_passed[x][y] = fc2_passed[x][y] + k[x] * j
            y = y + 1
        x = x + 1
    fc1_passed = np.zeros(fc1.shape)
    x = 0
    for i in fc1:
        y = 0
        for j in i:
            for k in fc2_passed:
                fc1_passed[x][y] = fc1_passed[x][y] + k[x] * j
            y = y + 1
        x = x + 1

    output_failed = np.zeros(output.shape)
    for key, value in failed_data.items():
        x = 0
        for i in output:
            y = 0
            for j in i:
                output_failed[x][y] = output_failed[x][y] + j * failed_data[key]
                y = y + 1
            x = x + 1
    fc3_failed = np.zeros(fc3.shape)
    x = 0
    for i in fc3:
        y = 0
        for j in i:
            for k in output_failed:
                fc3_failed[x][y] = fc3_failed[x][y]+k[x]*j
            y = y + 1
        x = x + 1
    fc2_failed = np.zeros(fc2.shape)
    x = 0
    for i in fc2:
        y = 0
        for j in i:
            for k in fc3_failed:
                fc2_failed[x][y] = fc2_failed[x][y] + k[x] * j
            y = y + 1
        x = x + 1
    fc1_failed = np.zeros(fc1.shape)
    x = 0
    for i in fc1:
        y = 0
        for j in i:
            for k in fc2_failed:
                fc1_failed[x][y] = fc1_failed[x][y] + k[x] * j
            y = y + 1
        x = x + 1
    if not os.path.exists(filepath + '/passed/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/passed/')
    if not os.path.exists(filepath + '/failed/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/failed/')
    np.savetxt(filepath +'/passed/output_passed.txt', output_passed)
    np.savetxt(filepath +'/passed/fc3_passed.txt', fc3_passed)
    np.savetxt(filepath +'/passed/fc2_passed.txt', fc2_passed)
    np.savetxt(filepath +'/passed/fc1_passed.txt', fc1_passed)

    np.savetxt(filepath +'/failed/output_failed.txt', output_failed)
    np.savetxt(filepath +'/failed/fc3_failed.txt', fc3_failed)
    np.savetxt(filepath +'/failed/fc2_failed.txt', fc2_failed)
    np.savetxt(filepath +'/failed/fc1_failed.txt', fc1_failed)
def get_calculate(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        #print(dirpath)
        print(dirpath.rsplit("\\", 1))
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            calculate_(dirpath)
def get_calculate_new(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        #print(dirpath)
        print(dirpath.rsplit("\\", 1))
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            calculate_new(dirpath)
def get_weight(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        #print(dirpath)
        print(dirpath.rsplit("\\", 1))
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            print(1)
            Wight.get_weight_(dirpath)
def get_proportion(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            #print(1)
            Wight.get_proportion_(dirpath)
def get_difference(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            #print(1)
            Get_difference.get_difference_(dirpath)
#获取所有变异体的运行情况
def get_predicted(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            processing(dirpath)
#神经元前向传递运行获取信息,同时获取前向和后向还有最终排名
def get_predicted_forward(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            processing_forward(dirpath, 33)
            #单独排序
            #Score.get_neural_rank_forward_layer(dirpath)
            Score.get_neural_rank_forward(dirpath)
def get_weight_rank(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            Score.get_weight_score_(dirpath,10)
            Score.get_weight_rank_(dirpath)
def get_neural_rank(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            Score.get_neural_score_(dirpath, 10)
            Score.get_neural_rank_(dirpath)
def get_neural_rank_new(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            Score.get_weight_score_(dirpath, 10)
            Score.get_neural_score_new(dirpath)
            Score.get_neural_rank_(dirpath)
def get_neural_rank_new2(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            Score.get_neural_score_new2(dirpath, 33)
            Score.get_neural_rank_(dirpath)
#同时获取反向 和最终结果 需要在前向排序之后
def get_neural_rank_new2_final(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath+'\\'):
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            Score.get_neural_score_new2(dirpath, 33)
            Score.get_neural_rank_(dirpath)
            Score.get_neural_rank_final(dirpath)
#统计错误用例个数
def get_failed_num(filepath):
    for dirpath, dirnames, filenames in os.walk(filepath + '\\'):
        # print(dirpath)
        #print(dirpath.rsplit("\\", 1))
        if 'mutant' in dirpath.rsplit("\\", 1)[1]:
            failed_number(dirpath)
#神经元的前向传递
def processing_forward(filepath, n):
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    net.load_state_dict(torch.load(filepath + '/model_paras.pkl'))
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
    i = 0
    for images, labels in test_loader:
        i = i + 1
        images = images.reshape(-1, 28 * 28)
        # print(images)
        output = net(images)
        z_ = net.fc1(images)
        x_ = net.fc2(z_)
        c_ = net.fc3(x_)
        v_ = net.relu(c_)
        r_ = net.output(v_)
        #print(output)
        _, predicted = torch.max(output, 1)
        #print(predicted)
        #print('****************')
        if i == 1:
            a = predicted
            #b = labels
            z = z_
            x = x_
            c = c_
            v = v_
            r = r_
        else:
            a = torch.cat([a, predicted], dim=0)
            z = torch.cat([z, z_], dim=0)
            x = torch.cat([x, x_], dim=0)
            c = torch.cat([c, c_], dim=0)
            v = torch.cat([v, v_], dim=0)
            r = torch.cat([r, r_], dim=0)
            #b = torch.cat([b, labels], dim=0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 获取预测标签和实际标签
    #np.savetxt(filepath + '/predicted/labels.txt', b)
    np.savetxt(filepath + '/predicted/predicted.txt', a)
    # np.savetxt(filepath + '/predicted/fc1_output.txt', z.detach().numpy())
    # np.savetxt(filepath + '/predicted/fc2_output.txt', x.detach().numpy())
    # np.savetxt(filepath + '/predicted/fc3_output.txt', c.detach().numpy())
    # np.savetxt(filepath + '/predicted/relu_output.txt', v.detach().numpy())
    # np.savetxt(filepath + '/predicted/output_output.txt', r.detach().numpy())
    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    flag = get_failed_data_(filepath)
    neural_fc1_passed = np.zeros(z.shape[1])
    neural_fc1_failed = np.zeros(z.shape[1])
    y = 0
    for i in z:
        if y not in flag.keys():
            for j in range(z.shape[1]):
                neural_fc1_passed[j] = neural_fc1_passed[j] + i[j]
        else:
            for j in range(z.shape[1]):
                neural_fc1_failed[j] = neural_fc1_failed[j] + i[j]
        y = y + 1
    neural_fc2_passed = np.zeros(x.shape[1])
    neural_fc2_failed = np.zeros(x.shape[1])
    y = 0
    for i in x:
        if y not in flag.keys():
            for j in range(x.shape[1]):
                neural_fc2_passed[j] = neural_fc2_passed[j] + i[j]
        else:
            for j in range(x.shape[1]):
                neural_fc2_failed[j] = neural_fc2_failed[j] + i[j]
        y = y + 1
    neural_fc3_passed = np.zeros(v.shape[1])
    neural_fc3_failed = np.zeros(v.shape[1])
    y = 0
    for i in v:
        if y not in flag.keys():
            for j in range(v.shape[1]):
                neural_fc3_passed[j] = neural_fc3_passed[j] + i[j]
        else:
            for j in range(v.shape[1]):
                neural_fc3_failed[j] = neural_fc3_failed[j] + i[j]
        y = y + 1
    neural_output_passed = np.zeros(r.shape[1])
    neural_output_failed = np.zeros(r.shape[1])
    y = 0
    for i in r:
        if y not in flag.keys():
            for j in range(r.shape[1]):
                neural_output_passed[j] = neural_output_passed[j] + i[j]
        else:
            for j in range(r.shape[1]):
                neural_output_failed[j] = neural_output_failed[j] + i[j]
        y = y + 1
    # neural_output_passed = Normalize(neural_output_passed)
    # neural_fc3_passed = Normalize(neural_fc3_passed)
    # neural_fc2_passed = Normalize(neural_fc2_passed)
    # neural_fc1_passed = Normalize(neural_fc1_passed)
    #
    # neural_output_failed = Normalize(neural_output_failed)
    # neural_fc3_failed = Normalize(neural_fc3_failed)
    # neural_fc2_failed = Normalize(neural_fc2_failed)
    # neural_fc1_failed = Normalize(neural_fc1_failed)

    neural_output_score = np.zeros(neural_output_passed.shape)
    x = 0
    for i in neural_output_passed:
        if (neural_output_failed[x] < 0) and (i < 0):
            neural_output_score[x] = (-i) ** n / (-neural_output_failed[x] - i)
        elif (neural_output_failed[x] < 0) and (i >= 0):
            neural_output_score[x] = 0
        elif (neural_output_failed[x] >= 0) and (i < 0):
            neural_output_score[x] = (neural_output_failed[x] - i) ** n / (neural_output_failed[x] - i)
        elif (neural_output_failed[x] >= 0) and (i >= 0):
            # print(failed,passed)
            if neural_output_failed[x] + i == 0:
                neural_output_score[x] = 0
            else:
                neural_output_score[x] = (neural_output_failed[x]) ** n / (neural_output_failed[x] + i)
        x = x + 1
    neural_fc3_score = np.zeros(neural_fc3_passed.shape)
    x = 0
    for i in neural_fc3_passed:
        if (neural_fc3_failed[x] < 0) and (i < 0):
            neural_fc3_score[x] = (-i) ** n / (-neural_fc3_failed[x] - i)
        elif (neural_fc3_failed[x] < 0) and (i >= 0):
            neural_fc3_score[x] = 0
        elif (neural_fc3_failed[x] >= 0) and (i < 0):
            neural_fc3_score[x] = (neural_fc3_failed[x] - i) ** n / (neural_fc3_failed[x] - i)
        elif (neural_fc3_failed[x] >= 0) and (i >= 0):
            # print(failed,passed)
            if neural_fc3_failed[x] + i == 0:
                neural_fc3_score[x] = 0
            else:
                neural_fc3_score[x] = (neural_fc3_failed[x]) ** n / (neural_fc3_failed[x] + i)
        x = x + 1
    neural_fc2_score = np.zeros(neural_fc2_passed.shape)
    x = 0
    for i in neural_fc2_passed:
        if (neural_fc2_failed[x] < 0) and (i < 0):
            neural_fc2_score[x] = (-i) ** n / (-neural_fc2_failed[x] - i)
        elif (neural_fc2_failed[x] < 0) and (i >= 0):
            neural_fc2_score[x] = 0
        elif (neural_fc2_failed[x] >= 0) and (i < 0):
            neural_fc2_score[x] = (neural_fc2_failed[x] - i) ** n / (neural_fc2_failed[x] - i)
        elif (neural_fc2_failed[x] >= 0) and (i >= 0):
            # print(failed,passed)
            if neural_fc2_failed[x] + i == 0:
                neural_fc2_score[x] = 0
            else:
                neural_fc2_score[x] = (neural_fc2_failed[x]) ** n / (neural_fc2_failed[x] + i)
        x = x + 1
    neural_fc1_score = np.zeros(neural_fc1_passed.shape)
    x = 0
    for i in neural_fc1_passed:
        if (neural_fc1_failed[x] < 0) and (i < 0):
            neural_fc1_score[x] = (-i) ** n / (-neural_fc1_failed[x] - i)
        elif (neural_fc1_failed[x] < 0) and (i >= 0):
            neural_fc1_score[x] = 0
        elif (neural_fc1_failed[x] >= 0) and (i < 0):
            neural_fc1_score[x] = (neural_fc1_failed[x] - i) ** n / (neural_fc1_failed[x] - i)
        elif (neural_fc1_failed[x] >= 0) and (i >= 0):
            # print(failed,passed)
            if neural_fc1_failed[x] + i == 0:
                neural_fc1_score[x] = 0
            else:
                neural_fc1_score[x] = (neural_fc1_failed[x]) ** n / (neural_fc1_failed[x] + i)
        x = x + 1
    if not os.path.exists(filepath + '/score_forward/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/score_forward/')
    np.savetxt(filepath + '/score_forward/neural_output_score.txt',neural_output_score)
    np.savetxt(filepath + '/score_forward/neural_fc3_score.txt',neural_fc3_score)
    np.savetxt(filepath + '/score_forward/neural_fc2_score.txt',neural_fc2_score)
    np.savetxt(filepath + '/score_forward/neural_fc1_score.txt',neural_fc1_score)
#单个变异体运行的方法
def processing(filepath):
    net = torch.nn.Sequential(
        OrderedDict([
            ("fc1", torch.nn.Linear(784, 500)),
            ("fc2", torch.nn.Linear(500, 500)),
            ("fc3", torch.nn.Linear(500, 500)),
            ("relu", torch.nn.ReLU()),
            ("output", torch.nn.Linear(500, 10)),
        ])
    )
    net.load_state_dict(torch.load(filepath + '/model_paras.pkl'))
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
    i = 0
    for images, labels in test_loader:
        i = i + 1
        images = images.reshape(-1, 28 * 28)
        # print(images)
        output = net(images)
        #print(output)
        _, predicted = torch.max(output, 1)
        #print(predicted)
        #print('****************')
        if i == 1:
            a = predicted
            b = labels
        else:
            a = torch.cat([a, predicted], dim=0)
            b = torch.cat([b, labels], dim=0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 获取预测标签和实际标签
    np.savetxt(filepath+'/predicted/labels.txt', b)
    np.savetxt(filepath+'/predicted/predicted.txt', a)
    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
#计算错误用例个数
def failed_number(filepath):
    failed_data = calculate_failed_data(filepath)
    #failed_num = {}
    num = 0
    for key, value in failed_data.items():
        num = num + value
    #failed_num[filepath.rsplit("\\", 1)[1]] = num
    print(num)
    filepath1 = filepath.rsplit("\\", 1)[0]
    if not os.path.exists(filepath1 + '\\report\\'):
        os.makedirs(filepath1 + '\\report\\')
    if not os.path.exists(filepath1+'\\report\\failed_num.npy'):  # 若不存在路径则创建
        failed_num = {}
        np.save(filepath1+'\\report\\failed_num.npy', failed_num)
    failed_num = np.load(filepath1+'\\report\\failed_num.npy').item()
    failed_num[filepath.rsplit("\\", 1)[1]] = num
    np.save(filepath1 + '\\report\\failed_num.npy', failed_num)
    # print(filepath.rsplit("\\", 1)[1])
    #print(filepath)
    #np.savetxt(filepath + '/predicted/failed_num.txt', num)
def weight_test(filepath):
    # get_weight(filepath)
    # get_proportion(filepath)
    # get_difference(filepath)
    # get_predicted(filepath)

    get_calculate(filepath)
    #弃用
    # get_calculate_new(filepath)
    get_weight_rank(filepath)
    get_failed_num(filepath)
    excel.ge_excel(filepath)
def neural_test(filepath):
    get_weight(filepath)
    get_proportion(filepath)
    get_difference(filepath)
    get_predicted(filepath)
    get_calculate(filepath)
    get_neural_rank_new2(filepath)
    get_failed_num(filepath)
    excel.ge_excel(filepath)
def neural_test_forward(filepath):
    #get_weight(filepath)
    #get_difference(filepath)
    get_predicted_forward(filepath)
    get_failed_num(filepath)
    excel.ge_excel_forward(filepath)
def neural_test_final(filepath):
    # get_weight(filepath)
    get_proportion(filepath)
    # get_difference(filepath)
    # #前向传播计算和排名
    # get_predicted_forward(filepath)
    # #反向传播计算
    get_calculate(filepath)
    # #反向传播排名和最终排名
    get_neural_rank_new2_final(filepath)
    get_failed_num(filepath)
    excel.ge_excel_final(filepath)
if __name__ == '__main__':
    # a = [[2*math.exp(-23),1.7*math.exp(-23),3*math.exp(-18)]]
    # a = Normalize(a)
    # print(a)
    #neural_test_forward('D:\\fault_localization\\test_weight')
    #neural_test_final('D:\\fault_localization\\test_weight')

    # neural_test_forward('E:\\fault_localization\\Neuron_Activation_Inverse')
    # neural_test_forward('E:\\fault_localization\\Neuron_Switch')
    # neural_test_forward('E:\\fault_localization\\weight')
    # neural_test_forward('E:\\fault_localization\\weight_shuffing')

    #get_weight('D:\\fault_localization\\1')
    #get_difference('D:\\fault_localization\\1')
    #neural_test_final('D:\\fault_localization\\weight_shuffing')
    #get_weight_rank('D:\\fault_localization\\weight')
    #get_failed_num('D:\\fault_localization\\weight')
    #excel.ge_excel('D:\\fault_localization\\weight')


    # get_weight_rank('D:\\fault_localization\\weight')
    #weight_test('D:\\fault_localization\\test_weight')
    #weight_test('D:\\fault_localization\\weight')
    #neural_test('E:\\fault_localization\\Neuron_Activation_Inverse')
    #neural_test('F:\\fault_localization\\NAI中途关机')
    neural_test('F:\\fault_localization\\test')
    #可以使用
    # neural_test('E:\\fault_localization\\Neuron_Effect_Blocking')
    # neural_test('E:\\fault_localization\\Neuron_Switch')
    # neural_test('E:\\fault_localization\\weight')
    # neural_test('E:\\fault_localization\\weight_shuffing')

    #excel.ge_excel('D:\\fault_localization\\weight_shuffing')
    #neural_test('D:\\fault_localization\\Neuron_Switch')

    # get_calculate_new('D:\\fault_localization\\test')
    # get_neural_rank('D:\\fault_localization\\test')
    # get_failed_num('D:\\fault_localization\\test')
    #excel.ge_excel('D:\\fault_localization\\test')
    #neural_test_final('D:\\fault_localization\\test')
    #excel.ge_excel('D:\\fault_localization\\weight')

    #get_difference('D:\\fault_localization\\test')
    # get_neural_rank('D:\\fault_localization\\Neuron_Effect_Blocking')
    #excel.ge_excel('D:\\fault_localization\\Neuron_Effect_Blocking')
    #get_weight('D:\\fault_localization\\Neuron_Switch_1')
    #get_difference('D:\\fault_localization\\Neuron_Activation_Inverse')

    #get_weight_rank('D:\\fault_localization\\weight_shuffing')
    #excel.get_excel('D:\\fault_localization\\weight_shuffing')

    # get_calculate_new('D:\\fault_localization\\weight_shuffing')
    # get_weight_rank('D:\\fault_localization\\weight_shuffing')
    # get_failed_num('D:\\fault_localization\\weight_shuffing')
    # excel.ge_excel('D:\\fault_localization\\weight_shuffing')