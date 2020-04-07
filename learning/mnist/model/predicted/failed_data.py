import numpy as np
def get_failed_data():
    a = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/predicted/labels.txt')
    b = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/predicted/predicted.txt')
    flag = {}
    i = 0
    while i < a.size:
        if(a[i]!=b[i]):
            flag[i] = b[i]
        i=i+1
    #不一致的结果的位置为key，值为错误的分类标签
    return flag
def calculate_failed_data():
    failed_data = {}
    flag = get_failed_data()
    for key, value in flag.items():
        if value not in failed_data:
            failed_data[value] = 1
        else:
            failed_data[value] = failed_data[value] + 1
    #各个分类结果中，错误的个数
    return failed_data

def calculate_passed_data():
    a = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/predicted/predicted.txt')
    j = 0
    failed_data = get_failed_data()
    passed_data = {}
    for i in a:
        j = j + 1
        if j not in failed_data:
            if i not in passed_data:
                passed_data[i] = 1
            else:
                passed_data[i] = passed_data[i]+1
    #各个分类结果中，正确的个数
    return passed_data
def calculate_():
    passed_data = calculate_passed_data()
    failed_data = calculate_failed_data()
    output = np.loadtxt(
        'C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/weight_proportion/proportion.output.weight.txt')
    fc3 = np.loadtxt(
        'C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/weight_proportion/proportion.fc3.weight.txt')
    fc2 = np.loadtxt(
        'C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/weight_proportion/proportion.fc2.weight.txt')
    fc1 = np.loadtxt(
        'C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/weight_proportion/proportion.fc1.weight.txt')
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
    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/output_passed.txt', output_passed)
    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc3_passed.txt', fc3_passed)
    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc2_passed.txt', fc2_passed)
    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc1_passed.txt', fc1_passed)

    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/output_failed.txt', output_failed)
    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc3_failed.txt', fc3_failed)
    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc2_passed.txt', fc2_failed)
    np.savetxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc1_passed.txt', fc1_failed)

calculate_()
