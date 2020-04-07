import numpy as np
import heapq
def get_weight_score():
    output_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/output_passed.txt')
    fc3_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc3_passed.txt')
    fc2_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc2_passed.txt')
    fc1_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc1_passed.txt')

    output_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/output_failed.txt')
    fc3_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc3_failed.txt')
    fc2_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc2_failed.txt')
    fc1_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc1_failed.txt')
    n = 3
    output_score = np.zeros(output_passed.shape)
    x = 0
    for i in output_failed:
        y = 0
        for j in i:
            if (j<0) and (output_passed[x][y] < 0):
                output_score[x][y] = (-output_passed[x][y]) ** n / (-j - output_passed[x][y])
            elif(j<0) and (output_passed[x][y] >= 0):
                output_score[x][y] = 0.01 / (-j + output_passed[x][y])#认为没有错误用例，分子使用一个较小的数
            elif(j>=0) and (output_passed[x][y] < 0):
                output_score[x][y] = (j-output_passed[x][y]) ** n / (j - output_passed[x][y])
            elif (j >= 0) and (output_passed[x][y] >= 0):
                output_score[x][y] = j ** n / (j + output_passed[x][y])
            y = y + 1
        x = x + 1

    fc3_score = np.zeros(fc3_failed.shape)
    x = 0
    for i in fc3_failed:
        y = 0
        for j in i:
            if (j<0) and (fc3_passed[x][y] < 0):
                fc3_score[x][y] = (-fc3_passed[x][y]) ** n / (-j - fc3_passed[x][y])
            elif(j<0) and (fc3_passed[x][y] >= 0):
                fc3_score[x][y] = 0.01 / (-j + fc3_passed[x][y])#认为没有错误用例，分子使用一个较小的数
            elif(j>=0) and (fc3_passed[x][y] < 0):
                fc3_score[x][y] = (j - fc3_passed[x][y]) ** n / (j - fc3_passed[x][y])
            elif (j >= 0) and (fc3_passed[x][y] >= 0):
                fc3_score[x][y] = j ** n / (j + fc3_passed[x][y])
            y = y + 1
        x = x + 1

    fc2_score = np.zeros(fc2_failed.shape)
    x = 0
    for i in fc2_failed:
        y = 0
        for j in i:
            if (j<0) and (fc2_passed[x][y] < 0):
                fc2_score[x][y] = (-fc2_passed[x][y]) ** n / (-j - fc2_passed[x][y])
            elif(j<0) and (fc2_passed[x][y] >= 0):
                fc2_score[x][y] = 0.01 / (-j + fc2_passed[x][y])#认为没有错误用例，分子使用一个较小的数
            elif(j>=0) and (fc2_passed[x][y] < 0):
                fc2_score[x][y] = (j - fc2_passed[x][y]) ** n / (j - fc2_passed[x][y])
            elif (j >= 0) and (fc2_passed[x][y] >= 0):
                fc2_score[x][y] = j ** n / (j + fc2_passed[x][y])
            y = y + 1
        x = x + 1

    fc1_score = np.zeros(fc1_failed.shape)
    x = 0
    for i in fc1_failed:
        y = 0
        for j in i:
            if (j<0) and (fc1_passed[x][y] < 0):
                fc1_score[x][y] = (-fc1_passed[x][y]) ** n / (-j - fc1_passed[x][y])
            elif(j<0) and (fc1_passed[x][y] >= 0):
                fc1_score[x][y] = 0.01 / (-j + fc1_passed[x][y])#认为没有错误用例，分子使用一个较小的数
            elif(j>=0) and (fc1_passed[x][y] < 0):
                fc1_score[x][y] = (j - fc1_passed[x][y]) ** n / (j - fc1_passed[x][y])
            elif (j >= 0) and (fc1_passed[x][y] >= 0):
                fc1_score[x][y] = j ** n / (j + fc1_passed[x][y])
            y = y + 1
        x = x + 1
    np.savetxt('mutants/score/weight/output_score.txt',output_score)
    np.savetxt('mutants/score/weight/fc3_score.txt',fc3_score)
    np.savetxt('mutants/score/weight/fc2_score.txt',fc2_score)
    np.savetxt('mutants/score/weight/fc1_score.txt',fc1_score)
def get_neural_score():
    output_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/output_passed.txt')
    fc3_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc3_passed.txt')
    fc2_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc2_passed.txt')
    fc1_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc1_passed.txt')

    output_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/output_failed.txt')
    fc3_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc3_failed.txt')
    fc2_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc2_failed.txt')
    fc1_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc1_failed.txt')
    n = 3
    neural_output_score = np.zeros(output_passed.shape[0])
    x = 0
    for i in output_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            failed = failed + j
            passed = passed + output_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_output_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_output_score[x] = 0.01 / (-failed + passed)
        elif (failed >= 0) and (passed < 0):
            neural_output_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            neural_output_score[x] = (failed) ** n / (failed + passed)
        x = x + 1

    neural_fc3_score = np.zeros(fc3_passed.shape[0])
    x = 0
    for i in fc3_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            failed = failed + j
            passed = passed + fc3_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_fc3_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_fc3_score[x] = 0.01 / (-failed + passed)
        elif (failed >= 0) and (passed < 0):
            neural_fc3_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            neural_fc3_score[x] = (failed) ** n / (failed + passed)
        x = x + 1

    neural_fc2_score = np.zeros(fc2_passed.shape[0])
    x = 0
    for i in fc2_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            failed = failed + j
            passed = passed + fc2_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_fc2_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_fc2_score[x] = 0.01 / (-failed + passed)
        elif (failed >= 0) and (passed < 0):
            neural_fc2_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            neural_fc2_score[x] = (failed) ** n / (failed + passed)
        x = x + 1

    neural_fc1_score = np.zeros(fc1_passed.shape[0])
    x = 0
    for i in fc1_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            failed = failed + j
            passed = passed + fc1_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_fc1_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_fc1_score[x] = 0.01 / (-failed + passed)
        elif (failed >= 0) and (passed < 0):
            neural_fc1_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            neural_fc1_score[x] = (failed) ** n / (failed + passed)
        x = x + 1
    np.savetxt('mutants/score/neural/neural_output_score.txt', neural_output_score)
    np.savetxt('mutants/score/neural/neural_fc3_score.txt', neural_fc3_score)
    np.savetxt('mutants/score/neural/neural_fc2_score.txt', neural_fc2_score)
    np.savetxt('mutants/score/neural/neural_fc1_score.txt', neural_fc1_score)
def get_weight_rank():
    output_score = np.loadtxt('mutants/score/weight/output_score.txt')
    fc3_score = np.loadtxt('mutants/score/weight/fc3_score.txt')
    fc2_score = np.loadtxt('mutants/score/weight/fc2_score.txt')
    fc1_score = np.loadtxt('mutants/score/weight/fc1_score.txt')
    top_k = 100
    dict={}
    for z in range(top_k):
        m = 0
        n = 0
        middle = 0
        x = 0
        for i in output_score:
            y = 0
            for j in i:
                if j > middle:
                    if j not in dict.keys():
                        middle = j
                        m = x
                        n = y
                y = y + 1
            x = x + 1
        dict[middle] = [m,n]
    print(dict)
    np.save('mutants/score/output_top_k.npy', dict)
    dict = {}
    for z in range(top_k):
        m = 0
        n = 0
        middle = 0
        x = 0
        for i in fc3_score:
            y = 0
            for j in i:
                if j > middle:
                    if j not in dict.keys():
                        middle = j
                        m = x
                        n = y
                y = y + 1
            x = x + 1
        dict[middle] = [m, n]
    print(dict)
    np.save('mutants/score/fc3_top_k.npy', dict)
    dict = {}
    for z in range(top_k):
        m = 0
        n = 0
        middle = 0
        x = 0
        for i in fc2_score:
            y = 0
            for j in i:
                if j > middle:
                    if j not in dict.keys():
                        middle = j
                        m = x
                        n = y
                y = y + 1
            x = x + 1
        dict[middle] = [m, n]
    print(dict)
    np.save('mutants/score/fc2_top_k.npy', dict)
    dict = {}
    for z in range(top_k):
        m = 0
        n = 0
        middle = 0
        x = 0
        for i in fc1_score:
            y = 0
            for j in i:
                if j > middle:
                    if j not in dict.keys():
                        middle = j
                        m = x
                        n = y
                y = y + 1
            x = x + 1
        dict[middle] = [m, n]
    print(dict)
    np.save('mutants/score/fc1_top_k.npy', dict)
def get_neural_rank():
    neural_output_score = np.loadtxt('mutants/score/neural/neural_output_score.txt')
    neural_fc3_score = np.loadtxt('mutants/score/neural/neural_fc3_score.txt')
    neural_fc2_score = np.loadtxt('mutants/score/neural/neural_fc2_score.txt')
    neural_fc1_score = np.loadtxt('mutants/score/neural/neural_fc1_score.txt')
    top_k = 100
    dict = {}
    for z in range(top_k):
        m = 0
        middle = 0
        x = 0
        for i in neural_output_score:
            if i >= middle:
                if i not in dict.keys():
                    middle = i
                    m = x
            x = x + 1
        dict[middle] = m
    print(dict)
    np.save('mutants/score/neural_output_top_k.npy', dict)
    dict = {}
    for z in range(top_k):
        m = 0
        middle = 0
        x = 0
        for i in neural_fc3_score:
            if i > middle:
                if i not in dict.keys():
                    middle = i
                    m = x
            x = x + 1
        dict[middle] = m
    print(dict)
    np.save('mutants/score/neural_fc3_top_k.npy', dict)
    dict = {}
    for z in range(top_k):
        m = 0
        middle = 0
        x = 0
        for i in neural_fc2_score:
            if i > middle:
                if i not in dict.keys():
                    middle = i
                    m = x
            x = x + 1
        dict[middle] = m
    print(dict)
    np.save('mutants/score/neural_fc2_top_k.npy', dict)
    dict = {}
    for z in range(top_k):
        m = 0
        middle = 0
        x = 0
        for i in neural_fc1_score:
            if i > middle:
                if i not in dict.keys():
                    middle = i
                    m = x
            x = x + 1
        dict[middle] = m
    print(dict)
    np.save('mutants/score/neural_fc1_top_k.npy', dict)
def judge_weight_rank():
    differ = np.load('mutants/predicted/differ.npy').item()
    output_top_k = np.load('mutants/score/output_top_k.npy').item()
    fc3_top_k = np.load('mutants/score/fc3_top_k.npy').item()
    fc2_top_k = np.load('mutants/score/fc2_top_k.npy').item()
    fc1_top_k = np.load('mutants/score/fc1_top_k.npy').item()
    dict = {}
    i = 0
    for j in differ['output.weight.txt']:
        x = 0
        for k, v in output_top_k.items():
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('output层')
    for k, v in dict.items():
        print(k, ': 第', v, '命中')
    dict = {}
    i = 0
    for j in differ['fc3.weight.txt']:
        x = 0
        for k,v in fc3_top_k.items():
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('fc3层')
    for k, v in dict.items():
        print(k, ': 第', v, '命中')
    dict = {}
    i = 0
    for j in differ['fc2.weight.txt']:
        x = 0
        for k, v in fc2_top_k.items():
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('fc2层')
    for k, v in dict.items():
        print(k, ': 第', v, '命中')
    dict = {}
    i = 0
    for j in differ['fc1.weight.txt']:
        x = 0
        for k, v in fc1_top_k.items():
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('fc1层')
    for k,v in dict.items():
        print(k,': 第',v,'命中')
def judge_neural_rank():
    neural_differ = np.load('mutants/predicted/neural_differ.npy').item()
    neural_output_top_k = np.load('mutants/score/neural_output_top_k.npy').item()
    neural_fc3_top_k = np.load('mutants/score/neural_fc3_top_k.npy').item()
    neural_fc2_top_k = np.load('mutants/score/neural_fc2_top_k.npy').item()
    neural_fc1_top_k = np.load('mutants/score/neural_fc1_top_k.npy').item()
    dict = {}
    i = 0
    for j in neural_differ['output.weight.txt']:
        x = 0
        for k, v in neural_output_top_k.items():
            print(k,v)
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('output层')
    for k, v in dict.items():
        print(k, ': 第', v, '命中')
    dict = {}
    i = 0
    for j in neural_differ['fc3.weight.txt']:
        x = 0
        for k, v in neural_fc3_top_k.items():
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('fc3层')
    for k, v in dict.items():
        print(k, ': 第', v, '命中')
    dict = {}
    i = 0
    for j in neural_differ['fc2.weight.txt']:
        x = 0
        for k, v in neural_fc2_top_k.items():
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('fc2层')
    for k, v in dict.items():
        print(k, ': 第', v, '命中')
    dict = {}
    i = 0
    for j in neural_differ['fc1.weight.txt']:
        x = 0
        for k, v in neural_fc1_top_k.items():
            x = x + 1
            if v == j:
                i = i + 1
                dict[i] = x
    print('fc1层')
    for k, v in dict.items():
        print(k, ': 第', v, '命中')

get_neural_score()
get_neural_rank()
judge_neural_rank()

get_weight_score()
get_weight_rank()
judge_weight_rank()

