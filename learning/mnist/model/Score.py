import numpy as np
from sklearn import preprocessing
import os
import heapq
def Normalize(data):
 shape = data.shape
 data = data.reshape(-1, 1)
 m = np.mean(data)
 mx = data.max()
 mn = data.min()
 data = preprocessing.minmax_scale(data, feature_range=(-1,1))
 return data.reshape(shape)
def Normalize(data,x,y):
 shape = data.shape
 data = data.reshape(-1, 1)
 m = np.mean(data)
 mx = data.max()
 mn = data.min()
 data = preprocessing.minmax_scale(data, feature_range=(x,y))
 return data.reshape(shape)
def get_weight_score():
    output_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/output_passed.txt')
    fc3_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc3_passed.txt')
    fc2_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc2_passed.txt')
    fc1_passed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/passed/fc1_passed.txt')

    output_passed = Normalize(output_passed)
    fc3_passed = Normalize(fc3_passed)
    fc2_passed = Normalize(fc2_passed)
    fc1_passed = Normalize(fc1_passed)
    # output_passed = preprocessing.minmax_scale(output_passed, feature_range=(-1,1))

    output_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/output_failed.txt')
    fc3_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc3_failed.txt')
    fc2_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc2_failed.txt')
    fc1_failed = np.loadtxt('C:/Users/LZ/PycharmProjects/PyTorch/learning/mnist/model/mutants/failed/fc1_failed.txt')

    output_failed = Normalize(output_failed)
    fc3_failed = Normalize(fc3_failed)
    fc2_failed = Normalize(fc2_failed)
    fc1_failed = Normalize(fc1_failed)
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
#n为d*方法的参数
def get_weight_score_(filepath, n):
    output_passed = np.loadtxt(filepath + '/passed/output_passed.txt')
    fc3_passed = np.loadtxt(filepath + '/passed/fc3_passed.txt')
    fc2_passed = np.loadtxt(filepath + '/passed/fc2_passed.txt')
    fc1_passed = np.loadtxt(filepath + '/passed/fc1_passed.txt')

    output_passed = Normalize(output_passed)
    fc3_passed = Normalize(fc3_passed)
    fc2_passed = Normalize(fc2_passed)
    fc1_passed = Normalize(fc1_passed)

    # output_passed = output_passed * 10 / (500 * 500 * 500)
    # fc3_passed = fc3_passed * 500 / (10 * 500 * 500)
    # fc2_passed = fc2_passed * 500 / (10 * 500 * 500)
    # fc1_passed = fc1_passed * 500 / (10 * 500 * 500)

    # output_passed = output_passed * 10 * 10
    # fc3_passed = fc3_passed * 500 * 500
    # fc2_passed = fc2_passed * 500 * 500
    # fc1_passed = fc1_passed * 500 * 500
    # output_passed = preprocessing.minmax_scale(output_passed, feature_range=(-1,1))

    output_failed = np.loadtxt(filepath + '/failed/output_failed.txt')
    fc3_failed = np.loadtxt(filepath + '/failed/fc3_failed.txt')
    fc2_failed = np.loadtxt(filepath + '/failed/fc2_failed.txt')
    fc1_failed = np.loadtxt(filepath + '/failed/fc1_failed.txt')

    output_failed = Normalize(output_failed)
    fc3_failed = Normalize(fc3_failed)
    fc2_failed = Normalize(fc2_failed)
    fc1_failed = Normalize(fc1_failed)

    # output_failed = output_failed * 10 / (500 * 500 * 500)
    # fc3_failed = fc3_failed * 500 / (10 * 500 * 500)
    # fc2_failed = fc2_failed * 500 / (10 * 500 * 500)
    # fc1_failed = fc1_failed * 500 / (10 * 500 * 500)

    # output_failed = output_failed * 10 * 10
    # fc3_failed = fc3_failed * 500 * 500
    # fc2_failed = fc2_failed * 500 * 500
    # fc1_failed = fc1_failed * 500 * 500
    # n = 3
    output_score = np.zeros(output_passed.shape)
    x = 0
    for i in output_failed:
        y = 0
        for j in i:
            if (j<0) and (output_passed[x][y] < 0):
                output_score[x][y] = (-output_passed[x][y]) ** n / (-j - output_passed[x][y])
            elif(j<0) and (output_passed[x][y] >= 0):
                output_score[x][y] = 0#认为没有错误用例，分子使用一个较小的数
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
                fc3_score[x][y] = 0#认为没有错误用例，分子使用一个较小的数
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
                fc2_score[x][y] = 0#认为没有错误用例，分子使用一个较小的数
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
                fc1_score[x][y] = 0#认为没有错误用例，分子使用一个较小的数
            elif(j>=0) and (fc1_passed[x][y] < 0):
                fc1_score[x][y] = (j - fc1_passed[x][y]) ** n / (j - fc1_passed[x][y])
            elif (j >= 0) and (fc1_passed[x][y] >= 0):
                fc1_score[x][y] = j ** n / (j + fc1_passed[x][y])
            y = y + 1
        x = x + 1
    if not os.path.exists(filepath + '/score/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/score/')
    np.savetxt(filepath + '/score/output_score.txt',output_score)
    np.savetxt(filepath + '/score/fc3_score.txt',fc3_score)
    np.savetxt(filepath + '/score/fc2_score.txt',fc2_score)
    np.savetxt(filepath + '/score/fc1_score.txt',fc1_score)
def get_neural_score_new(filepath):
    output_score = np.loadtxt(filepath + '/score/output_score.txt')
    fc3_score=np.loadtxt(filepath + '/score/fc3_score.txt')
    fc2_score=np.loadtxt(filepath + '/score/fc2_score.txt')
    fc1_score=np.loadtxt(filepath + '/score/fc1_score.txt')
    neural_output_score = np.zeros(output_score.shape[0])
    x = 0
    for i in output_score:
        for j in i:
            neural_output_score[x] = neural_output_score[x] + j
        x = x + 1
    neural_fc3_score = np.zeros(fc3_score.shape[0])
    x = 0
    for i in fc3_score:
        for j in i:
            neural_fc3_score[x] = neural_fc3_score[x] + j
        x = x + 1
    neural_fc2_score = np.zeros(fc2_score.shape[0])
    x = 0
    for i in fc2_score:
        for j in i:
            neural_fc2_score[x] = neural_fc2_score[x] + j
        x = x + 1
    neural_fc1_score = np.zeros(fc1_score.shape[0])
    x = 0
    for i in fc1_score:
        for j in i:
            neural_fc1_score[x] = neural_fc1_score[x] + j
        x = x + 1
    if not os.path.exists(filepath + '/score/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/score/')
    np.savetxt(filepath + '/score/neural_output_score.txt',neural_output_score)
    np.savetxt(filepath + '/score/neural_fc3_score.txt',neural_fc3_score)
    np.savetxt(filepath + '/score/neural_fc2_score.txt',neural_fc2_score)
    np.savetxt(filepath + '/score/neural_fc1_score.txt',neural_fc1_score)
def get_neural_score_new2(filepath,n):
    output_passed = np.loadtxt(filepath + '/passed/output_passed.txt')
    fc3_passed = np.loadtxt(filepath + '/passed/fc3_passed.txt')
    fc2_passed = np.loadtxt(filepath + '/passed/fc2_passed.txt')
    fc1_passed = np.loadtxt(filepath + '/passed/fc1_passed.txt')
    #12.9 在计算前不归一化
    # output_passed = Normalize(output_passed,-1,1)
    # fc3_passed = Normalize(fc3_passed,-1,1)
    # fc2_passed = Normalize(fc2_passed,-1,1)
    # fc1_passed = Normalize(fc1_passed,-1,1)


    output_failed = np.loadtxt(filepath + '/failed/output_failed.txt')
    fc3_failed = np.loadtxt(filepath + '/failed/fc3_failed.txt')
    fc2_failed = np.loadtxt(filepath + '/failed/fc2_failed.txt')
    fc1_failed = np.loadtxt(filepath + '/failed/fc1_failed.txt')
    #12.9 在计算前不归一化
    # output_failed = Normalize(output_failed,-1,1)
    # fc3_failed = Normalize(fc3_failed,-1,1)
    # fc2_failed = Normalize(fc2_failed,-1,1)
    # fc1_failed = Normalize(fc1_failed,-1,1)


    neural_output_score = np.zeros(output_passed.shape[0])
    x = 0
    for i in output_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            if j >=0 :
                failed = failed + j
            else:
                passed = passed - j
            if output_passed[x][y] >= 0 :
                passed = passed + output_passed[x][y]
            else:
                failed = failed - output_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_output_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_output_score[x] = 0
            #neural_output_score[x] = 0.01 / (-failed + passed)
        elif (failed >= 0) and (passed < 0):
            neural_output_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            if failed+passed ==0:
                neural_output_score[x] = 0
            else:
                neural_output_score[x] = (failed) ** n / (failed + passed)
        x = x + 1

    neural_fc3_score = np.zeros(fc3_passed.shape[0])
    x = 0
    for i in fc3_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            if j >= 0:
                failed = failed + j
            else:
                passed = passed - j
            if fc3_passed[x][y] >= 0:
                passed = passed + fc3_passed[x][y]
            else:
                failed = failed - fc3_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_fc3_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_fc3_score[x] = 0
        elif (failed >= 0) and (passed < 0):
            neural_fc3_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            #print(failed+passed)
            if failed + passed == 0:
                neural_fc3_score[x] = 0
            else:
                neural_fc3_score[x] = (failed) ** n / (failed + passed)
        x = x + 1

    neural_fc2_score = np.zeros(fc2_passed.shape[0])
    x = 0
    for i in fc2_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            if j >= 0:
                failed = failed + j
            else:
                passed = passed - j
            if fc2_passed[x][y] >= 0:
                passed = passed + fc2_passed[x][y]
            else:
                failed = failed - fc2_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_fc2_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_fc2_score[x] = 0
        elif (failed >= 0) and (passed < 0):
            neural_fc2_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            # print('aaaaaaaaaaaaaaaaaaaaa',(failed) ** n / (failed + passed))
            if failed + passed == 0:
                neural_fc2_score[x] = 0
            else:
                neural_fc2_score[x] = (failed) ** n / (failed + passed)
        x = x + 1

    neural_fc1_score = np.zeros(fc1_passed.shape[0])
    x = 0
    for i in fc1_failed:
        y = 0
        passed = 0
        failed = 0
        for j in i:
            if j >= 0:
                failed = failed + j
            else:
                passed = passed - j
            if fc1_passed[x][y] >= 0:
                passed = passed + fc1_passed[x][y]
            else:
                failed = failed - fc1_passed[x][y]
            y = y + 1
        if (failed < 0) and (passed < 0):
            neural_fc1_score[x] = (-passed) ** n / (-failed - passed)
        elif (failed < 0) and (passed >= 0):
            neural_fc1_score[x] = 0
        elif (failed >= 0) and (passed < 0):
            neural_fc1_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            #print(failed,passed)
            if failed+passed == 0:
                neural_fc1_score[x] = 0
            else:
                neural_fc1_score[x] = (failed) ** n / (failed + passed)
        x = x + 1
    if not os.path.exists(filepath + '/score/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/score/')
    np.savetxt(filepath + '/score/neural_output_score.txt',neural_output_score)
    np.savetxt(filepath + '/score/neural_fc3_score.txt',neural_fc3_score)
    np.savetxt(filepath + '/score/neural_fc2_score.txt',neural_fc2_score)
    np.savetxt(filepath + '/score/neural_fc1_score.txt',neural_fc1_score)
    # np.savetxt('mutants/score/neural/neural_output_score.txt', neural_output_score)
    # np.savetxt('mutants/score/neural/neural_fc3_score.txt', neural_fc3_score)
    # np.savetxt('mutants/score/neural/neural_fc2_score.txt', neural_fc2_score)
    # np.savetxt('mutants/score/neural/neural_fc1_score.txt', neural_fc1_score)
def get_neural_score_(filepath,n):
    output_passed = np.loadtxt(filepath + '/passed/output_passed.txt')
    fc3_passed = np.loadtxt(filepath + '/passed/fc3_passed.txt')
    fc2_passed = np.loadtxt(filepath + '/passed/fc2_passed.txt')
    fc1_passed = np.loadtxt(filepath + '/passed/fc1_passed.txt')

    output_passed = Normalize(output_passed)
    fc3_passed = Normalize(fc3_passed)
    fc2_passed = Normalize(fc2_passed)
    fc1_passed = Normalize(fc1_passed)

    # output_passed = output_passed*10*10
    # fc3_passed = fc3_passed*500*500
    # fc2_passed = fc2_passed*500*500
    # fc1_passed = fc1_passed*500*500

    # output_passed = output_passed * 10/(500*500*500)
    # fc3_passed = fc3_passed * 500 /(10*500*500)
    # fc2_passed = fc2_passed * 500 /(10*500*500)
    # fc1_passed = fc1_passed * 500 /(10*500*500)
    # output_passed = preprocessing.minmax_scale(output_passed, feature_range=(-1,1))

    output_failed = np.loadtxt(filepath + '/failed/output_failed.txt')
    fc3_failed = np.loadtxt(filepath + '/failed/fc3_failed.txt')
    fc2_failed = np.loadtxt(filepath + '/failed/fc2_failed.txt')
    fc1_failed = np.loadtxt(filepath + '/failed/fc1_failed.txt')

    output_failed = Normalize(output_failed)
    fc3_failed = Normalize(fc3_failed)
    fc2_failed = Normalize(fc2_failed)
    fc1_failed = Normalize(fc1_failed)

    # output_failed = output_failed*10*10
    # fc3_failed = fc3_failed*500*500
    # fc2_failed = fc2_failed*500*500
    # fc1_failed = fc1_failed*500*500

    # output_failed = output_failed * 10 /(500*500*500)
    # fc3_failed = fc3_failed * 500 /(10*500*500)
    # fc2_failed = fc2_failed * 500 /(10*500*500)
    # fc1_failed = fc1_failed * 500 /(10*500*500)
    #n = 3
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
            neural_output_score[x] = 0
            #neural_output_score[x] = 0.01 / (-failed + passed)
        elif (failed >= 0) and (passed < 0):
            neural_output_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            if failed+passed ==0:
                neural_output_score[x] = 0
            else:
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
            neural_fc3_score[x] = 0
        elif (failed >= 0) and (passed < 0):
            neural_fc3_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            #print(failed+passed)
            if failed + passed == 0:
                neural_fc3_score[x] = 0
            else:
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
            neural_fc2_score[x] = 0
        elif (failed >= 0) and (passed < 0):
            neural_fc2_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            # print('aaaaaaaaaaaaaaaaaaaaa',(failed) ** n / (failed + passed))
            if failed + passed == 0:
                neural_fc2_score[x] = 0
            else:
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
            neural_fc1_score[x] = 0
        elif (failed >= 0) and (passed < 0):
            neural_fc1_score[x] = (failed-passed) ** n / (failed - passed)
        elif (failed >= 0) and (passed >= 0):
            #print(failed,passed)
            if failed+passed == 0:
                neural_fc1_score[x] = 0
            else:
                neural_fc1_score[x] = (failed) ** n / (failed + passed)
        x = x + 1
    if not os.path.exists(filepath + '/score/'):  # 若不存在路径则创建
        os.makedirs(filepath + '/score/')
    np.savetxt(filepath + '/score/neural_output_score.txt',neural_output_score)
    np.savetxt(filepath + '/score/neural_fc3_score.txt',neural_fc3_score)
    np.savetxt(filepath + '/score/neural_fc2_score.txt',neural_fc2_score)
    np.savetxt(filepath + '/score/neural_fc1_score.txt',neural_fc1_score)
    # np.savetxt('mutants/score/neural/neural_output_score.txt', neural_output_score)
    # np.savetxt('mutants/score/neural/neural_fc3_score.txt', neural_fc3_score)
    # np.savetxt('mutants/score/neural/neural_fc2_score.txt', neural_fc2_score)
    # np.savetxt('mutants/score/neural/neural_fc1_score.txt', neural_fc1_score)
def get_weight_rank_(filepath):
    output_score = np.loadtxt(filepath +'/score/output_score.txt')
    fc3_score = np.loadtxt(filepath +'/score/fc3_score.txt')
    fc2_score = np.loadtxt(filepath +'/score/fc2_score.txt')
    fc1_score = np.loadtxt(filepath +'/score/fc1_score.txt')
    differ = np.load(filepath +'/predicted/differ.npy').item()
    target_score = -100
    #不会进入的分支
    if len(differ)==1:
        for k,v in differ.items():
            print(k,v)
            if 'output' in k:
                if v:
                    target_score = output_score[v[0][0],v[0][1]]
            elif 'fc3' in k:
                if v:
                    target_score = fc3_score[v[0][0], v[0][1]]
            elif 'fc2' in k:
                if v:
                    target_score = fc2_score[v[0][0],v[0][1]]
            elif 'fc1' in k:
                if v:
                    target_score = fc1_score[v[0][0],v[0][1]]
        print(target_score)
    else:
        for k,v in differ.items():
            print(k,v)
            if 'output' in k:
                if v:
                    if target_score <output_score[v[0][0],v[0][1]]:
                        target_score = output_score[v[0][0],v[0][1]]
            elif 'fc3' in k:
                if v:
                    if target_score < fc3_score[v[0][0], v[0][1]]:
                        target_score = fc3_score[v[0][0], v[0][1]]
            elif 'fc2' in k:
                if v:
                    if target_score < fc2_score[v[0][0], v[0][1]]:
                        target_score = fc2_score[v[0][0],v[0][1]]
            elif 'fc1' in k:
                if v:
                    if target_score < fc1_score[v[0][0], v[0][1]]:
                        target_score = fc1_score[v[0][0],v[0][1]]
        print(target_score)
    output_score_ = output_score.reshape(-1,1)
    fc3_score_ = fc3_score.reshape(-1,1)
    fc2_score_ = fc2_score.reshape(-1,1)
    fc1_score_ = fc1_score.reshape(-1,1)
    result = np.vstack((output_score_,fc3_score_))
    result = np.vstack((result,fc2_score_))
    result = np.vstack((result,fc1_score_))
    target = 1
    flag = False
    for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
        if result[i][0] > target_score:
            target = target + 1
    filepath1 = filepath.rsplit("/", 1)[0]
    if not os.path.exists(filepath1 + '/report/'):
        os.makedirs(filepath1 + '/report/')
    if not os.path.exists(filepath1+'/report/report.npy'):  # 若不存在路径则创建
        report = {}
        np.save(filepath1+'/report/report.npy', report)
    report = np.load(filepath1+'/report/report.npy').item()
    report[filepath.rsplit("/", 1)[1]] = [target,target/len(result)]
    #print(report)
    np.save(filepath1+'/report/report.npy', report)
    #print(target)
def get_neural_rank_(filepath):
    output_score = np.loadtxt(filepath +'/score/neural_output_score.txt')
    fc3_score = np.loadtxt(filepath +'/score/neural_fc3_score.txt')
    fc2_score = np.loadtxt(filepath +'/score/neural_fc2_score.txt')
    fc1_score = np.loadtxt(filepath +'/score/neural_fc1_score.txt')
    differ = np.load(filepath +'/predicted/neural_differ.npy').item()
    #12.9 在排序前进行归一化
    output_score = Normalize(output_score,0,1)
    fc3_score = Normalize(fc3_score,0,1)
    fc2_score = Normalize(fc2_score,0,1)
    fc1_score = Normalize(fc1_score,0,1)
    target_score = -100
    # for k,v in differ.items():
    #     print(k,v)
    #     if 'output' in k:
    #         print(len(v))
    #         if v:
    #             target_score = output_score[v[0]]
    #     elif 'fc3' in k:
    #         if v:
    #             target_score = fc3_score[v[0]]
    #     elif 'fc2' in k:
    #         if v:
    #             target_score = fc2_score[v[0]]
    #     elif 'fc1' in k:
    #         if v:
    #             target_score = fc1_score[v[0]]
    #     print(target_score)
    for k,v in differ.items():
        #print(k,v)
        if 'output' in k:
            if v:
                for i in v:
                    if target_score < output_score[i]:
                        target_score = output_score[i]
        elif 'fc3' in k:
            if v:
                for i in v:
                    if target_score < fc3_score[i]:
                        target_score = fc3_score[i]
        elif 'fc2' in k:
            if v:
                for i in v:
                    if target_score < fc2_score[i]:
                        target_score = fc2_score[i]
        elif 'fc1' in k:
            if v:
                for i in v:
                    print(fc1_score[i])
                    if target_score < fc1_score[i]:
                        target_score = fc1_score[i]
    #print(target_score)
    output_score_ = output_score.reshape(-1,1)
    fc3_score_ = fc3_score.reshape(-1,1)
    fc2_score_ = fc2_score.reshape(-1,1)
    fc1_score_ = fc1_score.reshape(-1,1)
    result = np.vstack((output_score_,fc3_score_))
    result = np.vstack((result,fc2_score_))
    result = np.vstack((result,fc1_score_))
    target = 1
    # flag = False
    for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
        if result[i][0] > target_score:
            target = target + 1
    filepath1 = filepath.rsplit("/", 1)[0]
    if not os.path.exists(filepath1 + '/report/'):
        os.makedirs(filepath1 + '/report/')
    if not os.path.exists(filepath1+'/report/report.npy'):  # 若不存在路径则创建
        report = {}
        np.save(filepath1+'/report/report.npy', report)
    report = np.load(filepath1+'/report/report.npy').item()
    report[filepath.rsplit("/", 1)[1]] = [target,target/len(result)]
    print(report)
    np.save(filepath1+'/report/report.npy', report)
    print(target)
def get_neural_rank_forward(filepath):
    output_score = np.loadtxt(filepath +'/score_forward/neural_output_score.txt')
    fc3_score = np.loadtxt(filepath +'/score_forward/neural_fc3_score.txt')
    fc2_score= np.loadtxt(filepath +'/score_forward/neural_fc2_score.txt')
    fc1_score = np.loadtxt(filepath +'/score_forward/neural_fc1_score.txt')
    differ = np.load(filepath +'/predicted/neural_differ.npy').item()

    output_score = Normalize(output_score,0,1)
    fc3_score = Normalize(fc3_score,0,1)
    fc2_score = Normalize(fc2_score,0,1)
    fc1_score = Normalize(fc1_score,0,1)

    target_score = -100
    # for k,v in differ.items():
    #     print(k,v)
    #     if 'output' in k:
    #         print(len(v))
    #         if v:
    #             target_score = output_score[v[0]]
    #     elif 'fc3' in k:
    #         if v:
    #             target_score = fc3_score[v[0]]
    #     elif 'fc2' in k:
    #         if v:
    #             target_score = fc2_score[v[0]]
    #     elif 'fc1' in k:
    #         if v:
    #             target_score = fc1_score[v[0]]
    #     print(target_score)
    for k,v in differ.items():
        print(k,v)
        if 'output' in k:
            if v:
                for i in v:
                    if target_score < output_score[i]:
                        target_score = output_score[i]
        elif 'fc3' in k:
            if v:
                for i in v:
                    if target_score < fc3_score[i]:
                        target_score = fc3_score[i]
        elif 'fc2' in k:
            if v:
                for i in v:
                    if target_score < fc2_score[i]:
                        target_score = fc2_score[i]
        elif 'fc1' in k:
            if v:
                for i in v:
                    print(fc1_score[i])
                    if target_score < fc1_score[i]:
                        target_score = fc1_score[i]
    print(target_score)
    output_score_ = output_score.reshape(-1,1)
    fc3_score_ = fc3_score.reshape(-1,1)
    fc2_score_ = fc2_score.reshape(-1,1)
    fc1_score_ = fc1_score.reshape(-1,1)
    # #统计占比
    # list = []
    # dict = {}
    # dict[1] = 0
    # dict[2] = 0
    # dict[3] = 0
    # dict[4] = 0
    # for j in range(100):
    #     biggest = -100
    #     for i in output_score:
    #         if i > biggest and i not in list:
    #             biggest = i
    #             flag = 4
    #     for i in fc3_score:
    #         if i > biggest and i not in list:
    #             biggest = i
    #             flag = 3
    #     for i in fc2_score:
    #         if i > biggest and i not in list:
    #             biggest = i
    #             flag = 2
    #     for i in fc1_score:
    #         if i > biggest and i not in list:
    #             biggest = i
    #             flag = 1
    #     list.append(biggest)
    #     dict[flag] = dict[flag] + 1
    # print(list)
    # print(dict)

    result = np.vstack((output_score_,fc3_score_))
    result = np.vstack((result,fc2_score_))
    result = np.vstack((result,fc1_score_))
    target = 1
    # flag = False
    for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
        if result[i][0] > target_score:
            target = target + 1
    filepath1 = filepath.rsplit("/", 1)[0]
    if not os.path.exists(filepath1 + '/report_forward/'):
        os.makedirs(filepath1 + '/report_forward/')
    if not os.path.exists(filepath1+'/report_forward/report.npy'):  # 若不存在路径则创建
        report = {}
        np.save(filepath1+'/report_forward/report.npy', report)
    report = np.load(filepath1+'/report_forward/report.npy').item()
    report[filepath.rsplit("/", 1)[1]] = [target,target/len(result)]
    print(report)
    np.save(filepath1+'/report_forward/report.npy', report)
    print(target)
def get_neural_rank_forward_layer(filepath):
    output_score = np.loadtxt(filepath +'/score_forward/neural_output_score.txt')
    fc3_score = np.loadtxt(filepath +'/score_forward/neural_fc3_score.txt')
    fc2_score= np.loadtxt(filepath +'/score_forward/neural_fc2_score.txt')
    fc1_score = np.loadtxt(filepath +'/score_forward/neural_fc1_score.txt')
    differ = np.load(filepath +'/predicted/neural_differ.npy').item()

    # output_score = Normalize(output_score,0,1)
    # fc3_score = Normalize(fc3_score,0,1)
    # fc2_score = Normalize(fc2_score,0,1)
    # fc1_score = Normalize(fc1_score,0,1)

    target_score = -100
    # for k,v in differ.items():
    #     print(k,v)
    #     if 'output' in k:
    #         print(len(v))
    #         if v:
    #             target_score = output_score[v[0]]
    #     elif 'fc3' in k:
    #         if v:
    #             target_score = fc3_score[v[0]]
    #     elif 'fc2' in k:
    #         if v:
    #             target_score = fc2_score[v[0]]
    #     elif 'fc1' in k:
    #         if v:
    #             target_score = fc1_score[v[0]]
    #     print(target_score)
    for k,v in differ.items():
        print(k,v)
        if 'output' in k:
            if v:
                for i in v:
                    if target_score < output_score[i]:
                        target_score = output_score[i]
        elif 'fc3' in k:
            if v:
                for i in v:
                    if target_score < fc3_score[i]:
                        target_score = fc3_score[i]
        elif 'fc2' in k:
            if v:
                for i in v:
                    if target_score < fc2_score[i]:
                        target_score = fc2_score[i]
        elif 'fc1' in k:
            if v:
                for i in v:
                    print(fc1_score[i])
                    if target_score < fc1_score[i]:
                        target_score = fc1_score[i]
    print(target_score)
    for k,v in differ.items():
        if 'output' in k:
            if v:
                result = output_score.reshape(-1, 1)
                target = 1
                # flag = False
                for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
                    if result[i][0] > target_score:
                        target = target + 1
                filepath1 = filepath.rsplit("/", 1)[0]
                if not os.path.exists(filepath1 + '/report_forward/'):
                    os.makedirs(filepath1 + '/report_forward/')
                if not os.path.exists(filepath1 + '/report_forward/report.npy'):  # 若不存在路径则创建
                    report = {}
                    np.save(filepath1 + '/report_forward/report.npy', report)
                report = np.load(filepath1 + '/report_forward/report.npy').item()
                report[filepath.rsplit("/", 1)[1]] = [target, target / len(result)]
                print(report)
                np.save(filepath1 + '/report_forward/report.npy', report)
                print(target)
        elif 'fc3' in k:
            if v:
                result = fc3_score.reshape(-1, 1)
                target = 1
                # flag = False
                for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
                    if result[i][0] > target_score:
                        target = target + 1
                filepath1 = filepath.rsplit("/", 1)[0]
                if not os.path.exists(filepath1 + '/report_forward/'):
                    os.makedirs(filepath1 + '/report_forward/')
                if not os.path.exists(filepath1 + '/report_forward/report.npy'):  # 若不存在路径则创建
                    report = {}
                    np.save(filepath1 + '/report_forward/report.npy', report)
                report = np.load(filepath1 + '/report_forward/report.npy').item()
                report[filepath.rsplit("/", 1)[1]] = [target, target / len(result)]
                print(report)
                np.save(filepath1 + '/report_forward/report.npy', report)
                print(target)
        elif 'fc2' in k:
            if v:
                result = fc2_score.reshape(-1, 1)
                target = 1
                # flag = False
                for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
                    if result[i][0] > target_score:
                        target = target + 1
                filepath1 = filepath.rsplit("/", 1)[0]
                if not os.path.exists(filepath1 + '/report_forward/'):
                    os.makedirs(filepath1 + '/report_forward/')
                if not os.path.exists(filepath1 + '/report_forward/report.npy'):  # 若不存在路径则创建
                    report = {}
                    np.save(filepath1 + '/report_forward/report.npy', report)
                report = np.load(filepath1 + '/report_forward/report.npy').item()
                report[filepath.rsplit("/", 1)[1]] = [target, target / len(result)]
                print(report)
                np.save(filepath1 + '/report_forward/report.npy', report)
                print(target)
        elif 'fc1' in k:
            if v:
                result = fc1_score.reshape(-1, 1)
                target = 1
                # flag = False
                for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
                    if result[i][0] > target_score:
                        target = target + 1
                filepath1 = filepath.rsplit("/", 1)[0]
                if not os.path.exists(filepath1 + '/report_forward/'):
                    os.makedirs(filepath1 + '/report_forward/')
                if not os.path.exists(filepath1 + '/report_forward/report.npy'):  # 若不存在路径则创建
                    report = {}
                    np.save(filepath1 + '/report_forward/report.npy', report)
                report = np.load(filepath1 + '/report_forward/report.npy').item()
                report[filepath.rsplit("/", 1)[1]] = [target, target / len(result)]
                print(report)
                np.save(filepath1 + '/report_forward/report.npy', report)
                print(target)
    # output_score_ = output_score.reshape(-1,1)
    # fc3_score_ = fc3_score.reshape(-1,1)
    # fc2_score_ = fc2_score.reshape(-1,1)
    # fc1_score_ = fc1_score.reshape(-1,1)
    # result = np.vstack((output_score_,fc3_score_))
    # result = np.vstack((result,fc2_score_))
    # result = np.vstack((result,fc1_score_))

def get_neural_rank_final(filepath):
    output_score_forward = np.loadtxt(filepath +'/score_forward/neural_output_score.txt')
    fc3_score_forward = np.loadtxt(filepath +'/score_forward/neural_fc3_score.txt')
    fc2_score_forward = np.loadtxt(filepath +'/score_forward/neural_fc2_score.txt')
    fc1_score_forward = np.loadtxt(filepath +'/score_forward/neural_fc1_score.txt')
    output_score = np.loadtxt(filepath + '/score/neural_output_score.txt')
    fc3_score = np.loadtxt(filepath + '/score/neural_fc3_score.txt')
    fc2_score = np.loadtxt(filepath + '/score/neural_fc2_score.txt')
    fc1_score = np.loadtxt(filepath + '/score/neural_fc1_score.txt')

    output_score_forward = Normalize(output_score_forward, 0, 1)
    fc3_score_forward = Normalize(fc3_score_forward, 0, 1)
    fc2_score_forward = Normalize(fc2_score_forward, 0, 1)
    fc1_score_forward = Normalize(fc1_score_forward, 0, 1)

    output_score = Normalize(output_score, 0, 1)
    fc3_score = Normalize(fc3_score, 0, 1)
    fc2_score = Normalize(fc2_score, 0, 1)
    fc1_score = Normalize(fc1_score, 0, 1)
    differ = np.load(filepath +'/predicted/neural_differ.npy').item()
    #指数
    n=1
    x = 0
    for i in output_score:
        output_score[x] = output_score[x] ** n + output_score_forward[x] ** n
        #output_score[x] = output_score[x] + output_score_forward[x]
        x = x + 1
    x = 0
    for i in fc3_score:
        fc3_score[x] = fc3_score[x] ** n + fc3_score_forward[x] ** n
        #fc3_score[x] = fc3_score[x] + fc3_score_forward[x]
        x = x + 1
    x = 0
    for i in fc2_score:
        fc2_score[x] = fc2_score[x] ** n + fc2_score_forward[x] ** n
        #fc2_score[x] = fc2_score[x] + fc2_score_forward[x]
        x = x + 1
    x = 0
    for i in fc1_score:
        fc1_score[x] = fc1_score[x] ** n + fc1_score_forward[x] ** n
        #fc1_score[x] = fc1_score[x] + fc1_score_forward[x]
        x = x + 1
    print(output_score)
    target_score = -100
    for k,v in differ.items():
        print(k,v)
        if 'output' in k:
            if v:
                for i in v:
                    if target_score < output_score[i]:
                        target_score = output_score[i]
        elif 'fc3' in k:
            if v:
                for i in v:
                    if target_score < fc3_score[i]:
                        target_score = fc3_score[i]
        elif 'fc2' in k:
            if v:
                for i in v:
                    if target_score < fc2_score[i]:
                        target_score = fc2_score[i]
        elif 'fc1' in k:
            if v:
                for i in v:
                    print(fc1_score[i])
                    if target_score < fc1_score[i]:
                        target_score = fc1_score[i]
    print(target_score)
    output_score_ = output_score.reshape(-1,1)
    fc3_score_ = fc3_score.reshape(-1,1)
    fc2_score_ = fc2_score.reshape(-1,1)
    fc1_score_ = fc1_score.reshape(-1,1)
    result = np.vstack((output_score_,fc3_score_))
    result = np.vstack((result,fc2_score_))
    result = np.vstack((result,fc1_score_))
    target = 1
    # flag = False
    for i in range(len(result) - 1):  # 这个循环负责设置冒泡排序进行的次数
        if result[i][0] > target_score:
            target = target + 1
    filepath1 = filepath.rsplit("/", 1)[0]
    if not os.path.exists(filepath1 + '/report_final/'):
        os.makedirs(filepath1 + '/report_final/')
    if not os.path.exists(filepath1+'/report_final/report.npy'):  # 若不存在路径则创建
        report = {}
        np.save(filepath1+'/report_final/report.npy', report)
    report = np.load(filepath1+'/report_final/report.npy').item()
    report[filepath.rsplit("/", 1)[1]] = [target,target/len(result)]
    print(report)
    np.save(filepath1+'/report_final/report.npy', report)
    print(target)
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

# get_neural_score()
# get_neural_rank()
# judge_neural_rank()
#get_weight_score()
# get_weight_rank()
#judge_weight_rank()

