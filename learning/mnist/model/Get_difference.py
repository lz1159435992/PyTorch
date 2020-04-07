import numpy as np
import os
def get_difference():
    dict = {}
    neural_dict = {}
    for dirpath, dirnames, filenames in os.walk('mutants/weight'):
        for filename in filenames:
            for dirpath_, dirnames_, filenames_ in os.walk('weight'):
                for filename_ in filenames_:
                    if filename == filename_:
                        if 'weight' in filename:
                            list = []
                            neural_list = []
                            a = np.loadtxt(dirpath + '/' + filename)
                            b = np.loadtxt(dirpath_ + '/' + filename_)
                            x = 0
                            for i in a:
                                y = 0
                                for j in i:
                                    if j != b[x][y]:
                                        list.append([x, y])
                                        neural_list.append(x)
                                    y = y + 1
                                x = x + 1
                            dict[filename] = list
                            neural_dict[filename] = neural_list
    for k,v in dict.items():
        print(k,v)
        print(len(v))
    np.save('mutants/predicted/differ.npy', dict)
    np.save('mutants/predicted/neural_differ.npy', neural_dict)
get_difference()