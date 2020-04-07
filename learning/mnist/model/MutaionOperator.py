# import pickle as pickle
#
# f = open('model_paras.pkl', 'rb+')
# info = pickle.load(f)
# print
# info
import torch
import copy
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import math
from collections import OrderedDict
import random
import re


class MutaionOperator(object):
    def __init__(self, ration, model, acc_tolerant=0.90, verbose=True, test=True, test_data_laoder=None):
        self.ration = ration
        self.model = model
        self.verbose = verbose
        self.test_data_laoder = test_data_laoder

    def ns(self, skip=10):
        unique_neurons = 0
        mutation_model = self.model
        for idx_layer, param in mutation_model.named_parameters():
            shape = param.size()#参数矩阵的形状
            dim = len(shape)  # 定轴参数大小 即是几维的矩阵
            unique_neurons_layer = shape[0]#矩阵的列数，即是前一层的输出
            # skip the bias
            if dim > 1 and unique_neurons_layer >= skip:
                import math
                temp = unique_neurons_layer * self.ration
                num_mutated = math.floor(temp) if temp > 2. else math.ceil(temp)
                mutated_neurons = np.random.choice(unique_neurons_layer,
                                                   int(num_mutated), replace=False)
                m = mutated_neurons

                switch = copy.copy(mutated_neurons)
                np.random.shuffle(switch)

                param.data[mutated_neurons] = param.data[switch]
                if self.verbose:
                    print(">>:mutated neurons in {0} layer:{1}/{2}".format(idx_layer, len(mutated_neurons),

                                                                           unique_neurons_layer))

        return mutation_model


if __name__ == '__main__':
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
    i = 0
    for name, param in net.named_parameters():
        i= i+1
        if i == 3:
            param[random.randint(0,500)][random.randint(0,500)] = random.uniform(300,500)
        print(name, param)
    # operator = MutaionOperator(ration=0.1, model=net)
    # operator.ns()
    torch.save(net.state_dict(), 'mutants/model_paras.pkl')

