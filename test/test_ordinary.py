import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
import os
import sys
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
from scipy.optimize import minimize
import time as tm

sys.path.append('../methods')
sys.path.append('../Problems')

import DD1 as F1
import DEB as F2
import Far1 as F3
import FDS as F4
import FF1 as F5
import Hil1 as F6
import JOS_ill1 as F7
import JOS_ill2 as F8
import VU1 as F9
import WIT1 as F10
import WIT2 as F11
import WIT3 as F12



import PG as M1
import APG_ls as M2
import BB as M3
import ABB_ls as M4

########################subproblem solver tolerance = 1e-9 and 1e-11##############################

######################################### ordinary problem ############################################
problems = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12]    #
Problem = ["DD1","DEB","Far1","FDS","FF1","Hil1","Imb1","Imb2","VU1","WIT1","WIT2","WIT3"]        #
methods = [M1,M2,M3,M4]                                                                        #
Method = ["PGMO","APGMO","SPGMO","ASPGMO"]                                                           #
######################################################################################################
# methods = [M4]                                                                        #
# Method = ["ASPGMO"]


# ######################################### problem QP ############################################
# problems = [QP1, QP2, QP3, QP4, QP5, QP6]                                                    #
# methods = [M2,M3,M4]                                                                         #
# Method = ["VMMO","BB","BBQN"]                                                                #
# Problem = ["QPa","QPb","QPc","QPd","QPe","QPf"]                                              #
# #################################################################################################



erro = 1e-6
k_max = 500

num = 200
Iter = np.zeros((len(problems) * num, len(methods)))
Feval = np.zeros((len(problems) * num, len(methods)))
Time = np.zeros((len(problems) * num, len(methods)))

print(Iter.shape)


for method in methods:
    c = methods.index(method)  # index of method
    for Func in problems:
        r = problems.index(Func)  # index of problem
        print(Method[c], Problem[r])
        dim = Func.dim
        width = Func.width
        interval_start = Func.interval_start
        np.random.seed(1110)
        X = width * np.random.rand(num, dim) + interval_start

        iter = []
        feval = []
        time = []
        endpoint = []

        for i in range(num):
            start = tm.perf_counter()
            list, k, s = method.MGD(Func,X[i], erro,k_max)
            end = tm.perf_counter()

            if len(np.shape(list)) == 1:
                endpoint.append(list)
            else:
                endpoint.append(list[-1])

            iter.append(len(list)-1) # save iteration
            feval.append(k) # save function evaluation
            time.append(end - start) # save time
        print('iter: {}'.format(sum(iter) / num),
              'feval: {}'.format(sum(feval) / num),
              'time: {}'.format(round(1000 * (sum(time) / num), 2)))
        Iter[r*num:(r+1)*num,c] = iter
        Feval[r * num:(r + 1) * num, c] = feval
        Time[r * num:(r + 1) * num, c] = time
        np.savetxt('../data/Ordinary/e2000' + Method[c] + Problem[r], endpoint)

np.savetxt('../data/Profile/Iter2000', Iter)
np.savetxt('../data/Profile/Feval2000', Feval)
np.savetxt('../data/Profile/Time2000', Time)
