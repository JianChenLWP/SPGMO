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

# import DD1 as F1
# import DEB as F2
# import Far1 as F3
# import FDS as F4
# import FF1 as F5
# import Hil1 as F6
# import JOS_ill1 as F7
# import JOS_ill2 as F8
# import VU1 as F9
# import WIT1 as F10
# import WIT2 as F11
# import WIT3 as F12


import PG_L as M1
import APG_L as M2
import APG_mu as M3
import BB_L as M4
import ABB_L as M5
import ABB_mu as M6

import QPa as QP1
import QPb as QP2
import QPc as QP3
import QPd as QP4
import QPe as QP5
import QPf as QP6
import QPg as QP7
import QPh as QP8

########################subproblem solver tolerance = 1e-9 and 1e-11##############################


######################################### ordinary problem ############################################
problems = [QP1,QP2,QP3,QP4,QP5,QP6,QP7,QP8]    #
Problem = ["QPa","QPb","QPc","QPd","QPe","QPf","QPg","QPh"]        #
methods = [M1,M2,M3,M4,M5,M6]                                                                        #
Method = ["PGMO","APGMO","APGMO-sc","SPGMO","ASPGMO","ASPGMO-sc"]                                                           #
######################################################################################################
# problems = [QP6]    #
# Problem = ["QPf"]
# methods = [M6]                                                                        #
# Method = ["ASPGMO-sc"]


# ######################################### problem QP ############################################
# problems = [QP1, QP2, QP3, QP4, QP5, QP6]                                                    #
# methods = [M2,M3,M4]                                                                         #
# Method = ["VMMO","BB","BBQN"]                                                                #
# Problem = ["QPa","QPb","QPc","QPd","QPe","QPf"]                                              #
# #################################################################################################



erro = 1e-4
k_max = 3000

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
            list, k, s, dist = method.MGD(Func,X[i], erro,k_max)
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
              'time: {}'.format(round(1000 * (sum(time) / num), 2)),
              'error: {}'.format(dist[-1]))
        Iter[r*num:(r+1)*num,c] = iter
        Feval[r * num:(r + 1) * num, c] = feval
        Time[r * num:(r + 1) * num, c] = time
        np.savetxt('../data/QP/L2000' + Method[c] + Problem[r], endpoint)

np.savetxt('../data/Profile/Iter_QP2000', Iter)
np.savetxt('../data/Profile/Feval_QP2000', Feval)
np.savetxt('../data/Profile/Time_QP2000', Time)


