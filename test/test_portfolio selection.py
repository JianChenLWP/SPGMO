import numpy as np
import time as tm
import sys
sys.path.append('../methods')
sys.path.append('../Problems')
import PG_ls_PS as M1
import APG_ls_PS as M2
import BB_ls_PS as M3
import ABB_ls_PS as M4

import FF25a as QP1
import FF25b as QP2
import FF25c as QP3
import FF25d as QP4

import FF32a as QP5
import FF32b as QP6
import FF32c as QP7
import FF32d as QP8

import FF48a as QP9
import FF48b as QP10
import FF48c as QP11
import FF48d as QP12

import FF100a as QP13
import FF100b as QP14
import FF100c as QP15
import FF100d as QP16


######################################### ordinary problem ############################################
problems = [QP1,QP2,QP3,QP4,QP5,QP6,QP7,QP8,QP9,QP10,QP11,QP12,QP13,QP14,QP15,QP16]    #
Problem = ["FF25a","FF25b","FF25c","FF25d","FF32a","FF32b","FF32c","FF32d","FF48a","FF48b","FF48c","FF48d","FF100a","FF100b","FF100c","FF100d"]        #
methods = [M1,M2,M3,M4]                                                                        #
Method = ["PGMO","APGMO","SPGMO","ASPGMO"]                                                             #
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



erro = 1e-6
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
        X = X / np.sum(X, axis=1).reshape((len(X), 1))    #simplex feasible

        iter = []
        feval = []
        time = []
        endpoint = []

        for i in range(num):
            start = tm.perf_counter()
            list, k, s = method.MGD(Func,X[i], erro,k_max)
            end = tm.perf_counter()
            # print("round", i,len(list)-1,round(1000 * (end - start), 2))

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
        np.savetxt('../data/PS/PS' + Method[c] + Problem[r], endpoint)

np.savetxt('../data/Profile/Iter_PS', Iter)
np.savetxt('../data/Profile/Feval_PS', Feval)
np.savetxt('../data/Profile/Time_PS', Time)

