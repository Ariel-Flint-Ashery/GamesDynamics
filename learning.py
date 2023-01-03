# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:30:20 2022

@author: ariel


Compare learning speeds across various agents. I.e. one agent learns on RL,
another agent learns using regret.

Find rates of convergence etc. 


"""

import random
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from blotto_utils import *
from rm_module import *
#%%

def trainDualLearnerCurrent(G, iterations):
    s, k, a, val = G
    #intiate
    rlActionList = generateActions(s[0], k) #RL Agent actions
    rmActionList = generateActions(s[1], k) #RM Agent actions
    rlStrategy = np.array(defaultStrat(s[0], k))
    rmStrategy = np.array(defaultStrat(s[1], k))
    theta = np.array(defaultStrat(s[0], k))
    x = np.identity(a[0])
    #rmActionUtility = [0] * a[1]
    rmStrats, rlStrats = [], []
    rlActionHistory = []
    R = []
    rlAvgUtility, rmAvgUtility = [], []
    mu = 2*sum(val)*a[1] + 1
    for t in tqdm(range(iterations)):
    
        regretSum = [0]*a[1]
        #choose Agent actions
        rlAction, rlIndex = getAction(s[0], rlStrategy, k) 
        rlActionHistory.append(rlAction)
        rmAction, rmIndex = getAction(s[1], rmStrategy, k)
        #rmActionList = generateActions(s[1], k)
        #train RL Agent
        rlUtility = getUtility(rlAction, rmAction, val)
        if rlUtility > 0:
            theta += rlUtility*x[rlIndex]
            rlStrategy = theta/sum(np.abs(theta))
        else:
            pass
            
        #train RM Agent
        
        for action in rlActionHistory:
            for j in range(a[1]):
                rmActionUtility = getUtility(rmActionList[j], action, val)
                regretSum[j] += rmActionUtility - getUtility(rmAction, action, val)
        

        rmStrategy, r = getStrategyCurrent(regretSum, mu, rmIndex, a[1], t+1)
        R.append(r) #average regret across all actions at time t
        rmStrats.append(rmStrategy)
        rlStrats.append(rlStrategy)
        rlAvgUtility.append(utilityAverage(s, k, rlStrategy, rmStrategy, val))
        rmAvgUtility.append(utilityAverage(s, k, rmStrategy, rlStrategy, val), enemy = True)
    return (rlStrats, rmStrats), R, (rlAvgUtility, rmAvgUtility)
#%%
def trainDualLearner(G, regretSum, iterations):
    s, k, a, val = G
    #intiate
    rlActionList = generateActions(s[0], k) #RL Agent actions
    rmActionList = generateActions(s[1], k) #RM Agent actions
    rlStrategy = np.array(defaultStrat(s[0], k))
    rmStrategy = np.array(defaultStrat(s[1], k))
    theta = np.array(defaultStrat(s[0], k))
    x = np.identity(a[0])
    rmStrategySum = [0] * a[1]
    rmNormSum = []
    rmStrats, rlStrats = [], []
    #rlActionHistory = []
    R = []
    rlAvgUtility, rlAvgUtility = [], []
    mu = 2*sum(val)*a[1] + 1
    for t in tqdm(range(iterations)):
        
        rmStrategy, rmStrategySum, r = getStrategy(regretSum, rmStrategySum, a[1])
        R.append(r)
        #choose Agent actions
        rlAction, rlIndex = getAction(s[0], rlStrategy, k) 
        #rlActionHistory.append(rlAction)
        rmAction, rmIndex = getAction(s[1], rmStrategy, k)
        #rmActionList = generateActions(s[1], k)
        #train RL Agent
        rlUtility = getUtility(rlAction, rmAction, val)
        if rlUtility > 0:
            theta += rlUtility*x[rlIndex]
            rlStrategy = theta/sum(np.abs(theta))
        else:
            pass
            
        #train RM Agent
        

        for j in range(a[1]):
            rmActionUtility = getUtility(rmActionList[j], rlAction, val)
            regretSum[j] += rmActionUtility - getUtility(rmAction, rlAction, val)
        
        rmNormSum.append(sum(rmStrategySum))
       
        #R.append(r) #average regret across all actions at time t
        rmStrats.append(rmStrategy)
        rlStrats.append(rlStrategy)
        rlAvgUtility.append(utilityAverage(s, k, rlStrategy, rmStrategy, val))
        rmAvgUtility.append(utilityAverage(s, k, rmStrategy, rlStrategy, val, enemy = True))
    return (rlStrats, rmStrats), (rmStrategySum, rmNormSum, (rlAvgUtility,rmAvgUtility), R)

def runDualLearner(G, iterations):
    s, k, a, val = G
    regretSum = [0] * a[1]
    strats, data = trainDualLearner(G, regretSum, iterations)
    strategySum = data[0]
    normalizingSum = 0
    #actions = 21
    avgStrategy = [0] * a[1]
    for i in range(0,a[1]):
        normalizingSum += strategySum[i]
    for i in range(0,a[1]):
        if normalizingSum > 0:
            avgStrategy[i] = strategySum[i] / normalizingSum
        else:
            print("normalizing sum = 01")
            avgStrategy[i] = 1.0 / a[1]
    return (strats[0], avgStrategy), (strats[1], data[1], data[2], data[3])
    
#%%
"RUN TRAIN CURRENT"
#initialise
s = [5,5]
G = generateGame(s, 3, [1, 1, 1])
s, k, a, val = G
#iterations = 50000
iterations = 5000
#run
strats, regret, avgUtility = trainDualLearnerCurrent(G, iterations)
#%%
#save simulation
with open(f'dualcstrat1_{iterations}.pkl', 'wb') as f:
    pkl.dump(strats[0], f)
with open('dualcstrat2_{iterations}.pkl', 'wb') as f:
    pkl.dump(strats[1], f)  
with open(f'dualcR_{iterations}.pkl', 'wb') as f:
    pkl.dump(regret, f)  

with open(f'dualcRLUtility_{iterations}.pkl', 'wb') as f:
    pkl.dump(avgUtility[0], f)  

with open(f'dualcRMUtility_{iterations}.pkl', 'wb') as f:
    pkl.dump(avgUtility[1], f)  

#%%    
#plot regret
prange = 500
plt.plot(np.array(range(iterations))[:prange]+1, regret[:prange])
plt.xlabel('Iterations')
plt.ylabel('Regret')
plt.grid()
plt.savefig(f'Dual Learner Current Regret truncated at {prange}, {iterations} its.png')
plt.show()
#%%
#plot average utility
plt.plot(range(iterations), avgUtility)
plt.xlabel('Iterations')
plt.ylabel('Average Utility')
plt.grid()
plt.savefig(f'Dual Learner Current Average Utility {iterations} its')
plt.show()

#%%
#calculate correlation
corr = np.corrcoef(regret[50:2000], avgUtility[50:2000])
print(corr)
#%%
#calculate average change in strategy vector
ds1 = [sum(np.array(strats[0][i+1])-np.array(strats[0][i]))/a[0] for i in range(len(strats[0])-1)]
ds2 = [sum(np.array(strats[1][i+1])-np.array(strats[1][i]))/a[1] for i in range(len(strats[1])-1)]

#plot average change in strategy vector
plt.plot(np.array(range(iterations))[1:]+1, ds1, label = 'RL')
plt.xlabel('Iterations')
plt.ylabel('Average change in strategy')
plt.grid()
plt.savefig(f'Dual Learner Current Average change in strategy RL {iterations} its.png')
plt.show()

plt.plot(np.array(range(iterations))[1:]+1, ds2, label = 'RM')
plt.xlabel('Iterations')
plt.ylabel('Average change in strategy')
plt.grid()
plt.savefig(f'Dual Learner Current Average change in strategy RM {iterations} its.png')
plt.show()
#%%
#plot maximum change in strategy vector (to detect strong changes)
maxds1 = [max(np.abs(np.array(strats[0][i+1])-np.array(strats[0][i])))/a[0] for i in range(len(strats[0])-1)]
maxds2 = [max(np.abs(np.array(strats[1][i+1])-np.array(strats[1][i])))/a[1] for i in range(len(strats[1])-1)]

plt.plot(np.array(range(iterations-1))+1, maxds1, label = 'RL')
plt.xlabel('Iterations')
plt.ylabel('Maximum change in strategy')
plt.savefig(f'Dual Learner Current Maximum change in strategy RL {iterations} its.png')
plt.show()

plt.plot(np.array(range(iterations-1))+1, maxds2, label = 'RM')
plt.xlabel('Iterations')
plt.ylabel('Maximum change in strategy')
plt.savefig(f'Dual Learner Current Maximum change in strategy RM {iterations} its.png')
plt.show()

#%%
#SPACE FOR STRATEGY ZOOMING TO ANALYSE BEHAVIOUR
df1 = stratTodf(s[0], strats[0][-1], k)
df2 = stratTodf(s[1], strats[1][-1], k)
#%%
plotStrat(df1)
plotStrat(df2)

#%%

#strats, data = runDualLearner(G, 100)
#%%
# ydata = convergence1PlayerPrep(G, data[0], data[1])
# # %%
# curveFitPlotter(np.array(range(5)), ydata[:5], 'RM')
# #%%
# plt.plot(ydata)

#%%

"RUN TRAIN"
#initialise
s = [5,5]
G = generateGame(s, 3, [1, 1, 1])
s, k, a, val = G
iterations = 10000000

strats, data = runDualLearner(G, iterations) #data: rmStrat history, rmNormSum, average utility

regret = convergence1PlayerPrep(G, data[0], data[1])
rawRegret = np.array(data[3])/(np.array(range(iterations))+1)
#save simulation
with open(f'dualNellerstrat1_{iterations}.pkl', 'wb') as f:
    pkl.dump(strats[0], f)
with open(f'dualNellerstrat2_{iterations}.pkl', 'wb') as f:
    pkl.dump(strats[1], f)  
with open(f'dualNellerR_{iterations}.pkl', 'wb') as f:
    pkl.dump(regret, f)  

with open(f'dualNellerRLUtility_{iterations}.pkl', 'wb') as f:
    pkl.dump(data[2][0], f)  
    
with open(f'dualNellerRMUtility_{iterations}.pkl', 'wb') as f:
    pkl.dump(data[2][1], f)  
    
with open(f'dualNellerRawRegret_{iterations}.pkl', 'wb') as f:
    np.save(f, rawRegret)  
#%%
# #plot regret
# plt.plot(range(iterations), regret)
# plt.xlabel('Iteration')
# plt.ylabel('Average Normalised Regret')
# plt.savefig(f'Dual Learner Neller Normalised Regret {iterations} its.png')
# plt.show()
#%%
plt.plot(np.array(range(iterations))[1:200] + 1, rawRegret[1:200])
plt.xlabel('Iteration')
plt.ylabel('Average Regret')
plt.savefig(f'Dual Learner Neller Raw Regret {iterations} its truncated at 200.png')
plt.show()

#%%
# colorList = ['blue']*iterations
# colorList[300000:1100000] = 'red'
#%%
plt.plot(np.array(range(iterations))+1, data[3], color = 'blue')
plt.plot(np.array(range(iterations))[270000:1120000]+1, data[3][270000:1120000], color = 'red')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Regret')
plt.grid()
plt.savefig('Dual Learner (Neller) Cumulative Regret.png')
plt.show()
#%%
#plot convergence speed
#curveFitPlotter(np.array(range(iterations))+1, regret, 'RM', 'Dual Learner Convergence Speed')
curveFitPlotter(np.array(range(iterations))+1, rawRegret, 'RM', f'Dual Learner True Convergence Speed {iterations} its')
#plot average utility
#%%
plt.plot(np.array(range(iterations))+1, data[2][1], color ='blue')
plt.plot(np.array(range(iterations))[270000:1120000]+1, data[2][270000:1120000], color ='red')
plt.xlabel('Iterations')
plt.ylabel('Average Utility')
plt.grid()
plt.savefig(f'Dual Learner (Neller) Average Utility {iterations} its')
plt.show()
#%%


#%%
# #calculate average change in strategy vector
# ds1 = [sum(np.array(strats[0][i+1])-np.array(strats[0][i]))/a[0] for i in range(len(strats[0])-1)]
# ds2 = [sum(np.array(strats[1][i+1])-np.array(strats[1][i]))/a[1] for i in range(len(strats[1])-1)]

# #plot average change in strategy vector
# plt.plot(np.array(range(iterations))+1, ds1, label = 'RL')
# plt.xlabel('Iterations')
# plt.ylabel('Average change in strategy')
# plt.savefig('Dual Learner Current Average change in strategy RL.png')
# plt.show()

# plt.plot(np.array(range(iterations))+1, ds2, label = 'RM')
# plt.xlabel('Iterations')
# plt.ylabel('Average change in strategy')
# plt.savefig('Dual Learner Current Average change in strategy RM.png')
# plt.show()

# #plot maximum change in strategy vector (to detect strong changes)
# maxds1 = [max(np.abs(np.array(strats[0][i+1])-np.array(strats[0][i])))/a[0] for i in range(len(strats[0])-1)]
# maxds2 = [max(np.abs(np.array(strats[1][i+1])-np.array(strats[1][i])))/a[1] for i in range(len(strats[1])-1)]

# plt.plot(np.array(range(iterations))+1, maxds1, label = 'RL')
# plt.xlabel('Iterations')
# plt.ylabel('Maximum change in strategy')
# plt.savefig('Dual Learner Current Maximum change in strategy RL.png')
# plt.show()

# plt.plot(np.array(range(iterations))+1, maxds2, label = 'RM')
# plt.xlabel('Iterations')
# plt.ylabel('Maximum change in strategy')
# plt.savefig('Dual Learner Current Maximum change in strategy RM.png')
# plt.show()
#%%
"RUN TRAIN CURRENT FOR ASYMMETRIC BLOTTO GAME"
#initialise
s = [5,4]
G = generateGame(s, 3, [1, 1, 1])
s, k, a, val = G
#iterations = 50000
iterations = 1000
#run
strats, regret, avgUtility = trainDualLearnerCurrent(G, iterations)
#%%
#save simulation
with open(f'dualcstrat1_{iterations}_{s}_{val}.pkl', 'wb') as f:
    pkl.dump(strats[0], f)
with open('dualcstrat2_{iterations}_{s}_{val}.pkl', 'wb') as f:
    pkl.dump(strats[1], f)  
with open(f'dualcR_{iterations}_{s}_{val}.pkl', 'wb') as f:
    pkl.dump(regret, f)  

with open(f'dualcRLUtility_{iterations}_{s}_{val}.pkl', 'wb') as f:
    pkl.dump(avgUtility[0], f)  

with open(f'dualcRMUtility_{iterations}_{s}_{val}.pkl', 'wb') as f:
    pkl.dump(avgUtility[1], f)  

#%%    
#plot regret
prange = 1000
plt.plot(np.array(range(iterations))[:prange]+1, regret[:prange])
plt.xlabel('Iterations')
plt.ylabel('Regret')
plt.grid()
plt.savefig(f'Dual Learner Current Regret truncated at {prange}, {iterations} its, _{s}_{val}.png')
plt.show()
#%%
#plot average utility
plt.plot(range(iterations), avgUtility)
plt.xlabel('Iterations')
plt.ylabel('Average Utility')
plt.grid()
plt.savefig(f'Dual Learner Current Average Utility {iterations} its _{s}_{val}')
plt.show()

#%%
#calculate correlation
corr = np.corrcoef(regret[50:2000], avgUtility[50:2000])
print(corr)
#%%
#calculate average change in strategy vector
ds1 = [sum(np.array(strats[0][i+1])-np.array(strats[0][i]))/a[0] for i in range(len(strats[0])-1)]
ds2 = [sum(np.array(strats[1][i+1])-np.array(strats[1][i]))/a[1] for i in range(len(strats[1])-1)]

#plot average change in strategy vector
plt.plot(np.array(range(iterations))[1:]+1, ds1, label = 'RL')
plt.xlabel('Iterations')
plt.ylabel('Average change in strategy')
plt.grid()
plt.savefig(f'Dual Learner Current Average change in strategy RL {iterations} its _{s}_{val}.png')
plt.show()

plt.plot(np.array(range(iterations))[1:]+1, ds2, label = 'RM')
plt.xlabel('Iterations')
plt.ylabel('Average change in strategy')
plt.grid()
plt.savefig(f'Dual Learner Current Average change in strategy RM {iterations} its _{s}_{val}.png')
plt.show()

#%%


