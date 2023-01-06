# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 16:56:36 2022

@author: ariel
"""

import random
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
def erevRothRL1Player(G, iterations, oppStrategy):
    #initialise
    s, k, a, val = G
    actions = generateActions(s[0], k)
    #myStrategy = np.array(myStrategy)
    defStrat = defaultStrat(s[0], k)
    myStrategy = np.array(defStrat)
    theta = np.array(defStrat)
    x = np.identity(len(actions))
    history = []
    for t in tqdm(range(0,iterations)):
        myAction, myIndex = getAction(s[0], myStrategy, k)
        oppAction, oppIndex = getAction(s[1], oppStrategy, k)
        u = getUtility(myAction, oppAction, val)
        if u > 0:
            theta += u*x[myIndex]
            myStrategy = theta/sum(np.abs(theta))
        history.append(myStrategy)

    return myStrategy, history
#%%
def erevRothRL2Player(G, iterations):
    #initialise
    s, k, a, val = G
    actions1 = generateActions(s[0], k)
    actions2 = generateActions(s[1], k)
    #myStrategy = np.array(myStrategy)
    defStrat1 = defaultStrat(s[0], k)
    defStrat2 = defaultStrat(s[1], k)
    myStrategy = np.array(defStrat1)
    oppStrategy = np.array(defStrat2)
    theta1 = np.array(defStrat1)
    theta2 = np.array(defStrat2)
    x1 = np.identity(len(actions1))
    x2 = np.identity(len(actions2))
    for t in tqdm(range(0,iterations)):
        myAction, myIndex = getAction(s[0], myStrategy, k)
        oppAction, oppIndex = getAction(s[1], oppStrategy, k)
        u1 = getUtility(myAction, oppAction, val)
        u2 = getUtility(oppAction, myAction, val)
        if u1 > 0:
            theta1 += u1*x1[myIndex]
            myStrategy = theta1/sum(np.abs(theta1))
        
        if u2 > 0:
            theta2 += u2*x2[oppIndex]
            oppStrategy = theta2/sum(np.abs(theta2))
        
    return myStrategy, oppStrategy
#%%
"Find average behaviour for RL against default strategy"

s = [5,5]
G = generateGame(s, 3)
s, k, a, val = G
iterations = 100000
numberGames = 200

rlStratList, historyList = [], []
avgRLStrat = np.array([0.0]* a[0]) #across n games
for i in range(numberGames):
    rlStrat, history = erevRothRL1Player(G, iterations, defaultStrat(s[1], 3)) 
    avgRLStrat += np.array(rlStrat)
    rlStratList.append(rlStrat)
    historyList.append(history)
    
#normalize average RL strategy
avgRLStrat /= numberGames

# rlFinalStrat = rlStratList[-1]

#plot convergence epsilon of RL
#%%
#find average convergence rate
epsilonHistory = []
for history in historyList:
    e = [sum(history[i+1] - history[i])/a[0] for i in range(len(history)-1)]
    epsilonHistory.append(np.array(e))
    
averageEpsilon = sum(epsilonHistory)/numberGames

#%%

plt.plot(np.array(range(len(averageEpsilon)))[:10000]+1, np.abs(averageEpsilon)[:10000]) #label = f'final epsilon = {averageEpsilon[-1]}')
plt.ylabel('Average epsilon')
plt.xlabel('Iterations')
#plt.legend()
plt.savefig('RL Average Convergence Plot.png')
plt.show()
#%%
curveFitPlotter(np.array(range(len(averageEpsilon)))+1, np.abs(averageEpsilon), '1', 'RL convergence linear')
#curveFitPlotter(np.array(range(len(epsilonHistory[0]))[:10000])+1, np.abs(epsilonHistory[0])[:10000], '1', 'RL convergence linear')
#%%
plt.plot(np.array(range(len(epsilonHistory[0]))[:200])+1, np.abs(epsilonHistory[0][:200]), label = f'final epsilon = {epsilonList[-1]}')
plt.ylabel('epsilon')
plt.xlabel('Iterations')
plt.legend()
plt.savefig('RL Example Convergence Plot.png')
plt.show()
#%%

#%%
#set colors
colorList1 = ['red']*a[0]

#list of Nash equlibrium strats indicies (found using Hart method)
#Note: this Nash list specifically applies for the game 
#G = generateGame(s, 3, [1, 1, 1])
#for a different game, find the other NE manually and change this list



#plot average strategy of RL
#%%
df = stratTodf(s[0], avgRLStrat, k)
plotStrat(df, 'RL Average MIXED Strat', colorList1)


#%%
"PLAY AGAINST A FIXED PURE STRATEGY"
s = [5,5]
G = generateGame(s, 3)
s, k, a, val = G
iterations = 1000
numberGames = 100

rlStratList, historyList = [], []
avgRLStrat = np.array([0.0]* a[0]) #across n games
for i in range(numberGames):
    rlStrat, history = erevRothRL1Player(G, iterations, [0,1]+[0]*19) 
    avgRLStrat += np.array(rlStrat)
    rlStratList.append(rlStrat)
    historyList.append(history)
    
#normalize average RL strategy
avgRLStrat /= numberGames

# rlFinalStrat = rlStratList[-1]

#plot convergence epsilon of RL
#%%
#find average convergence rate
epsilonHistory = []
for history in historyList:
    e = [sum(history[i+1] - history[i])/a[0] for i in range(len(history)-1)]
    epsilonHistory.append(np.array(e))
    
averageEpsilon = sum(epsilonHistory)/numberGames

#%%

plt.plot(np.array(range(len(averageEpsilon)))[:1000]+1, np.abs(averageEpsilon)[:1000]) #label = f'final epsilon = {averageEpsilon[-1]}')
plt.ylabel('Average epsilon')
plt.xlabel('Iterations')
#plt.legend()
#plt.savefig('RL Average Convergence Plot Pure Opp 1000.png')
plt.show()


#%%
curveFitPlotter(np.array(range(len(averageEpsilon)))[:30]+1, np.abs(averageEpsilon[:30]), '1', 'RL convergence linear Pure Opp')
#curveFitPlotter(np.array(range(len(epsilonHistory[0]))[:10000])+1, np.abs(epsilonHistory[0])[:10000], '1', 'RL convergence linear')
#%%
plt.plot(np.array(range(len(epsilonHistory[0])))+1, np.abs(epsilonHistory[0]))#, label = f'final epsilon = {epsilonList[-1]}')
plt.ylabel('epsilon')
plt.xlabel('Iterations')
plt.legend()
plt.savefig('RL Example Convergence Plot Pure Opp.png')
plt.show()

#%%
colorList1 = ['red']*a[0]


#list of Nash equlibrium strats indicies (found using Hart method)
#Note: this Nash list specifically applies for the game 
#G = generateGame(s, 3, [1, 1, 1])
#for a different game, find the other NE manually and change this list

# Nash1 = [2, 3, 7, 9, 11, 14, 15, 16, 17]
# for i in range(len(colorList1)):
#     if i in Nash1:
#         colorList1[i] = 'green'
        
#find average final strategy for RL player

df1 = stratTodf(s[0], avgRLStrat.tolist(), k)
#%%
#plot average strategies
plotStrat(df1, f'Player 1 RL Avg FINAL Strategy {iterations} its {numberGames} games', colorList1)