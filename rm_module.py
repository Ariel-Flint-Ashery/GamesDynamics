# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:39:15 2022

@author: ariel

Regret Matching Module
"""

import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from blotto_utils import *
#%%
def getStrategy(regretSum, strategySum, a):
    normalizingSum = 0
    strategy = [0] * a
    # Normalizingsum is the sum of positive regrets. 
    # This ensures we do not 'over-adjust' our strategy
    for i in range(0, a):
        if regretSum[i] > 0:
            strategy[i] = regretSum[i]
        else:
            strategy[i] = 0
    normalizingSum = sum(strategy)
    # This loop normalizes our updated strategy
    for i in range(0, a):
        if normalizingSum > 0:
            strategy[i] = strategy[i]/normalizingSum
        else:
            print("normalizing sum == 0")
            strategy[i] = 1.0 / a
        strategySum[i] += strategy[i]
    return (strategy, strategySum, normalizingSum/a)

def getStrategyCurrent(regretSum, mu, index, a, t):
    strategy = [0] * a
    avgRegret = [regret/t for regret in regretSum]
    r = [0]*a
    for i in range(0,a):
        if i == index:
            continue
        
        if avgRegret[i] > 0:
            strategy[i] = avgRegret[i]*(1/mu)
            r[i] = avgRegret[i] 
        if avgRegret[i] <= 0:
            strategy[i] = 0

    strategy[index] = 1 - sum(strategy)
    
    return strategy, sum(r)/a


def train1Player(G, iterations, regretSum, oppStrategy):
    s, k, a, val = G
    actionUtility = [0] * a[0]
    strategySum = [0] * a[0]
    #regretTotal = []
    avgUtilityRoll = []
    normSum = []
    strats = []
    for i in tqdm(range(0, iterations)):
        ##Retrieve Actions
        t = getStrategy(regretSum, strategySum, a[0])
        strategy = t[0]
        strategySum = t[1]
        myAction = getAction(s[0], strategy, k)[0]
        oppAction = getAction(s[1], oppStrategy, k)[0]
        actionList = generateActions(s[0], k)
        
        for j in range(len(actionList)):
            actionUtility[j] = getUtility(actionList[j], oppAction, val)
         
        #avgUtilityList = []
        #Add the regrets from this decision
        for j in range(a[0]):
            regretSum[j] += actionUtility[j] - getUtility(myAction, oppAction, val)
            #avgUtilityList.append(avgUtility)
            #regretSum[i] += actionUtility[i] - getUtility(myAction,oppAction)
        #print(strategySum)
        normSum.append(sum(strategySum))
        strats.append(strategy)
        #avgUtilityRoll.append(sum(avgUtilityList)/actions)
        #regretTotal.append(t[2])
        avgUtilityRoll.append(utilityAverage(s, k, strategy, oppStrategy, val))
    return strategySum, (strats, normSum, avgUtilityRoll)

def getMaxExploitStrategy(G, iterations,oppStrategy = None):
    s, k, a, val = G
    regretSum = [0] * a[0]
    #print('Initial regret sum is', regretSum)
    if oppStrategy == None:
        oppStrategy = defaultStrat(s[1], k)
    
    strategySum, delta = train1Player(G, iterations,regretSum,oppStrategy)
    normalizingSum = 0
    #actions = 21
    avgStrategy = [0] * a[0]
    for i in range(0,a[0]):
        normalizingSum += strategySum[i]
    for i in range(0,a[0]):
        if normalizingSum > 0:
            avgStrategy[i] = strategySum[i] / normalizingSum
        else:
            print("normalizing sum = 01")
            avgStrategy[i] = 1.0 / a[0]
    return avgStrategy, delta


def train2Player(G, iterations,regretSum1,regretSum2,p2Strat):
    ##Adapt Train Function for two players
    s, k, a, val = G
    actionList1 = generateActions(s[0], k)
    actionList2 = generateActions(s[1], k)
    actionUtility1 = [0] * a[0]
    actionUtility2 = [0] * a[1]
    strategySum1 = [0] * a[0]
    strategySum2 = [0] * a[1]
    normSum1 = []
    normSum2 = []
    strats1 = []
    strats2 = []
    avgUtilityRoll1 = []
    avgUtilityRoll2 = []
    regret = []
    for i in tqdm(range(0,iterations)):
        ##Retrieve Actions
        t1 = getStrategy(regretSum1,strategySum1, a[0])
        strategy1 = t1[0]
        strategySum1 = t1[1]
        regret.append(t1[2])
        myAction = getAction(s[0], strategy1, k)[0]
        
        if i == 0:
            print("""--------------------
                     NEW ACTION BEGINING
                  --------------------""")
            otherAction = getAction(s[1], p2Strat, k)[0]
        else:
            t2 = getStrategy(regretSum2,strategySum2, a[1])
            strategy2 = t2[0]
            strategySum2 = t2[1]
            #r1 = t2[2]
            otherAction = getAction(s[1], strategy2, k)[0]
        
        
        for j in range(a[0]):
            actionUtility1[j] = getUtility(actionList1[j],otherAction, val)
        for j in range(a[1]):
            actionUtility2[j] = getUtility(actionList2[j],myAction, val)
    
        for j in range(a[0]):
            regretSum1[j] += actionUtility1[j] - getUtility(myAction, otherAction, val)
        for j in range(a[1]):    
            regretSum2[j] += actionUtility2[j] - getUtility(otherAction, myAction, val)
            
        normSum1.append(sum(strategySum1))
        strats1.append(strategy1)
        normSum2.append(sum(strategySum2))

        if i == 0:
            strats2.append(p2Strat)
            avgUtilityRoll1.append(utilityAverage(s, k, strategy1, p2Strat, val))
            avgUtilityRoll2.append(utilityAverage(s, k, p2Strat, strategy1, val, enemy = True))
        else:
            strats2.append(strategy2)
            avgUtilityRoll1.append(utilityAverage(s, k, strategy1, strategy2, val))
            avgUtilityRoll2.append(utilityAverage(s, k, strategy2, strategy1, val, enemy = True))
            
    return (strategySum1, strategySum2), (strats1, strats2, normSum1, normSum2, avgUtilityRoll1, avgUtilityRoll2, regret)
#%%
def train2PlayerCurrent(G, iterations):
    ##Adapt Train Function for two players
    s, k, a, val = G
    actionList1 = generateActions(s[0], k)
    actionList2 = generateActions(s[1], k)
    strategy1 = defaultStrat(s[0], k)
    strategy2 = defaultStrat(s[1], k)
    R1, R2 = [], []
    strats1 = []
    strats2 = []
    avgUtilityRoll1 = []
    avgUtilityRoll2 = []
    actionsHistory1, actionsHistory2 = [], []
    z1, z2 = [0]*a[0], [0]*a[1]
    mu = 2*sum(val)*max(a) + 1
    for i in tqdm(range(0,iterations)):
        regretSum1 = [0] * a[0] 
        regretSum2 = [0] * a[1]
        ##Retrieve Actions
        myAction, myIndex = getAction(s[0], strategy1, k)
        otherAction, otherIndex = getAction(s[1], strategy2, k)
        #strategySum1 = t1[1]
        z1[myIndex] += 1
        z2[otherIndex] += 1
                
        actionsHistory1.append(myAction)
        actionsHistory2.append(otherAction)
        for action in actionsHistory2:
            for j in range(a[0]):
                actionUtility = getUtility(actionList1[j], action, val)
                regretSum1[j] += actionUtility - getUtility(myAction, action, val)
                
        for action in actionsHistory1:
            for j in range(a[1]):
                actionUtility = getUtility(actionList2[j], action, val)
                regretSum2[j] += actionUtility - getUtility(otherAction, action, val)
        
        #print(regretSum1)
        
        strategy1, r1 = getStrategyCurrent(regretSum1, mu, myIndex, a[0], i+1)
        strategy2, r2 = getStrategyCurrent(regretSum2, mu, myIndex, a[1], i+1)
        strats1.append(strategy1)
        strats2.append(strategy2)
        R1.append(r1)
        R2.append(r2)

        avgUtilityRoll1.append(utilityAverage(s, k, strategy1, strategy2, val))
        avgUtilityRoll2.append(utilityAverage(s, k, strategy2, strategy1, val, enemy = True))
        
    return (strategy1, strategy2), (strats1, strats2, R1, R2, avgUtilityRoll1, avgUtilityRoll2, z1, z2)
#%%
def optimise2Player(G, iterations,p2Strat = None):
    s, k, a, val = G
    
    if p2Strat == None:
        p2Strat = defaultStrat(s[1], k)
        
    strats, delta = train2Player(G, iterations,[0] * a[0], [0] * a[1], p2Strat)
    s1 = sum(strats[0])
    s2 = sum(strats[1])
    for i in range(a[0]):
        if s1 > 0:
            strats[0][i] = strats[0][i]/s1
            
    for i in range(a[1]):
        if s2 > 0:
            strats[1][i] = strats[1][i]/s2
    return strats, delta

#%%

# def optimise2PlayerCurrent(G, iterations, p2Strat = None):
#     s, k, a, val = G
    
#     if p2Strat == None:
#         p2Strat = defaultStrat(s[1], k)
        
#     _, x = train2PlayerCurrent(G, iterations,[0] * a[0], [0] * a[1], p2Strat)
    
#     return (x[0][-1], x[1][-1]), x 
    
#%%
def linear(x,a,b):
    return a*x + b
#%%
def curveFitPlotter(iterations, ydata, player, fname, func = linear):
    ydata = np.log(ydata)[1:]
    xdata = np.log(iterations)[1:]
    popt, pcov = curve_fit(func, xdata, ydata)

    plt.plot(xdata, ydata, marker = '.', label = 'raw')
    plt.plot(xdata, func(xdata, *popt), label = f'fit, power={popt[0]:.6f}') #error = {np.sqrt(np.diag(pcov))[0]}
    plt.grid()
    plt.xlabel('log(t)')
    plt.ylabel(f'log(Strategy delta (Player {player}))')
    plt.legend()
    plt.savefig(f'{fname}.png')
    plt.show()
    plt.clf()
    
    #print(np.sqrt(np.diag(pcov)))

#%%
def convergence1PlayerPrep(G, s1, n1):
    s, k, a, val = G
    
    #preprocessing
    
    d1 = [[] for strat in s1]
    for (strat,norm, delta) in zip(s1, n1, d1):
        for i in range(a[0]):
            delta.append(strat[i]/norm)

    ylist1 = [] #list where each element is a list of the strategy deltas for each action
    for i in range(a[0]):
        y = []
        for delta in d1:
            y.append(delta[i])
    
        ylist1.append(y)
        
    yarrays1 = [np.array(y) for y in ylist1]
    yarr_avg1 = sum(yarrays1)/len(yarrays1)
    return yarr_avg1

    # xdata = np.log(np.array(range(int(iterations*alpha))))[1:]
    # ydata1 = np.log(xarr_avg1[:int(iterations*alpha)])[1:]
    # curveFitPlotter(linear, xdata, ydata1, '1')

def convergence2PlayerPrep(G, s1, s2, n1, n2):
    s, k, a, val = G
    
    #preprocessing
    
    d1 = [[] for strat in s1]
    for (strat,norm, delta) in zip(s1, n1, d1):
        for i in range(a[0]):
            delta.append(strat[i]/norm)

    d2 = [[] for strat in s2]
    for (strat,norm, delta) in zip(s2, n2, d2):
        # if norm == n2[0]:
        #     delta += [0]*a[1]
            
        # else:
        #     for i in range(a[1]):
        #         delta.append(strat[i]/norm)
        for i in range(a[1]):
            delta.append(strat[i]/(norm+1))
    
    ylist1 = [] #list where each element is a list of the strategy deltas for each action
    for i in range(a[0]):
        y = []
        for delta in d1:
            y.append(delta[i])
    
        ylist1.append(y)
    
    ylist2 = [] 
    for i in range(a[1]):
        y = []
        for delta in d2:
            #print(len(delta))            
            y.append(delta[i])

        ylist2.append(y)
    
    yarrays1 = [np.array(y) for y in ylist1]
    yarr_avg1 = sum(yarrays1)/len(yarrays1)
    yarrays2 = [np.array(y) for y in ylist2]
    yarr_avg2 = sum(yarrays2)/len(yarrays2)
    
    #find convergence order of Player 1 & 2
    
    return yarr_avg1, yarr_avg2
    # xdata = np.log(np.array(range(int(iterations*alpha))))[1:]
    # ydata1 = np.log(xarr_avg1[:int(iterations*alpha)])[1:]
    # ydata2 = np.log(xarr_avg2[:int(iterations*alpha)])[1:]
    
    # curveFitPlotter(linear, xdata, ydata1, '1')
    # curveFitPlotter(linear, xdata, ydata2, '2')
    
