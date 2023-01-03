# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:44:47 2022

@author: ariel

Utility module for (discrete) Colonel Blotto games
"""

import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


#%%
random.seed(12345)

def getNumberActions(s, k):
    return math.comb(s+k-1, k-1)

def generateGame(s,k, val=None):
    a = [0]*len(s)
    for i in range(len(s)):
        a[i] = getNumberActions(s[i], k)
    if val == None:
        #val = np.ones((1,k))
        val = [1]*k
    return (s, k, a, val)

def budgetSplit(s,k,sofar=[]):
    if k==1:
        yield sofar + [s]
    else:
        for n2 in range(0,s+1):
            for b in budgetSplit(s-n2,k-1,sofar+[n2]):
                yield b 

def generateActions(s, k):
    return list(budgetSplit(s,k))


#%%
def getUtility(p1,p2, val):
    u = 0        
    for i in range(len(p1)):
        if p1[i] > p2[i]:
            u+=val[i]
            
        elif p1[i] < p2[i]:
            u-=val[i]
    return u

# def getEnemyUtility(p1,p2, val): #ONLY APPLIES FOR CONSTANT SUM GAME
#     u = 0        
#     for i in range(len(p1)):
#         if p1[i] > p2[i]:
#             u+=val[i]
            
#         elif p1[i] < p2[i]:
#             u-=val[i]
#     return u
#%%
#default as in equally weighted mixed strat
def defaultStrat(s, k):
    actionProfiles = generateActions(s, k)
    strat = []
    for i in range(len(actionProfiles)):
        strat.append(1/len(actionProfiles))
    return strat

#%%
def getAction(s, strategy, k):
    rand = random.random()
    actions = generateActions(s, k)
    leftSum = 0
    rightSum = 0
    for i in range(len(strategy)):
        rightSum+=strategy[i]
        if rand > leftSum and rand <= rightSum:
            return actions[i], i
        else:
            leftSum+=strategy[i]
    return actions[0], 0

#%%
def utilityAverage(s, k, g, r, val=None, enemy = False):
    if enemy == False:
        actions1 = generateActions(s[0], k)
        actions2 = generateActions(s[1], k)
    else:
        actions1 = generateActions(s[1], k)
        actions2 = generateActions(s[0], k)
    actionUtilities = [0] * len(actions1)
    for i in range(len(actions1)):
        u = 0
        for j in range(len(actions2)):
            u += ((getUtility(actions1[i],actions2[j], val) * r[j]))
        actionUtilities[i] = u * g[i]
    return sum(actionUtilities)

#%%
def stratTodf(s, strategy, k):
    actions = generateActions(s, k)
    act_to_string = []
    for i in actions:
        act_to_string.append("".join(str(i)))
    return pd.DataFrame(strategy,act_to_string)

def plotStrat(x, fname, colorList):
    fig, ax = plt.subplots()
    bars = ax.bar(x.index, x.values.flatten(), color = colorList)
    ax.set_xticklabels(x.index, rotation=45)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Action')
    plt.tight_layout()
    #ax.bar_label(bars, fmt = '%.3f', rotation = 90, fontsize = 8, padding = 10)
    plt.savefig(f'{fname}.png')
    plt.show()
    #plt.clf()
    #plt.cla()
