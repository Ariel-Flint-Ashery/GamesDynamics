# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:09:01 2022

@author: ariel

Test RM
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
# rmStrat1P, x = getMaxExploitStrategy(G, 10000)
# #%%
# convergence1Player(G, 10000, x[0], x[1])

#%%
# plt.plot(range(1300), x[2][:1300])
# plt.show()
#%%
"RM NELLER TESTS (SYMMETRIC)"
#constants
s = [5,5]
G = generateGame(s, 3, [1, 1, 1])
s, k, a, val = G
iterations = 1000000
#%%
#run
rmStrats2P, x = optimise2Player(G, iterations)

#find regret
regret1, regret2 = convergence2PlayerPrep(G, x[0], x[1], x[2], x[3])

#save regret
with open('RM_NL_regret1 {iterations}.pkl', 'wb') as f:
    pkl.dump(regret1, f) 
    
with open('RM_NL_regret2 {iterations}.pkl', 'wb') as f:
    pkl.dump(regret2, f) 


R = (regret1, regret2)
#plot regret
# for i in range(2):
#     plt.plot(range(iterations)[:1000], R[i][:1000])
#     plt.xlabel(f'Iterations')
#     plt.ylabel(f'Average Regret')
#     plt.savefig(f'Raw Regret RM (Neller) Player {i}.png')
#     plt.show()
#     plt.clf()

rawRegret = np.array(x[6])/(np.array(range(iterations))+1)
#%%
plt.plot(range(iterations)[:100000], rawRegret[:100000])
plt.xlabel('Iterations')
plt.ylabel('Average Regret')
plt.savefig('Raw unnormalised Regret RM (Neller).png')
plt.show()
#%%
#plot speed of convergence
curveFitPlotter(np.array(range(iterations))+1, regret1, '1', f'RM Neller Convergence Plot {iterations} its')
#%%
curveFitPlotter(np.array(range(iterations))[5:]+1, rawRegret[5:], '1', f'RM Neller NEW Convergence Plot {iterations} its')
#%%
"""
Even when plotting the curve fit up until the point 
where the average utility no longer increases, we 
see that the polynomial degree is not -0.5, but is
rather closer to -1. This supports our mixed 
convergence hypothesis.
"""
# show average strategies

df1 = stratTodf(s[0], rmStrats2P[0], k)
df2 = stratTodf(s[1], rmStrats2P[1], k)

#set colours
colorList1 = ['red']*a[0]
colorList2 = ['red']*a[1]

Nash1 = [2, 3, 7, 9, 11, 14, 15, 16, 17]
Nash2 = [2, 3, 7, 9, 11, 14, 15, 16, 17]
for i in range(len(colorList1)):
    if i in Nash1:
        colorList1[i] = 'green'
        
for i in range(len(colorList2)):
    if i in Nash2:
        colorList2[i] = 'green'

#plot average strategies
plotStrat(df1, 'Player 1 RM (Neller) Strategy 2 {iterations}', colorList1)
plotStrat(df2, 'Player 2 RM (Neller) Strategy 2 {iterations}', colorList2)
#%%
dist = [0]*6
for i in range(6):
    for action,prob in zip(actions, rmStrats2P[1]):
        dist[i] += action.count(i)*prob/3

print(dist)
#%%
S = rmStrats2P[1]
s1 = S[2] + S[3] + S[11] + S[14] + S[15] + S[17]
s2 = S[7] + S[9] + S[16]
print(s1, s2)
print(s1+s2)
#%%
"TEST RM_CURRENT (SYMMETRIC)"
#intialise
s = [5,5]
G = generateGame(s, 3, [1, 1, 1])
s, k, a, val = G
iterations = 2000
numberGames = 200
# iterations = 20000
# numberGames = 1

#%%
#x, data = train2PlayerCurrent(G, 2000)

#run 200 tests
regret1, regret2 = [], []
z1, z2 = [], []
strat1, strat2 = np.array([0.0]*a[0]), np.array([0.0]*a[1])
for i in range(numberGames):
    x, data = train2PlayerCurrent(G, iterations)
    strat1 += np.array(x[0])
    strat2 += np.array(x[1])
    regret1.append(data[2])
    regret2.append(data[3])
    z1.append(data[6])
    z2.append(data[7])
    
    print(f'iteration {i} complete')
    
#save data
with open('RMz1 {iterations} its {games} games.pkl', 'wb') as f:
    pkl.dump(z1, f) 
    
with open('RMz2 {iterations} its {games} games.pkl', 'wb') as f:
    pkl.dump(z2, f) 
    
with open('Rmregret1 {iterations} its {games} games.pkl', 'wb') as f:
    pkl.dump(regret1, f) 

with open('RMregret2 {iterations} its {games} games.pkl', 'wb') as f:
    pkl.dump(regret2, f) 
    
with open('RMstrat1 {iterations} its {games} games.npy', 'wb') as f:
    np.save(f, strat1)   

with open('RMstrat2 {iterations} its {games} games.npy', 'wb') as f:
    np.save(f, strat2)


#find average regret
avgRegret1 = sum([np.array(r) for r in regret1])/numberGames
avgRegret2 = sum([np.array(r) for r in regret2])/numberGames
#%%

#plot regret of player 1
plt.plot(range(iterations), avgRegret1)
plt.xlabel('Iterations')
plt.ylabel('Average Regret (Player 1)')
plt.savefig(f'Raw Regret RM current Player 1 {iterations} its {numberGames} games')
plt.show()
#%%
#speed of convergence of player 1
curveFitPlotter(np.array(range(iterations))[5:]+1, avgRegret1[5:], '1', f'RM Current Convergence Plot {iterations} its {numberGames} games')
#%%
curveFitPlotter(np.array(range(iterations))+1, np.array(regret1[0]), '1', f'RM Current Convergence Plot {iterations} its Single Game')
#%%
#LOAD DATA
# with open('RMz1.pkl', 'rb') as f:
#     z1 = pkl.load(f) 
with open('RMz2.pkl', 'rb') as f:
    z2 = pkl.load(f) 
    

#%%
#set colours
colorList1 = ['red']*a[0]
colorList2 = ['red']*a[1]

#list of Nash equlibrium strats indicies (found using Hart method)
#Note: this Nash list specifically applies for the game 
#G = generateGame(s, 3, [1, 1, 1])
#for a different game, find the other NE manually and change this list

Nash1 = [2, 3, 7, 9, 11, 14, 15, 16, 17]
Nash2 = [2, 3, 7, 9, 11, 14, 15, 16, 17]
for i in range(len(colorList1)):
    if i in Nash1:
        colorList1[i] = 'green'
        
for i in range(len(colorList2)):
    if i in Nash2:
        colorList2[i] = 'green'
#%%
#find average strategies using Z
#THIS IS THE ZT IN THE PAPER, AND THE THEOREM SAYS THIS SET DISTRIBUTION WILL EVENTUALLY
#CONERGE TO THE CE
z1avg = sum([np.array(z)/iterations for z in z1])/numberGames
df1 = stratTodf(s[0], z1avg.tolist(), k)
plotStrat(df1, f'Player 1 RM (current) Z Strategy {iterations} its {numberGames} games', colorList1)
#%%
z2avg = sum([np.array(z)/iterations for z in z2])/numberGames
df2 = stratTodf(s[1], z2avg.tolist(), k)
plotStrat(df2, f'Player 2 RM (current) Z Strategy {iterations} its {numberGames} games', colorList2)

#find average final strategy for each player

# df1 = stratTodf(s[0], (strat1/numberGames).tolist(), k)
# df2 = stratTodf(s[1], (strat2/numberGames).tolist(), k)


# #plot average strategies
# plotStrat(df1, f'Player 1 RM (current) Avg FINAL Strategy {iterations} its {numberGames} games', colorList1)
# plotStrat(df2, f'Player 2 RM (current) Avg FINAL Strategy {iterations} its {numberGames} games', colorList2)
#%%
actions = generateActions(s[1],k)
#%%

dist = [0]*6
for i in range(6):
    for action,prob in zip(actions, z2avg):
        dist[i] += action.count(i)*prob/3

print(dist)

#%%
#curveFitPlotter(np.array(range(100)), data[2], '1')
#%%

# plt.plot(regret2[-2])
# #%%

# with open('strat2.pkl', 'rb') as f:
#     newfile = pkl.load(f)
# #%%
# with open('regret2.pkl', 'rb') as f:
#     mynewlist2 = pkl.load(f)
    
# #%%
# with open('strat2.pkl', 'rb') as f:
#     mynewlist3 = pkl.load(f)
    
# #%%

# df1 = stratTodf(s[0], (np.array(z1[2])/1000).tolist(), k)
# plotStrat(df1)

# #%%

# z1avg = sum([np.array(z)/1000 for z in z1])/500
# df = stratTodf(s[0], z1avg.tolist(), k)
# plotStrat(df)

# #%%
# z2avg = sum([np.array(z)/1000 for z in z2])/500
# df = stratTodf(s[0], z2avg.tolist(), k)
# plotStrat(df)
#%%
s1 = z2avg[2] + z2avg[3] + z2avg[11] + z2avg[14] + z2avg[15] + z2avg[17]
s2 = z2avg[7] + z2avg[9] + z2avg[16]
#%%
S = strat1/numberGames
s1 = S[2] + S[3] + S[11] + S[14] + S[15] + S[17]
s2 = S[7] + S[9] + S[16]


#%%
"Test Train Current using regret = average utility"