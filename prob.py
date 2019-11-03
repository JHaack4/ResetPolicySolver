# todo: add probabilities to the report


# coding: utf-8

# In[42]:
import sys
csvFileName = sys.argv[1] #"""input.xlsx"""

# PARAMS
#goal_time = 98 * 60 
#reset_cost = 1 # seconds

# useNormal = False
# maxStdevUse = 3
# noiseAddedStdevMult = 0.5
###


# In[64]:


import numpy as np
import scipy.stats as stats
import pandas as pd
# import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from collections import Counter

def intToTime(a):
    a = int(a)
    s = a % 60
    m = (a // 60) % 60
    h = (a // 3600)
    st = ""
    if h > 0: 
        st = str(h) + ":"
    if m < 10:
        st += "0"
    st += str(m) + ":"
    if s < 10:
        st += "0"
    st += str(s)
    return st

def timeToInt(s):
    a = 0
    m = 1
    ts = s.split('.')
    tss = ts[0].split(':')
    for i in range(len(tss)-1, -1, -1):
        a += m * int(tss[i])
        m = m * 60
    return a

def splitTimesToProbDict(splitTimes, stdevs, stdNoiseMult=0.5, maxStd=3):
    ret = []
    for i in range(len(splitTimes)):
        arr = splitTimes[i]
        d = {}
        modStd = stdevs[i] * stdNoiseMult
        #print(modStd)
        
        for tm in arr:
            ms = range(int(round(max(0, tm-maxStd*modStd))),  1+int(round(tm+maxStd*modStd)) )
            probs = [stats.norm.pdf(m,tm,modStd) for m in ms] if len(ms) > 1 else [1]
            sum_probs = sum(probs)
            probs = [p/sum_probs for p in probs]
            
            for m,p in zip(ms,probs):
                if m in d:
                    d[m] += p / len(arr)
                else: d[m] = p / len(arr)
        
        ret.append(d)
    return ret

def splitTimesToProbDict_old(splitTimes):
    ret = []
    for i in splitTimes:
        c = Counter(i)
        d = {}
        s = len(i)
        for k in c.keys():
            d[k] = c[k] / s
        ret.append(d)
    return ret

def normalToProbDict(means, stdevs, maxStd=3):
    ret = []
    for i in range(len(means)):
        ms = range(int(round(max(0, means[i]-maxStd*stdevs[i]))),  1+int(round(means[i]+maxStd*stdevs[i])) )
        probs = [stats.norm.pdf(m,means[i],stdevs[i]) for m in ms] if len(ms) > 1 else [1]
        sum_probs = sum(probs)
        probs = [p/sum_probs for p in probs]
        d = {}
        for j in range(len(ms)):
            d[ms[j]] = probs[j]
        ret.append(d)
    return ret


# In[62]:


def is_blank(s):
    if s == '': return True
    if s == 'nan': return True
    if s == float('nan'): return True
    if isinstance(s, float) and np.isnan(s): return True
    return False

df = pd.read_excel(csvFileName, header=None, dtype=str)
assert(df[0][0] == 'SplitName')
assert(df[0][1] == 'ResetRate')
assert(df[0][2] == 'KeepLast')
assert(df[0][3] == 'TimeOffset')
assert(df[0][6] == 'Splits')
assert(df[0][4] == 'GoalTime')
assert(df[0][5] == 'ResetTime')
assert(df[2][4] == 'MaxStd')
assert(df[2][5] == 'NoiseStd')
df_rows, df_cols = df.shape
print(df)

splitNames = [df[i][0] for i in range(1,df_cols)]
resetProbs = [(lambda x: x if x < 1 else x/100)(float(df[i][1])) for i in range(1,df_cols)]
keepLast = [10 ** 10 if is_blank(df[i][2]) else int(df[i][2]) for i in range(1,df_cols)]
offsets = [0 if is_blank(df[i][3]) else int(df[i][3]) for i in range(1,df_cols)]
goal_time = timeToInt(df[1][4]) if ":" in df[1][4] else int(float(df[1][4]))
reset_cost = timeToInt(df[1][5]) if ":" in df[1][5] else int(float(df[1][5]))
useNormal = 'n' in df[3][4]
maxStdevUse = float(df[3][4][1:]) if 'n' in df[3][4] else float(df[3][4])
noiseAddedStdevMult = float(df[3][5])
splitTimes = []
print("split names: " + str(splitNames))
print("reset probs: " + str(resetProbs))
print("keep lasts: " + str(keepLast))
print("offsets: " + str(offsets))
print("goal time: " + intToTime(goal_time))
print("reset cost: " + intToTime(reset_cost))
print("maxStd: %.2f, noiseStd %.2f, normal=%s" % (maxStdevUse, noiseAddedStdevMult, str(useNormal)))

for j in range(1,df_cols):
    arr = []
    for i in range(6,df_rows):
        st = df[j][i]
        if is_blank(st):
            pass
        elif ":" in str(st):
            arr.append(timeToInt(str(st)))
        elif "x" in st:
            num_to_add = int(st[1:])
            for _ in range(num_to_add):
                arr.append(arr[-1])
        else:
            arr.append(int(float(st)))

    if keepLast[j-1] < len(arr):
        arr = arr[-1:-keepLast[j-1]-1:-1]
    if offsets[j-1] != 0:
        arr = [i+offsets[j-1] for i in arr]
    
    splitTimes.append(arr)
    
#print(splitTimes)  


# In[65]:


# splitNames = []
# resetProbs = []
# splitTimes = []

# with open(csvFileName,'r') as f:
#     i = -2
#     for line in f:
#         i += 1
#         if i==-1: continue
#         s = line.split(',')
#         splitNames.append(s[0])
#         rprob = (float(s[1]) if s[1] != '' else 0)
#         if rprob > 1: rprob = rprob / 100
#         resetProbs.append(rprob) 
#         keep = int(s[2]) if s[2] != '' and s[2] != 'x' else 1000000
#         found = 0
#         arr = []
#         if s[2] == 'x':
#             for j in range(3, len(s), 2):
#                 if s[j] == '': break
#                 t = timeToInt(s[j]) if (":" in s[j]) else int(s[j])
#                 cnt = int(s[j+1])
#                 for jj in range(cnt):
#                     arr.append(t)
#         else:
#             for j in range(len(s)-1,2,-1):
#                 if s[j] != '' and s[j] != '\n':
#                     stx = s[j].replace("\n","")
#                     arr.append(timeToInt(stx) if ":" in stx else int(stx))
#                     found += 1
#                     if found >= keep:
#                         break
#         splitTimes.append(arr)

# print(splitNames)
# print(resetProbs)
# #print([[intToTime(a) for a in i] for i in splitTimes])
            
# for i in range(len(splitNames)):
#     for j in range(len(splitTimes[i])):
#         plt.scatter(i,splitTimes[i][j])
# locs, labels = plt.xticks()
# plt.xticks(locs, [""] + splitNames)
# plt.show()

# splitNames = ["a", "b", "c", "d"]
# splitTimes = [[3,4,5,6,7,4,3,4], [3,3,2,1,5,3,2,3,5], [2,4,3,4,4,4,2,3,6], [6,6,5,4,5,6,7,7,10]]
# resetProbs = [0.5,0.5,0.5,0.5]

num_splits = len(splitNames)
max_time = sum([max(i) for i in splitTimes]) + 1
means = [np.average(i) for i in splitTimes]
stdevs = [np.std(i) for i in splitTimes]
bests = [min(i) for i in splitTimes]
worsts = [max(i) for i in splitTimes]
ranges = [max(i) - min(i) for i in splitTimes]

splitTimesProbDict = splitTimesToProbDict(splitTimes, stdevs, noiseAddedStdevMult, maxStdevUse)
splitTimesProbDictNormal = normalToProbDict(means, stdevs, maxStdevUse)
splitTimesDict = splitTimesProbDictNormal if useNormal else splitTimesProbDict
print("RAW SPLIT TIMES")
#print(splitTimes)
for i in range(num_splits):
    print("   " + splitNames[i], end=":: ")
    for j in splitTimes[i]:
        print("%d, " % j, end="")
    print("")
print("USED SPLIT PROBABILITIES")
#print(splitTimesDict)
for i in range(num_splits):
    print("   " + splitNames[i], end=":: ")
    for j in sorted(splitTimesDict[i]):
        print("%d:%.3f " % (j,splitTimesDict[i][j]), end="")
    print("")

info = []
print ("SORTED BY SPLIT")
for i in range(len(splitNames)):
    info.append((splitNames[i],means[i],stdevs[i],resetProbs[i],bests[i],ranges[i]))
    print("   %35s:\t mean=%.2f\tstdev=%.2f\tresetRate=%.2f\tgold=%d\trange=%d" % info[i])
    
print("SORTED BY MEAN")
info = sorted(info, key = lambda x: -x[1])
for i in range(len(info)): 
    print("   %35s:\t mean=%.2f\tstdev=%.2f\tresetRate=%.2f\tgold=%d\trange=%d" % info[i])
    
print("SORTED BY STDEV")
info = sorted(info, key = lambda x: -x[2])
for i in range(len(info)): 
    print("   %35s:\t mean=%.2f\tstdev=%.2f\tresetRate=%.2f\tgold=%d\trange=%d" % info[i])

print("SORTED BY RESET RATE")
info = sorted(info, key = lambda x: -x[3])
for i in range(len(info)): 
    print("   %35s:\t mean=%.2f\tstdev=%.2f\tresetRate=%.2f\tgold=%d\trange=%d" % info[i])
    
print("SORTED BY GOLD")
info = sorted(info, key = lambda x: -x[4])
for i in range(len(info)): 
    print("   %35s:\t mean=%.2f\tstdev=%.2f\tresetRate=%.2f\tgold=%d\trange=%d" % info[i])
    
print("SORTED BY RANGE")
info = sorted(info, key = lambda x: -x[5])
for i in range(len(info)): 
    print("   %35s:\t mean=%.2f\tstdev=%.2f\tresetRate=%.2f\tgold=%d\trange=%d" % info[i])
    


# In[117]:


def monteCarloProb(startSplit=0, startTime=0, useNormal=False, numSims = 10000):
    times = []
    splitLists = [list(i.keys()) for i in splitTimesDict]
    splitTimesUse = [i for i in splitLists]
    splitProbsUse = [[splitTimesDict[i][j] for j in splitLists[i]] for i in range(len(splitTimesDict))]
    
    for i in range(numSims):
        curTime = startTime
        curSplit = startSplit
        
        while True:
            
            #print(splitTimesUse[curSplit])
            #print(splitProbsUse[curSplit])
            #print(np.random.choice(splitTimesUse[curSplit], 1, splitProbsUse[curSplit]))
            curTime += np.random.choice(splitTimesUse[curSplit], 1, p=splitProbsUse[curSplit])[0]
#             if useNormal:
#                 curTime += round(np.random.normal(means[curSplit], stdevs[curSplit]))
#             else:
#                 curTime += splitTimes[curSplit][np.random.randint(len(splitTimes[curSplit]))]
            curSplit += 1
        
            if curSplit >= len(splitNames):
                times.append(curTime)
                break
    
    # round the times to nearest second...
    times = [round(i) for i in times]
    return times
    


def monteCarloET(resetPolicy, goal_time, reset_cost = 1, numSims=1000, useNormal=False):
    times_needed_to_PB = []
    
    splitLists = [list(i.keys()) for i in splitTimesDict]
    splitTimesUse = [i for i in splitLists]
    splitProbsUse = [[splitTimesDict[i][j] for j in splitLists[i]] for i in range(len(splitTimesDict))]
#     print(splitTimesUse)
#     print(splitProbsUse)
    
    for i in range(numSims):
        if (i % (numSims/100) == 0):
            print('.',end="") 
        cum_time = 0
        curTime = 0
        curSplit = 0
        
        while True:
            
#             if useNormal:
#                 curTime += round(np.random.normal(means[curSplit], stdevs[curSplit]))
#             else:
#                 curTime += splitTimes[curSplit][np.random.randint(len(splitTimes[curSplit]))]
            randr = np.random.choice(splitTimesUse[curSplit], 1, p=splitProbsUse[curSplit])[0]
            curTime += randr
            randReset = np.random.random() < resetProbs[curSplit]
            curSplit += 1
            #print(randr)
        
            if curSplit >= len(splitNames) and curTime < goal_time and not randReset: # goal met
                cum_time += curTime
                times_needed_to_PB.append(cum_time)
                break
            elif curSplit >= len(splitNames) or resetPolicy[curSplit] <= curTime or randReset: # reset
                cum_time += curTime + reset_cost
                curTime = 0
                curSplit = 0
                
            if cum_time > 30 * 24 * 60 * 60: # give up lol
                cum_time += curTime
                times_needed_to_PB.append(cum_time)
                break
    
    return times_needed_to_PB


# In[116]:


# this cell doesn't need to be ran...

# splitLists = [list(i.keys()) for i in splitTimesDict]
# splitTimesUse = [i for i in splitLists]
# splitProbsUse = [[splitTimesDict[i][j] for j in splitLists[i]] for i in range(len(splitTimesDict))]
# curSplit = 1
# for i,j in zip(splitTimesUse[1],splitProbsUse[1]):
#     print("%d %.3f" % (i,j))
# randr = np.random.choice(splitTimesUse[curSplit], 1, splitProbsUse[curSplit])[0]

# t1 = monteCarloProb(0,0,False)
# plt.hist(t1, normed=True, bins=max(t1)-min(t1)+1)
# plt.ylabel('Probability');
# plt.show()
# if True:
#     t2 = monteCarloET([10000000] * len(splitNames), goal_time, reset_cost, numSims = 100)
#     plt.hist(t2, bins=30)
#     plt.xlabel('Time');
#     plt.show()
#     print("\naverage time to PB:")
#     print(intToTime(np.mean(t2)))


# In[118]:


def computeProbabilityTree(goalTime, splitTimesDict):
    
    # PB[i][m] stores the probability that you beat the goal time
    # if you time from splits 0...i-1 is m
    PB = np.zeros((num_splits+1, max_time))
    
    # E[i][m] is the Expected finishing time at split i, time m
    E = np.zeros((num_splits+1, max_time))
    
    for m in range(max_time):
        PB[num_splits][m] = 1 if m < goalTime else 0
        E[num_splits][m] = m
        
    for i in range(num_splits-1, -1, -1):
        for m in range(max_time):            
            for k in splitTimesDict[i].keys():
                if m+k < max_time:
                    PB[i][m] += splitTimesDict[i][k] * (1-resetProbs[i]) * PB[i+1][m+k]
                    E[i][m] += splitTimesDict[i][k] * E[i+1][m+k]
                else:
                    E[i][m] += splitTimesDict[i][k] * max_time
                    
    # P[i][m] is the probability of reaching the state of split i, time m
    P = np.zeros((num_splits+1, max_time))
    
    for m in range(max_time):
        P[0][m] = 1 if m == 0 else 0
    
    for i in range(0,num_splits):
        for m in range(max_time):
            for k in splitTimesDict[i].keys():
                if m+k < max_time:
                    P[i+1][m+k] += P[i][m] * splitTimesDict[i][k] * (1 - resetProbs[i])
        
    return (P, E, PB)


def computeResetPolicyHelper(init_expected_time_to_PB, goal_time, splitTimesDict, reset_cost = 1):
    
    # ET[i][m] stores the expected time needed to PB if your
    # current state is a run with time m entering split i.
    # R[i][m] is 1 if you should reset in that state
    ET = np.zeros((num_splits+1, max_time))
    R = np.zeros((num_splits+1, max_time))
    
    for m in range(max_time):
        ET[num_splits][m] = 0 if m < goal_time else init_expected_time_to_PB + reset_cost
        R[num_splits][m] = 0 if m < goal_time else 1
        
    for i in range(num_splits-1, -1, -1):
        for m in range(max_time):
            ET[i][m] += resetProbs[i] * (init_expected_time_to_PB + reset_cost + means[i]) # reset from mistake
            for k in splitTimesDict[i].keys():
                if m+k < max_time:
                    ET[i][m] += splitTimesDict[i][k] * (1 - resetProbs[i]) * (ET[i+1][m+k] + k)
                else:
                    ET[i][m] += splitTimesDict[i][k] * (1 - resetProbs[i]) * (init_expected_time_to_PB + reset_cost + k)
            if ET[i][m] >= init_expected_time_to_PB + reset_cost:
                R[i][m] = 1
                ET[i][m] = init_expected_time_to_PB + reset_cost
    
    return (ET, R)

def computeResetPolicy(goal_time, splitTimesDict, reset_cost = 1):
    u_bound = max_time / PB[0][0]
    l_bound = 0
    cur = 0
    
    while u_bound > l_bound + 0.001 * l_bound:
        cur = (u_bound + l_bound)/2
        print("%.1f %.1f %.1f" % (l_bound,cur,u_bound))
        ET_use, R_use = computeResetPolicyHelper(cur, goal_time, splitTimesDict, reset_cost)
        if ET_use[0][0] > cur:
            l_bound = cur
        else: u_bound = cur
            
    ET_use, R_use = computeResetPolicyHelper(int(cur+1), goal_time, splitTimesDict, reset_cost)
    resetPolicy = np.zeros(num_splits+1)
    for s in range(num_splits+1):
        for m in range(max_time):
            if R_use[s][m] == 1:
                resetPolicy[s] = m
                break
    return (ET_use, R_use, resetPolicy)

P,E,PB = computeProbabilityTree(goal_time, splitTimesDict) 
#print(P)
print("DEBUG INFO")
print(" probability to PB = %.8f" % PB[0][0])
print(" expected run time (no resets) = %s" % intToTime(E[0][0]))
if PB[0][0] == 0:
    print("probability of PB is too low. Not computing reset policy")
    sys.exit()
ET,R,resetPolicy = computeResetPolicy(goal_time, splitTimesDict, reset_cost)

# for mmm in range(10, 200, 1):
#     ET1, R1 = computeResetPolicyHelper(mmm, 15, splitTimesDict)
#     print("init=%-10d res=%-10.1f %s" % (mmm, ET1[0][0], "here" if mmm>ET1[0][0] else ""))
    
np.set_printoptions(precision=3, suppress=True, linewidth=120)
# print(PB)
# print(E)
# print(P)
# print(ET)
# print(R)

# PRETTY PRINT REPORT
prob_too_small = 0.0001
num_resets_to_show = 10
interval_between_show = 1 if goal_time < 30*60 else (5 if goal_time < 70*60 else 10)

print ("\n\nREPORT")
print(" " + str(splitNames))
print(" " + str([intToTime(a) for a in resetPolicy]))
print(" probability to PB = %.8f" % PB[0][0])
print(" expected run time (no resets) = %s" % intToTime(E[0][0]))
print(" expected time to PB = " + intToTime(ET[0][0]))
print(" goal_time = " + str(goal_time))
print(" P = probability of being in this state")
print(" E = expected time to finish this run (conditioned on no resets)")
print(" PB = probability of getting a time less than the goal time")
print(" ET = expected time needed to PB from this state")
print ("START")
for s in range(num_splits):
    num_resets = 0
    
    for m in range(0, max_time, interval_between_show):
        if P[s][m]*interval_between_show < prob_too_small or num_resets >= num_resets_to_show: continue 
        
        pMult = interval_between_show
        resetS = "continue" if R[s][m] == 0 else 'reset'
        num_resets += R[s][m]
        print("   time=%-7s P=%-5.3f E=%-7s PB=%-5.3f ET=%-9s %s" % (intToTime(m), 
                P[s][m]*pMult, intToTime(E[s][m]), PB[s][m], intToTime(ET[s][m]), resetS))

    print(" SPLIT: " + splitNames[s])
print("END")
print(" expected time to PB = " + intToTime(ET[0][0]))
print("POLICY (for goal %s)" % intToTime(goal_time))
print("BEFORE   SPLIT       AFTER")
for i in range(len(splitNames)):
    print("%-9s%-28s%-9s" % (intToTime(resetPolicy[i]), splitNames[i], intToTime(resetPolicy[i+1] if i < len(splitNames) else goal_time)))
print("DONE\n\n")
if ET[0][0]/60/60 < 200:
    print("\n verify monte carlo = " + intToTime(np.mean(monteCarloET(resetPolicy, goal_time, reset_cost, numSims=100, useNormal=useNormal))))

