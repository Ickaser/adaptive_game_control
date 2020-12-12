import numpy as np
import wheelpy.muc as muc
import matplotlib.pyplot as plt
# import tclab
# import time
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
import pickle
un = muc.uReg
un.setup_matplotlib()

proj_dir = "./data/"

HP0 = 100
NPts = 6000
time = np.arange(NPts+1)
dt = time[1]-time[0]
Nsim = 100

# Generate a summary plot across skill level, style types
col_list = ["blue", "green", "red"]
lab_list = ["conservative", "moderate", "reckless"]
full_list = []
sk_list = range(1, 11)
st_list = lab_list
for i, st in enumerate(st_list):
    sub_list = []
    for sk in sk_list:
        dat_aa = np.loadtxt(proj_dir+f"enem_results_skill{sk}_style{st[0]}.txt")
        sub_list.append(dat_aa)
    full_list.append(sub_list)
full_arr = np.array(full_list)
print(full_arr.shape)

tau_arr = []
gain_arr = []
trends_list = []
for i, st in enumerate(full_arr):
    # print(f)
    # plt.plot(st, "x", color=col_list[i])
    # print(st.shape)
    trend_sub1 = []
    for j, sk in enumerate(st):
        # print(len(sk))
#         tau_list = []
#         gain_list = []
        trend_sub2 = []
        
        for sim in sk.T:
            trend_sub3 = []
    
            hp_prev = 10
            life_list = [0]
            for k, hp in enumerate(sim):
                if hp == hp_prev:
                    continue
                elif hp == 0:
                    life_list.append(k+1)
                hp_prev = hp 
#             trend_sub2.append(life_list)
#             print(life_list)
            life_list = np.array(life_list)
#             plt.plot(life_list, 0*life_list, "o")
#             plt.plot(sim)
#             plt.xlim(0, 300)
#             plt.show()
#             pass
            for l, ll in enumerate(life_list[1:]-life_list[:-1]):
                if len(trend_sub2)<=l:
                    trend_sub2.append([ll])
                else:
                    trend_sub2[l].append(ll)
#         print(trend_sub2)
        sub2_mean = [np.mean(tr) for tr in trend_sub2]
        
            
        print(sk_list[j], st_list[i], len(trend_sub2), sub2_mean)
        trend_sub1.append(sub2_mean)
    trends_list.append(trend_sub1)

# print(trends_list)
# full_trend = np.array(trends_list)

# Style, skill, sim #
coord = (0, 7, 4)
sample = full_arr[1, 5, :, 9]

hp_prev = 10
life_list = [0]
for k, hp in enumerate(sample):
    if hp == hp_prev:
        continue
    elif hp == 0:
        life_list.append(k+1)
    hp_prev = hp 
life_list = np.array(life_list)
len_list = life_list[1:]-life_list[:-1]
print(len_list)


sample_ave = trends_list[2][4]
print(sample_ave)

def SSE(ti, kind):
    trend = trends_list[kind[0]][kind[1]]
    max_ti = np.min([len(len_list)-1, len(trend)-1])
    if ti >= max_ti:
        ti = max_ti
    x1 = len_list[:ti]
    x2 = trend[:ti]
    SSE = np.sum((x1-x2)**2)/len(x1)
    return SSE
    
def all_SSE(ti):
    coord_list = [[(i, j) for j in range(10)] for i in range(3)]
    coord_list = np.reshape(coord_list, (30,2))
    SSE_list = [SSE(ti, coord) for coord in coord_list]
    return SSE_list
def opt_kind(ti):
    SSE = all_SSE(ti)
    opt = np.where(SSE==np.min(SSE))[0][0]
    kind1 = int(opt % 10)
    kind0 = int((opt-kind1)/10)
    return kind0, kind1
    
print(opt_kind(10))

with open("trendslist.data", "wb") as filehandle:
    pickle.dump(trends_list, filehandle)
    
col_list = ["red", "green", "blue"]
for i, ts1 in enumerate(trends_list):
    for ts2 in ts1:
        plt.plot(ts2, color=col_list[i])
plt.xlim(-.5,5)
plt.ylim(10,30)