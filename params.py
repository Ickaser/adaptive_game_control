
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy.random as rnd
import pickle

import sim 

HP0 = 100
Npts = 6000
time = np.arange(Npts+1)
Enemy0 = [10, 1]
At_char = 1
proj_dir = "./"


# Run a set of simulations, info for each, save to file
N_sim = 100
sk_list = range(1, 11)
st_list = ["conservative", "moderate", "reckless"]
for skill in sk_list:
    for style in st_list:
        # etod_list = []
        # for n in range(N_sim):
        #     HP_char, At_enem, HP_enem, etod = sim.sim_gameplay(style, skill)
        #     plt.plot(time[0:etod],HP_char[0:etod],'k',label='HP')
        #     etod_list.append(etod)
        # etod_arr = np.array(etod_list)
        HP_arr_arr = np.array([sim.sim_gameplay(style, skill, adj=True, control=False)[0] for n in range(N_sim)])
        HP_arr = np.mean(HP_arr_arr, axis=0)
        enem_arr_arr = np.array([sim.sim_gameplay(style, skill, adj=True, control=False)[2] for n in range(N_sim)])
        etod_arr = np.array([sim.sim_gameplay(style, skill, control=False)[3] for n in range(N_sim)])

        np.savetxt(proj_dir+"sim_results/"+f"sim_results_skill{skill}_style{style[0]}_mod.txt", HP_arr.T, fmt="%d")
        np.savetxt(proj_dir+"sim_results/"+f"sim_results_skill{skill}_style{style[0]}.txt", etod_arr.T, fmt="%d")
        np.savetxt(proj_dir+"sim_results/"+f"HP_results_skill{skill}_style{style[0]}.txt", HP_arr_arr.T, fmt="%d")
        np.savetxt(proj_dir+"sim_results/"+f"enem_results_skill{skill}_style{style[0]}.txt", enem_arr_arr.T, fmt="%d")

# Generate a summary plot across skill level, style types
col_list = ["blue", "green", "red"]
lab_list = ["conservative", "moderate", "reckless"]
full_list = []
for i, st in enumerate(st_list):
    sub_list = []
    for sk in sk_list:
        dat_aa = np.loadtxt(proj_dir+"sim_results/"+f"sim_results_skill{sk}_style{st[0]}.txt")
        sub_list.append(dat_aa)
    full_list.append(sub_list)
full_arr = np.array(full_list)
print(full_arr.shape)
ave_DPS = []
for i, st in enumerate(full_arr):
    # print(f)
    plt.plot(range(1, 11), st, "x", color=col_list[i])
    for sk in st.T:

        # plt.plot(range(1,11), sk, "x", color=col_list[i],)
        pass
    plt.plot(range(1, 11), np.mean(st, axis=1), "-", color=col_list[i], label=lab_list[i]+" playstyle")
    ave_DPS.append(100/np.mean(st, axis=1))
  
plt.legend()
plt.ylabel("Time steps before death")
plt.xlabel("Skill level")
plt.title("Comparison of game time vs. skill level and play style")

print(ave_DPS)

# Generate a summary plot across skill level, style types
col_list = ["blue", "green", "red"]
lab_list = ["conservative", "moderate", "reckless"]
full_list = []
for i, st in enumerate(st_list):
    sub_list = []
    for sk in sk_list:
        dat_aa = np.loadtxt(proj_dir+"sim_results/"+f"HP_results_skill{sk}_style{st[0]}.txt")
        sub_list.append(dat_aa)
    full_list.append(sub_list)
full_arr_step = np.array(full_list)
print(full_arr_step.shape)
tau_arr = []
gain_arr = []
for i, st in enumerate(full_arr_step):
    # print(f)
    # plt.plot(st, "x", color=col_list[i])
    # print(st.shape)
    for j, sk in enumerate(st):
        # print(len(sk))
        tau_list = []
        gain_list = []
        for sim in sk.T:
    
            etod = np.where(sim==0)[0][0]
            step = np.where(sim<=50)[0][0]
            qd = np.where(sim<=25)[0][0]
            tau_list.append(qd-step)
            DPS = (HP0 - sim[step:etod])/time[step:etod]
            alt_etod = full_arr[i, j]
            alt_DPS = 100/alt_etod
            gain = DPS[-1]-alt_DPS
            gain_list.append(gain)
            # plt.plot(time[:etod-step], DPS[:etod])
            # # plt.plot(range(1,11), sim, "x", color=col_list[i],)
            pass
        tau_arr.append(np.mean(tau_list))
        gain_arr.append(np.mean(gain_list))
        plt.title(f"skill {j+1}, style {lab_list[i][:3]}")
        # plt.show()
      # plt.plot(range(1, 11), np.mean(st, axis=1), "-", color=col_list[i], label=lab_list[i]+" playstyle")
tau_arr = np.array(tau_arr)
gain_arr = np.array(gain_arr)
print("tau", tau_arr)
print("gain", gain_arr)
dat = np.column_stack([tau_arr, gain_arr])
np.savetxt("params.txt", dat)
# plt.legend()
# plt.ylabel("DPS")
# plt.xlabel("Time after step change")
# plt.xlim(0, 100)
# plt.ylim()
# plt.title("Comparison of game time vs. skill level and play style")

gain1 = gain_arr[0:10]   #conservative
gain2 = gain_arr[10:20]  #moderate
gain3 = gain_arr[20:30]  #reckless
tau1 = tau_arr[0:10]     #conservative
tau2 = tau_arr[10:20]    #moderate
tau3 = tau_arr[20:30]    #reckless
plt.plot(sk_list, gain1, label=lab_list[0])
plt.plot(sk_list, gain2, label=lab_list[1])
plt.plot(sk_list, gain3, label=lab_list[2])
plt.legend()
plt.ylabel("Gain $K_p$")
plt.xlabel("Skill")

def line(x, m, b):
  return m*x+b
Kp1_fit, covar = curve_fit(line, sk_list, gain1)
Kp2_fit, covar = curve_fit(line, sk_list, gain2)
Kp3_fit, covar = curve_fit(line, sk_list, gain3)
# plt.plot(sk_list, line(sk_list, *Kp1_fit))
# plt.plot(sk_list, line(sk_list, *Kp2_fit))
# plt.plot(sk_list, line(sk_list, *Kp3_fit))

plt.show()
print(Kp1_fit)
print(Kp2_fit)
print(Kp3_fit)

def exp_c(x, c1, c2, offset):
  return c1*np.exp(x*c2) + offset
tau1_fit, covar = curve_fit(exp_c, sk_list, tau1)
tau2_fit, covar = curve_fit(exp_c, sk_list, tau2)
tau3_fit, covar = curve_fit(exp_c, sk_list, tau3)

print(tau1_fit)
print(tau2_fit)
print(tau3_fit)



plt.plot(sk_list, tau1, label=lab_list[0])
plt.plot(sk_list, tau2, label=lab_list[1])
plt.plot(sk_list, tau3, label=lab_list[2])
# plt.plot(sk_list, exp_c(sk_list, *tau1_fit))
# plt.plot(sk_list, exp_c(sk_list, *tau2_fit))
# plt.plot(sk_list, exp_c(sk_list, *tau3_fit))

plt.legend()
plt.ylabel("Time constant $\\tau_p$")
plt.xlabel("Skill")
plt.show()
