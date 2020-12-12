import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy.random as rnd
import pickle

Npts = 6000
HP0 = 100
Enemy0 = [10, 1]
At_char = 1
SP_DPS = HP0 / 150
proj_dir = "./"

# Get data analyzed previously
with open(proj_dir + 'trendslist.data', 'rb') as filehandle:
    trends_list = pickle.load(filehandle)


def Health(healths,enemy,style, skill):
    HP_char, HP_enem = healths
    At_enem = enemy[1]
    hit_roll = rnd.random()

    # P_hit
    P_hit_char = 1.1 - skill/10 # Odds of hitting the player
    P_hit_enem = 0.5 + skill/20 # Odds of hitting the enemy

    # Playstyle multipliers: higher means more damage is dealt
    if style=="conservative":
        # Multiplier ranges from .15 to 1
        mult = HP_char/HP0 if HP_char > 15 else .15
        # print("c", end="")
    elif style=="reckless":
        # Multiplier ranges from 1.2 to .8
        mult = (1.2 - .4*HP_char/HP0) # if HP_char > 50 else .5
    #     print("r", end="")
    elif style=="moderate":
        # Constant multiplier of .7
        mult = 0.7
    #     print("m", end="")
    else:
        raise ValueError("Bad style passed to Health function")
    P_hit_char *= mult
    P_hit_enem *= mult

    dHdt_char = -At_enem if hit_roll <= P_hit_char else 0
    dHdt_enem = -At_char if hit_roll <= P_hit_enem else 0
    # print(f"P(HC) {hit_char:.2f}, P(HE) {hit_enemy:.2f}, Phit {P_hit:.2f}, dHCdt {dHdt_char:.0f}, dHEdt {dHdt_enemy:.0f}")
    return dHdt_char, dHdt_enem




# Classification Mechanism
def classify(hp_enemy):
    hp_prev = 10
    temp_list = [0]
    for k, hp in enumerate(hp_enemy):
        if hp == hp_prev:
            continue
        elif hp == 0:
            temp_list.append(k+1)
        hp_prev = hp 
    temp_list = np.array(temp_list)
    life_list = temp_list[1:]-temp_list[:-1]
    li = len(life_list)
    kind = opt_kind(life_list, li)
    return kind

# Helper functions for classification
def SSE(life_list, li, kind):
    trend = trends_list[kind[0]][kind[1]]
    max_li = np.min([len(life_list), len(trend)])
    if li >= max_li:
        li = max_li
    x1 = life_list[:li]
    x2 = trend[:li]
    SSE = np.sum((x1-x2)**2)/len(x1)
    return SSE
    
def all_SSE(life_list, li):
    coord_list = [[(i, j) for j in range(10)] for i in range(3)]
    coord_list = np.reshape(coord_list, (30,2))
    SSE_list = [SSE(life_list, li, coord) for coord in coord_list]
    return SSE_list
def opt_kind(life_list, li):
    SSE = all_SSE(life_list, li)
    opt = np.where(SSE==np.min(SSE))[0][0]
    kind1 = int(opt % 10)
    kind0 = int((opt-kind1)/10)
    return kind0, kind1

# PID setup -------------------

dat = np.loadtxt("params.txt")
tau_arr, gain_arr = dat.T
gain1 = gain_arr[0:10]   #conservative
gain2 = gain_arr[10:20]  #moderate
gain3 = gain_arr[20:30]  #reckless

tau1 = tau_arr[0:10]     #conservative
tau2 = tau_arr[10:20]    #moderate
tau3 = tau_arr[20:30]    #reckless

# finding tau i and kc
def calc_params(kind):
    kind0, kind1 = kind
    ind = 10*kind0 + kind1
    gain = gain_arr[ind]
    tau = tau_arr[ind]

    tauI = tau / 5
    Kc = 1/gain * 3
    return Kc, tauI

#kc = np.array([Kc1,Kc2,Kc3])
#tauI = np.array([tau_I1,tau_I2, tau_I3])
##########

#parameters and arrays for the PID controller
SPH   = np.zeros(Npts)
H     = np.zeros(Npts)
error = np.zeros(Npts)
INTerr= np.zeros(Npts)
dHdt  = np.zeros(Npts)

tauD = 0

dt    = 1


def PID(i,SPH, H, H_last, Kc,tau_I, tau_D, INTerr_prev):  #where i is the current time
    # This function is written in terms of health, but works equivalently for a different PV.
    Abias = 1
    #P
    error = SPH-H
    sumierr = error*dt


    #ID 
    INTerr = sumierr+INTerr_prev
    dHdt = (H-H_last)/dt
    At_enem = Abias+Kc*error+(Kc/tau_I)*INTerr-Kc*tau_D*dHdt


    #limiting the range of the enemy attack
    if At_enem<0:
        At_enem = 0
        INTerr -= sumierr
    elif At_enem>20:
        At_enem = 20
        INTerr -= sumierr
      # print(At_enem,INTerr - INTerr_prev)

    return At_enem, INTerr


# Model Gameplay Simulation
def sim_gameplay(style, skill, adj=True, control=True):
    # set up arrays, values
    HP_char = np.zeros(Npts) #Character Health
    HP_char[0] = HP0
    interr_arr = np.array(HP_char)
  
    #Enemy Initial Stats
    Enemy = list(Enemy0)
    At_enem = np.ones(Npts)
    HP_enem = np.ones(Npts) * Enemy0[0]
    for i in range(Npts):
        # skip first time step
        if i == 0:
            #initialize enemy counter
            count_enem = 1
            # initialize DPS value
            DPS_last = 1
            interr = 0
            continue
    
        #Check to see if already dead
        chardead = HP_char[i-1] <= 0
        if chardead:
            etod = int(i * dt) #Record Estimated time of death
            break
    
        #PID logic  ***unfinished***
        enemdead = Enemy[0] <= 0
        if enemdead:
            count_enem += 1
            Enemy = list(Enemy0) 
            Enemy[1] += rnd.random()-.5 # + PID
            if control:
                # Decide which kind of player
                kind = classify(HP_enem[:i])
                # print(kind)
                # Determine parameters based on player kind
                Kc, tauI = calc_params(kind)
                # PID control with parameters
                DPS_now = (HP0 - HP_char[i-1])/i
                new_At, interr = PID(i, SP_DPS, DPS_now, DPS_last, Kc, tauI, tauD, interr)
                DPS_last = DPS_now
                # print("cntrl", end="")
                Enemy[1] += new_At
            elif adj and HP_char[i-1] <= 50:
                Enemy[1] += 1
  
        interr_arr[i] = interr
    
        # Discrete method
        HPc = HP_char[i-1]
        HPe = Enemy[0]
        step = Health([HPc, HPe], Enemy, style, skill)
        
        #Record new values
        HP_char[i] = HP_char[i-1] + step[0]*dt
        Enemy[0] += step[1]*dt
        At_enem[i] = Enemy[1]
        HP_enem[i] = Enemy[0]
    
    
        #Truncate recorded values
        if HP_char[i] <= 0:
            HP_char[i] = 0
            
        if HP_enem[i] <= 0:
            HP_enem[i] = 0
    
        #Diagnostics
        #print(HPc, HPe)
    # print("style", style, "skill", skill, "etod", etod)
    return HP_char, At_enem, HP_enem, etod if HP_char[-1] == 0 else None, interr_arr
