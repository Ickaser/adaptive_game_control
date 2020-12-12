import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy.random as rnd
import pickle
import sys
# %matplotlib inline

# from google.colab import drive
# drive.mount('/content/gdrive')


# Other files for this project
import sim 
# import params

# proj_dir = "/content/gdrive/My Drive/ProcControl Project/"
proj_dir = "./"

try:
    skill = int(sys.argv[1])
    if skill > 10:
        skill = 10
    elif skill < 1:
        skill = 1
except:
    raise ValueError("First command line argument should be skill, an int from 1-10.")
try:
    if sys.argv[2] == "c":
        style = "conservative"
    elif sys.argv[2] == "m":
        style = "moderate"
    elif sys.argv[2] == "r":
        style = "reckless"
    else:
        raise ValueError("Invalid style type in command argument.")
except:
    raise ValueError("Second command line argument should be style: c for conservative, m for moderate, r for reckless.")


Npts = 6000 # seconds
time = np.linspace(0,Npts-1,Npts)
dt = time[1]-time[0]

#Initialize Character Stats array
# At_char = np.ones(Npts)*10  #Character Attack
HP0  = 100
At_char = 1
# style = "reckless"
# style = "moderate"
HP_char = np.zeros(Npts) #Character Health
HP_char[0] = HP0


              
#Enemy Initial Stats
Enemy0 = [10,1] #Enemy0 = [Health, Attack]  #
Enemy = list(Enemy0)
At_enem = np.ones(Npts)
HP_enem = np.ones(Npts) * Enemy0[0]


#If we use average damage per time as our set point we will use the following
SP_DPS = HP0/100      #the PID will attempt to have the player loose this much health per time (damage/second)
SP_DPS = HP0/150

#########

tauD = 0

dt    = 1

# Run a simulation
HP_char, At_enem, HP_enem, etod, interr_arr = sim.sim_gameplay(style, skill, control=True)
filterwidth = 20
# DPS = -(HP_char[filterwidth:] - HP_char[:-filterwidth])/filterwidth
DPS = -(HP_char[1:] - HP_char[0]) / time[1:]
DPS[0] = 0
# print(At_enem[:etod])
err = SP_DPS - DPS
SAE = np.sum(np.abs(err[:etod]))
# print(err[:etod])
print("SAE", SAE)
print("etod", etod)

# Plot results up until time of death

plt.plot(time[0:etod],HP_char[0:etod],'k',label='HP')
plt.title('Character Health',fontsize=24)
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Health',fontsize=18)
plt.axvline(150, color="gray", ls="--", label="target time")
plt.text(etod*.8, 80, f"skill={skill}")
plt.text(etod*.7, 60, f"style={style}")
plt.legend(fontsize=14)
plt.ylim(0, 100)
plt.xlim(0, np.max([etod, 150]))
plt.savefig(proj_dir + f"health_skill{skill:d}_style{style[0]}.png");
plt.show()

plt.title("Damage per Second",fontsize=24)
plt.plot(time[0:etod], DPS[0:etod])
# plt.plot(time[0:etod], err[0:etod])
plt.text(0, 1.2, f"SAE={SAE}")
# plt.plot(time[0:etod], interr_arr[0:etod])
plt.plot([0,etod],[SP_DPS,SP_DPS])
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Damage per second',fontsize=18)
plt.ylim(0, 1.5)
plt.xlim(0, np.max([etod, 150]))
plt.axvline(150, color="gray", ls="--", label="")
plt.savefig(proj_dir + f"dps_skill{skill:d}_style{style[0]}.png");
plt.show()

plt.plot(time[0:etod],At_enem[0:etod],'k',label='Enemy attack')
plt.title('Enemy Attack',fontsize=24)
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Attack',fontsize=18)
plt.xlim(0, np.max([etod, 150]))
plt.savefig(proj_dir + f"eatk_skill{skill:d}_style{style[0]}.png");
plt.show();

plt.plot(time[0:etod],HP_enem[0:etod],'k',label='HP')
plt.title('Enemy Health',fontsize=24)
plt.xlabel('Time (sec)',fontsize=18)
plt.ylabel('Health',fontsize=18)
plt.show();


