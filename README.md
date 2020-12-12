# adaptive_game_control
ChEn 436 Process Control final project: adaptive game difficulty

The basic file structure here is as follows:

`sim.py` implements the basic gameplay simulation, as well as classification system and PID control system.

`main.py` will run a simulation with `skill` and `style` parameters set inside the file, and generate and save some plots about that simulation. (It calls `plt.show`, so depending on where you run the code, it will probably also show you the plots.)

`params.py` is used to run a large number of simulations, for the purpose of gathering statistics; it also uses those simulations to determine values for process gain $K_p$ and time constant $\tau_p$. (The generated values are stored in `params.txt` here, so you do not need to run this unless you want to recalculate $K_p$ and $\tau_p$. The simulation results are all stored in `sim_results` folder.

`trends_gen.py` generates the reference file which our classification system uses. (Again, we have provided this output, `trendslist.data`, which is a pickled Python list, so you do not need to run this file.) It uses saved output files from `params.py`, so if you choose to run this, you should run `params.py` first.


