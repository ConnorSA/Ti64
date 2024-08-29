from lammps_input_writer import *
from calculate_mu import *
import ase, ase.io, ase.build
import os, sys, yaml
from warnings import simplefilter
import random
from ase.constraints import ExpCellFilter, FixAtoms
from ase.optimize import LBFGS
import pyjulip

simplefilter("ignore")

yamlfile=str(sys.argv[1])
with open(yamlfile, 'r') as f:
    yamlinput=yaml.safe_load(f)


### lmp should be the statically compiled version of lammps. In this example we attach a ACE potential.
# e.g. "mpirun -np 4 ./lmp_mpi" or "srun ./lmp_mpiicpc"
LAMMPS_RUN_COMMAND = f"srun ./lmp"


#run folder
folder=f"ACE_6x6x40_mdstep_20k_dt_0.0015_V_{yamlinput['V']}_Al_{yamlinput['Al']}_iterlearn00"
if not(os.path.exists(folder)):
    os.mkdir(folder)

#As defined in a LAMMPS input file: make sure to end each line with \n 
potential_str=("pair_style      hybrid/overlay pace table spline 6000 \n"
               "pair_coeff * *  pace ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg.yace Al Ti V \n" 
               "pair_coeff 1 1  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Al_Al \n"
               "pair_coeff 1 2  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Al_Ti \n"
               "pair_coeff 1 3  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Al_V \n"
               "pair_coeff 2 2  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Ti_Ti \n"
               "pair_coeff 2 3  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Ti_V \n"
               "pair_coeff 3 3  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table V_V \n"
               )


#building things
topdir=os.getcwd()
atoms=ase.io.read('bcc_Ti_geomopti.castep')
a = atoms.cell.cellpar()[0]*np.sqrt(4/3)
atoms = ase.build.bulk('Ti', crystalstructure='bcc', a=a, orthorhombic='True')
N_x=6
N_y=6
N_z=40
sc=[N_x,N_y,N_z]
atoms = atoms*sc


#initialise ti64
##P_V=0.036
##P_Al=0.102
P_V=yamlinput['V']
P_Al=yamlinput['Al']
N_V=int(np.round(len(atoms)*P_V))
N_Al=int(np.round(len(atoms)*P_Al))
symbols=atoms.get_chemical_symbols()
Ti_to_X=['Al' for i in range(N_Al)]
Ti_to_X+=['V' for i in range(N_V)]
symbols = ['Ti' for i in range(len(atoms))]
swap_atoms = random.sample(range(len(atoms)),len(Ti_to_X))
for j in range(len(Ti_to_X)):
    symbols[swap_atoms[j]] = Ti_to_X[j]
    atoms.set_chemical_symbols(symbols)
ace_potential="ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg.json"
ACE=pyjulip.ACE1(ace_potential)
atoms.calc=ACE
atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
ecf = ExpCellFilter(atoms, hydrostatic_strain=True)
dyn = LBFGS(ecf)
dyn.run(0.001);
atoms.rattle(0.05)

pressures=yamlinput['pressures']
start_temp=yamlinput['start_temp']

##starting temp init: 1800K

#initialise interface pinning parameters
IP=IPparams(toploc=folder,
            potential=potential_str,
            pressure=pressures[0],
            temperature=start_temp,
            step=20000,
            step_setup=10000,
            hkl=[N_x,N_y,0],
            thermostat=0.75,
            barostat=1.75,
            dt=0.0015,
	    Nz=N_z,
            thinned=100,
            atoms=atoms.copy(),
            LAMMPS_RUN_COMMAND=LAMMPS_RUN_COMMAND,
            auto=True)
IP.write_IP_params()


#run one IP routine.

#default is melt_steps=100, tol=10, samples=10
run_till_converged(IP, IP.traj_name, melt_steps=50, tol=10, samples=10)

#phase diagram via isobars
for i, p in enumerate(pressures[1:]):
    crystal_cc = ReadLammps(f'{IP.location}/crystal_auto.out')
    liquid_cc = ReadLammps(f'{IP.location}/liquid_auto.out')
    next_temp=classius_clapeyron_next_temp(current_T=IP.temperature, dP=p-pressures[i], crystal=crystal_cc, liquid=liquid_cc, thinned=IP.thinned)
    IP.next_isobar_start(pressure=p, temperature=next_temp)
    run_till_converged(IP, IP.traj_name, melt_steps=50, tol=10, samples=10)


