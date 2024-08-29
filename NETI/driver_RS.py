import ase, ase.build, ase.io
import numpy
from lammps_python_interface import ThermoIntParams
from free_energy_getter import get_free_energy, get_RS_path
import os, sys, yaml
import numpy as np
from ase.constraints import ExpCellFilter, FixAtoms
from ase.optimize import LBFGS
import pyjulip
import random
import time

starttime = time.time()

#lmp is the statically compiled version of lammps. We use a ACE potential in this example.
#LAMMPS_RUN_COMMAND = "mpirun -np 16 ./lmp_g++_openmpi"
LAMMPS_RUN_COMMAND = f"srun ./lmp"


yamlfile=str(sys.argv[1])
with open(yamlfile, 'r') as f:
    yamlinput=yaml.safe_load(f)





#hcp=ase.io.read('data.Ti_hcp_ortho', format='lammps-data', style='atomic') #N=4
#bcc=ase.io.read('data.Ti_bcc_ortho', format='lammps-data', style='atomic') #N=2
#hex=ase.io.read('data.Ti_hex_ortho', format='lammps-data', style='atomic') #N=6



atoms=ase.io.read(yamlinput['primitive_cell'], format='lammps-data', style='atomic')
atoms.set_chemical_symbols(['Ti' for a in atoms])
#atoms=ase.build.bulk('Ti', orthorhombic=True, crystalstructure='bcc', a=3.2)
sc=yamlinput['supercell']
atoms=atoms*sc


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
dyn.run(0.0005);
atoms_unrattled=atoms.copy()
atoms.rattle(0.02)


inittemp=yamlinput['start_temp']
finaltemp=yamlinput['final_temp']

Trange=np.linspace(inittemp,finaltemp,51)

potential_str=("pair_style      hybrid/overlay pace table spline 6000 \n"
               "pair_coeff * *  pace ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg.yace Al Ti V \n"
               "pair_coeff 1 1  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Al_Al \n"
               "pair_coeff 1 2  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Al_Ti \n"
               "pair_coeff 1 3  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Al_V \n"
               "pair_coeff 2 2  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Ti_Ti \n"
               "pair_coeff 2 3  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table Ti_V \n"
               "pair_coeff 3 3  table ti64_combined_data_Apr08_rcut_6.0_order_3_degree_15_custom_reg_pairpot.table V_V \n"
               )




#pair_style = "eam/fs"
#pair_coeff = "* * Ti1.eam.fs Ti"
name=yamlinput['name']


aniso=True
if yamlinput['iso']:
    anisostring='iso'
    aniso = False
else:
    anisostring='aniso'

folder=f"RS-{name}-{sc[0]}x{sc[1]}x{sc[2]}-{Trange[0]}-{Trange[-1]}-{anisostring}-Al-{P_Al}-V-{P_V}-{yamlinput['label']}"
if not(os.path.exists(folder)):
    os.mkdir(folder)

TIP=ThermoIntParams(
    LAMMPS_RUN_COMMAND=LAMMPS_RUN_COMMAND, 
    atoms=atoms,
    atoms_unrattled=atoms_unrattled, 
    toploc=folder, 
    potential=potential_str,
    mass=[26.98, 47.867, 50.94],
    pressure=float(yamlinput['pressure']), 
    temperature=Trange[0], 
    timestep=0.001, 
    nstep=25000,
    nstep_setup=10000,
    nstep_eq=10000,  
    thermostat=0.95, 
    barostat=2.00,
    averaging_setup=100,
    averaging=100,
    thermoprint=20,
    aniso=aniso,
    randoms=[random.randrange(10000),random.randrange(10000),random.randrange(10000),random.randrange(10000),random.randrange(10000),random.randrange(10000), random.randrange(10000)]
    )


get_free_energy(TIP, fl_file=f'thermoint_{folder}-P{TIP.pressure:.1f}.fl')
get_RS_path(TIP, finaltemp=finaltemp, fl_file=f'thermoint_{folder}-P{TIP.pressure:.1f}.fl', rs_file=f'thermoint_{folder}-P{TIP.pressure:.1f}.rs')


endtime = time.time()

print('time:',(endtime-starttime)/3600, 'hrs')
