import lammps_python_interface as lpi
import os
import subprocess
import numpy as np
import ase,ase.io

from ase.units import kB

def read_headered_data(file_in : str):
    with open(file_in) as f:
        lines=f.readlines()
    labels=lines[0].split()
    mydict={}
    for i, val in enumerate(labels):
        mydict[val] = []
    for i, val in enumerate(lines[1:]):
        for j, val2 in enumerate(labels):
            mydict[val2].append(val.split()[j])
    return mydict

def write_params(TIP : lpi.ThermoIntParams, TS):
    with open(f'{TIP.toploc}/params.txt' , "w") as f:
        f.write('Paramters used: \n' +\
        f'pressure: {TIP.pressure}  \n ' +\
        f'TS temperature range: {TS.min()} - {TS.max()}  \n' +\
        f'TS steps: {len(TS)}  \n' +\
        f'number of steps: {TIP.nstep}  \n' +\
        f'thermostat: {TIP.thermostat}  \n' +\
        f'barostat: {TIP.barostat}  \n' +\
        f'averaging: {TIP.averaging}  \n' +\
        f'averaging setup: {TIP.averaging_setup}  \n' +\
        f'Number of atoms: {TIP.N}  \n' +\
        f'TS array: {TS}  \n'
        )
    return


HBAR=6.582119569e-16 ## eV per Hz
EV_TO_J=1.60218e-19
AA_TO_M=10e-10
AMU_TO_KG=1.66054e-27
BAR_TO_EV_PER_AA3=1/(160.21766208*10000)

def get_EC_FFE(k, m, T, N):
    k=k*EV_TO_J/(AA_TO_M**2) #eV/ \AA**2 to J
    m=m*AMU_TO_KG # g per mole to kg
    omega=np.sqrt(k/m) # Hz
    return 3*N*kB*T*np.log(omega*HBAR/(kB*T)) #eV


def get_COM_correction(k, N, V, T):
    return kB*T*np.log((N/V)*(2*np.pi*kB*T/(N*k))**(3/2)  )



def get_W_irr(file, thermoprint):
    RL=lpi.ReadLammps(file,block=1)
    Hamiltonian_potential = np.array(RL.dict['PotEng'], dtype=float) ##eV
    Hamiltonian_springs = -np.array(RL.dict['f_adiabat1'], dtype=float) #sign correction for potential energy
    Hamiltonian_springs += -np.array(RL.dict['f_adiabat2'], dtype=float)
    Hamiltonian_springs += -np.array(RL.dict['f_adiabat3'], dtype=float)
    Hamiltonian=Hamiltonian_potential+Hamiltonian_springs
    Lambda=np.array(RL.dict['f_adiabat1[1]'], dtype=float)
    fw = np.trapz(Hamiltonian, Lambda)
    RL=lpi.ReadLammps(file,block=3)
    Hamiltonian_potential = np.array(RL.dict['PotEng'], dtype=float) ##eV
    Hamiltonian_springs = -np.array(RL.dict['f_adiabat1'], dtype=float) #sign correction for potential energy
    Hamiltonian_springs += -np.array(RL.dict['f_adiabat2'], dtype=float)
    Hamiltonian_springs += -np.array(RL.dict['f_adiabat3'], dtype=float)
    Hamiltonian=Hamiltonian_potential+Hamiltonian_springs
    Lambda=np.array(RL.dict['f_adiabat1[1]'], dtype=float)
    bw = np.trapz(Hamiltonian, Lambda)
    w = 0.5*(fw - bw)
    return w

from scipy.integrate import cumtrapz
def get_w_RS(filein):
    ws=[]
    RL= lpi.ReadLammps(filein, block=1)
    flambda=np.array(RL.dict['v_lambda'], dtype=float)
    RL2= lpi.ReadLammps(filein, block=3)
    blambda=np.array(RL2.dict['v_lambda'], dtype=float)
    fdx=np.array(RL.dict['Enthalpy'], dtype=float)/flambda ###omg this solved it
    bdx=np.array(RL2.dict['Enthalpy'], dtype=float)/blambda
    wf = cumtrapz(fdx, flambda,initial=0)
    wb = cumtrapz(bdx[::-1], blambda[::-1],initial=0)
    w = (wf + wb)/2
    return w, flambda





def get_free_energy(TIP: lpi.ThermoIntParams, fl_file='thermoint.fl'):
    ###set up and run setup.
    TIP.setup_crystal()
    subprocess.run(f"{TIP.LAMMPS_RUN_COMMAND} < {TIP.location}/in.setup_crystal > {TIP.location}/setup_crystal.out", shell=True)
    setup_crystal = lpi.ReadLammps(f'{TIP.location}/setup_crystal.out')
    ###average dimensions
    Lx, Ly, Lz = setup_crystal.average_lx(TIP.averaging_setup), setup_crystal.average_ly(TIP.averaging_setup), setup_crystal.average_lz(TIP.averaging_setup)
    atomsforTI=TIP.atoms_unrattled #defaults to equilibrated otherwise.
    atomsforTI.set_cell([Lx, Ly, Lz], scale_atoms=True)
    ase.io.write(f'{TIP.location}/data.setup_crystal_end_meaned', atomsforTI, format='lammps-data')
    msd = setup_crystal.average_msd1(averaging=TIP.averaging_setup) ## \AA**2
    spring_const= [3*kB*TIP.temperature/msd] ## eV/ \AA**2
    msd = setup_crystal.average_msd2(averaging=TIP.averaging_setup) ## \AA**2
    spring_const.append(3*kB*TIP.temperature/msd) ## eV/ \AA**2
    msd = setup_crystal.average_msd3(averaging=TIP.averaging_setup) ## \AA**2
    spring_const.append(3*kB*TIP.temperature/msd) ## eV/ \AA**2
    TIP.setup_thermoint(spring_const)
    subprocess.run(f"{TIP.LAMMPS_RUN_COMMAND} < {TIP.location}/in.thermoint > {TIP.location}/thermoint.out", shell=True)
    thermoint_file=f"{TIP.location}/thermoint.out"
    W_irr = get_W_irr(thermoint_file, TIP.thermoprint_switch)
    EC_ffe = get_EC_FFE(k=spring_const[0], m=TIP.mass[0], T=TIP.temperature, N=TIP.N[0])
    EC_ffe += get_EC_FFE(k=spring_const[1], m=TIP.mass[1], T=TIP.temperature, N=TIP.N[1])
    EC_ffe += get_EC_FFE(k=spring_const[2], m=TIP.mass[2], T=TIP.temperature, N=TIP.N[2])
    RL=lpi.ReadLammps(thermoint_file,block=0) #read first NVT of run.
    COM_correction = get_COM_correction(spring_const[0], TIP.N[0], RL.average_vol(TIP.averaging_setup), TIP.temperature)
    COM_correction += get_COM_correction(spring_const[1], TIP.N[1], RL.average_vol(TIP.averaging_setup), TIP.temperature)
    COM_correction += get_COM_correction(spring_const[2], TIP.N[2], RL.average_vol(TIP.averaging_setup), TIP.temperature)
    F0 = EC_ffe+ W_irr + COM_correction # - TIP.isolated_energy
    PV = BAR_TO_EV_PER_AA3*TIP.pressure*RL.average_vol(TIP.averaging)
    G0 = F0 + PV
    G0b = G0/sum(TIP.N) - TIP.bulk_energy/sum(TIP.N)
    if not(os.path.exists(fl_file)):
        with open(fl_file , "w") as f:
            f.write(r'Pressure(NPT)  Pressure(NVT)  Temperature  G0/atom     F0/atom     W_irr/atom   spring0   spring1   spring2   COM_correct/atom   G0b/atom' +'\n')
    with open(fl_file , "a") as f:
        f.write(f"{TIP.pressure:13f}  {RL.average_pressure(TIP.averaging):13f}  {TIP.temperature:11f}  {G0/sum(TIP.N):10f}  {F0/sum(TIP.N):10f}  {W_irr/sum(TIP.N):10f}   {spring_const[0]:6f}   {spring_const[1]:6f}   {spring_const[2]:6f}   {COM_correction/sum(TIP.N):14f}   {G0b:14f} \n")
    return G0, F0, W_irr



def get_RS_path(TIP: lpi.ThermoIntParams, finaltemp, fl_file='thermoint.fl', rs_file='thermoint.rs'):
    setup_crystal = lpi.ReadLammps(f'{TIP.location}/setup_crystal.out')
    lambdaf=TIP.temperature/finaltemp
    TIP.setup_RS(lambdaf=lambdaf)
    subprocess.run(f"{TIP.LAMMPS_RUN_COMMAND} < {TIP.location}/in.RS > {TIP.location}/RS.out", shell=True)
    FL_path = read_headered_data(fl_file)
    G0=np.array(FL_path['G0/atom'], dtype=float) #per atom
    T0=TIP.temperature
    N=sum(TIP.N)
    w, Lambda_array=get_w_RS(f'{TIP.location}/RS.out')
    G=G0/Lambda_array  + (1/Lambda_array)*w/N + 1.5*kB*T0*np.log(Lambda_array)/Lambda_array
    Gi=G0/Lambda_array  + (1/Lambda_array)*w/N + 1.5*kB*T0*np.log(Lambda_array)/Lambda_array - TIP.isolated_energy/sum(TIP.N)
    Gb=G0/Lambda_array  + (1/Lambda_array)*w/N + 1.5*kB*T0*np.log(Lambda_array)/Lambda_array - TIP.bulk_energy/sum(TIP.N)
    T=T0/Lambda_array
    A = np.array([T, G, Gi, Gb]).T
    np.savetxt(rs_file, A, delimiter='\t', header=r"T   G0/atom   Gi0/atom   Gb0/atom", comments="")
    return



