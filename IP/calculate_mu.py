import numpy as np
import ase, ase.io
from ase.units import kB
from lammps_input_writer import *
import subprocess

EV_TO_BAR = 1602176 #1ev/A^3 = 1602176 bar
BAR_TO_EV=1/EV_TO_BAR


def delta_mu(pinning : ReadLammps, crystal  : ReadLammps, liquid  : ReadLammps, kappa : float, a : float, N : int, thinned=None):
    return -kappa*(pinning.average_Q(thinned) -a)*(crystal.average_Q(thinned) - liquid.average_Q(thinned))/N

def delta_mu_error(pinning : ReadLammps, crystal  : ReadLammps, liquid  : ReadLammps, kappa : float, a : float, N : int, thinned=None):
    Q_pin = np.array(pinning.dict['f_bias[3]'], dtype=float)[-thinned:]
    Q_cry = np.array(crystal.dict['f_bias[3]'], dtype=float)[-thinned:]
    Q_liq = np.array(liquid.dict['f_bias[3]'], dtype=float)[-thinned:]
    dmu_arr = -kappa*(Q_pin - a)*(Q_cry - Q_liq)/N
    return dmu_arr.std()

def energy_means_and_errors(current_P : float, current_T : float, pinning : ReadLammps, crystal  : ReadLammps, liquid  : ReadLammps, kappa : float, a : float, N : int, thinned=None):
    Q_pin = np.array(pinning.dict['f_bias[3]'], dtype=float)[-thinned:]
    Q_cry = np.array(crystal.dict['f_bias[3]'], dtype=float)[-thinned:]
    Q_liq = np.array(liquid.dict['f_bias[3]'], dtype=float)[-thinned:]
    dmu_arr = -kappa*(Q_pin - a)*(Q_cry - Q_liq)/N
    crystal_vol = np.array(crystal.dict['Volume'], dtype=float)[-thinned:]
    crystal_U = np.array(crystal.dict['TotEng'], dtype=float)[-thinned:]
    liquid_vol = np.array(liquid.dict['Volume'], dtype=float)[-thinned:]
    liquid_U = np.array(liquid.dict['TotEng'], dtype=float)[-thinned:]
    d_U = (crystal_U - liquid_U)/N
    d_V = (crystal_vol - liquid_vol)/N
    dmu_dT =(d_U + current_P*BAR_TO_EV*d_V - dmu_arr)/current_T
    return dmu_dT.mean(), dmu_dT.std(), (current_P*BAR_TO_EV*d_V).mean(), (current_P*BAR_TO_EV*d_V).std(), dmu_arr.mean(), dmu_arr.std(), d_U.mean(), d_U.std()

def isotherm_next_pressure(current_P : float, crystal : ReadLammps, liquid : ReadLammps, N : int, d_mu : float, thinned=None):
    d_V = (crystal.average_vol(thinned) - liquid.average_vol(thinned))/N
    return (current_P*BAR_TO_EV - d_mu/d_V)*EV_TO_BAR


def isobar_next_temp(current_P : float, current_T : float, crystal : ReadLammps, liquid : ReadLammps,
                     N : int, d_mu :float, thinned=None, step_prefactor=1):
    crystal_vol = crystal.average_vol(thinned)
    crystal_U = crystal.average_energy(thinned)
    liquid_vol = liquid.average_vol(thinned)
    liquid_U = liquid.average_energy(thinned)
    d_U = (crystal_U - liquid_U)/N
    d_V = (crystal_vol - liquid_vol)/N
    dmu_dT =(d_U + current_P*BAR_TO_EV*d_V -d_mu)/current_T #Note: d_mu needs to be here as d_U is looking at crystal and liquid (no biasing term)
    shift = step_prefactor*d_mu/dmu_dT
    if shift > 150:
        shift=150
    if shift < -150:
        shift=-150
    return  shift + current_T




def classius_clapeyron_next_temp(current_T : float, dP : float, crystal : ReadLammps, liquid : ReadLammps, thinned=None):
    dV = (crystal.average_vol(thinned) - liquid.average_vol(thinned))
    dH = (crystal.average_enthalpy(thinned) - liquid.average_enthalpy(thinned))
    shift = current_T*dV*dP*BAR_TO_EV/dH
    return current_T + shift




def run_till_converged(IP : IPparams, tmeltout : str, melt_steps : int, tol=50, samples=10):
    with open(tmeltout , "w") as f:
        f.write('T \t d_mu \t mean_Q \t mean_Qc \t mean_Ql \t std_Q \t std_Qc \t std_Ql \t IP_a \t IP_kappa \n')
    with open(tmeltout + '_energies', 'w') as f:
        f.write('T \t d_mu \t d_mu_std \t dmu_dT \t dmu_dT_std \t pdv \t pdv_std \t d_U \t d_U_std \n')

    temperature_history=np.array([IP.temperature])
    for i in range(melt_steps):
        IP.write_crystal()
        subprocess.run(f"{IP.LAMMPS_RUN_COMMAND} < {IP.location}/in.crystal_auto > {IP.location}/crystal_auto.out", shell=True)
        crystal = ReadLammps(f'{IP.location}/crystal_auto.out')
        atoms_crystal_end = ase.io.read(f'{IP.location}/data.crystal_end', format='lammps-data', style='atomic')
        Lx, Ly, Lz = crystal.average_lx(IP.thinned), crystal.average_ly(IP.thinned), crystal.average_lz(IP.thinned)
        atoms_crystal_end.set_cell([Lx, Ly, Lz], scale_atoms=True)
        ase.io.write(f'{IP.location}/data.crystal_end_meaned', atoms_crystal_end, format='lammps-data')

        IP.write_liquid()
        subprocess.run(f"{IP.LAMMPS_RUN_COMMAND} < {IP.location}/in.setup_liquid_auto > {IP.location}/setup_liquid_auto.out", shell=True)
        subprocess.run(f"{IP.LAMMPS_RUN_COMMAND} < {IP.location}/in.liquid_auto > {IP.location}/liquid_auto.out", shell=True)
        liquid = ReadLammps(f'{IP.location}/liquid_auto.out')

        if IP.auto==True: IP.update_IP(kappa_prefactor=IP.kappa_prefactor)

        IP.write_coex()
        subprocess.run(f"{IP.LAMMPS_RUN_COMMAND} < {IP.location}/in.setup_pinning_auto > {IP.location}/setup_pinning_auto.out", shell=True)
        subprocess.run(f"{IP.LAMMPS_RUN_COMMAND} < {IP.location}/in.pinning_auto > {IP.location}/pinning_auto.out", shell=True)
        pinning = ReadLammps(f'{IP.location}/pinning_auto.out')

#        d_mu = delta_mu(pinning, crystal, liquid, kappa=IP.kappa, a=IP.a, N=IP.N, thinned=IP.thinned)
#        new_T = isobar_next_temp(current_P=IP.pressure, current_T=IP.temperature, 
#                                 crystal=crystal, liquid=liquid, d_mu=d_mu, N=IP.N, thinned=IP.thinned, step_prefactor=step_prefactor)

        dmu_dT, dmu_dT_std, pdv, pdv_std, d_mu, d_mu_std, d_U, d_U_std = energy_means_and_errors(IP.pressure, IP.temperature, pinning, crystal, liquid, kappa=IP.kappa, a=IP.a, N=IP.N, thinned=IP.thinned)
        new_T = isobar_next_temp(current_P=IP.pressure, current_T=IP.temperature, 
                                 crystal=crystal, liquid=liquid, d_mu=d_mu, N=IP.N, thinned=IP.thinned, step_prefactor=IP.step_prefactor)
        with open(tmeltout , "a") as f:
            f.write(str(IP.temperature) + '\t' + str(d_mu) + '\t' +\
                '\t' + str(pinning.average_Q(IP.thinned)) + '\t' + str(crystal.average_Q(IP.thinned)) + '\t' + str(liquid.average_Q(IP.thinned)) + \
                '\t' + str(pinning.std_Q(IP.thinned)) + '\t' + str(crystal.std_Q(IP.thinned)) + '\t' + str(liquid.std_Q(IP.thinned)) + '\t' \
                + str(IP.a) + '\t' + str(IP.kappa) +'\n')

        with open(tmeltout + '_energies', 'a') as f:
            f.write(str(IP.temperature) + '\t' + str(d_mu) + '\t' + str(d_mu_std) + '\t' + str(dmu_dT) + '\t' + str(dmu_dT_std) +\
                '\t' + str(pdv) + '\t' + str(pdv_std) + '\t' + str(d_U) + '\t' + str(d_U_std) +  '\n')

        IP.temperature = new_T
        temperature_history= np.append(temperature_history, new_T)
        # break IP if tol is reached for number of samples.
        if i > samples:
            if temperature_history[-samples:].std() < tol:
                print(f'P = {IP.pressure}, tol reached at step {i}, with std: {temperature_history[-samples:].std()} (K)', flush=True)
                return
    print(f'P = {IP.pressure}, end of melt steps, finished with std: {temperature_history[-samples:].std()} (K)', flush=True)
    return



    
