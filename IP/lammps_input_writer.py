import numpy as np
import io, os
import ase, ase.io
from ase.units import kB

def write_in_crystal(potential : str, location: str, pressure : float, temperature : float, step = 10000, hkl=[6,6,0],
    barostat=1.00, thermostat=0.25, dt=0.001):
    param_string="units           metal \n" \
    "dimension	3  \n" \
    "boundary        p p p  \n" \
    "atom_style  atomic  \n" \
    f"read_data     {location}/data.crystal  \n" \
    "mass          1 26.98  \n" \
    "mass          2 47.867  \n" \
    "mass          3 50.94  \n" \
    f"{potential}" \
    "neighbor        0.3 bin  \n" \
    "neigh_modify    delay 10  \n" \
    f"timestep        {dt}  \n" \
    "run_style	verlet  \n" \
    f"fix ensemble all npt temp {temperature} {temperature} {thermostat} aniso {pressure} {pressure} {barostat} pchain 32  \n" \
    f"fix bias all rhok {hkl[0]} {hkl[1]} {hkl[2]} 0.0 0.0  \n" \
    f"##fix montecarlo all atom/swap 1000 2 999 {temperature} types 1 2 \n"\
    f"##fix montecarlo all atom/swap 1000 2 123 {temperature} types 3 2 \n"\
    "thermo 50  \n" \
    "thermo_style custom step temp press density f_bias[3] vol etotal enthalpy lx ly lz\n" \
    f"dump dumpXYZ all xyz 500 {location}/traj_crystal.xyz  \n" \
    f"run {step}  \n" \
    f"write_data {location}/data.crystal_end  \n"
    with open(f'{location}/in.crystal_auto' , "w") as f:
        f.write(param_string)
    return 
    
def write_in_setup_pinning(potential : str, location: str, pressure : float, temperature : float, step = 10000, hkl=[6,6,0], z_half=0, 
    barostat=1.00, thermostat=0.25, dt=0.001):
    param_string="units           metal \n" \
    "dimension	3  \n" \
    "boundary        p p p  \n" \
    "atom_style  atomic  \n" \
    f"read_data     {location}/data.crystal_end_meaned  \n" \
    "mass          1 26.98  \n" \
    "mass          2 47.867  \n" \
    "mass          3 50.94  \n" \
    f"{potential} "\
    "neighbor        0.3 bin  \n" \
    f"timestep        {dt}  \n" \
    "run_style	verlet  \n" \
    f"region left plane 0 0 {z_half} 0 0 1  \n" \
    "group left region left  \n" \
    f"velocity left create {temperature + 1500} 1 mom yes rot yes  \n" \
    "fix ensemble left nve     # Note: only move particle in left-hand side  \n" \
    f"fix langevin left langevin {temperature + 1500} {temperature} {thermostat} 2017  \n" \
    "thermo_style custom step temp pzz pe lz vol density etotal  \n" \
    "thermo 100  \n" \
    f"dump dumpXYZ all xyz 50 {location}/traj_setup.xyz  \n" \
    f"run {step}  \n" \
    f"write_data {location}/data.halfhalf  \n" 
    with open(f'{location}/in.setup_pinning_auto' , "w") as f:
        f.write(param_string)
    return param_string


def write_in_pinning(potential : str, location: str, pressure : float, temperature : float, step = 10000, hkl=[6,6,0],
IP_a=20.00, IP_kappa=4.0, barostat=1.00, thermostat=0.25, dt=0.001):
    param_string="units           metal \n" \
    "dimension	3  \n" \
    "boundary        p p p  \n" \
    "atom_style  atomic  \n" \
    f"read_data     {location}/data.halfhalf  \n" \
    "mass          1 26.98  \n" \
    "mass          2 47.867  \n" \
    "mass          3 50.94  \n" \
    f"{potential}"\
    "neighbor        0.3 bin \n" \
    f"timestep        {dt} \n" \
    "run_style	verlet \n" \
    f"velocity all create  {temperature}  1 mom yes rot yes \n" \
    f"fix ensemble all npt temp {temperature} {temperature} {thermostat} z {pressure} {pressure} {barostat} \n" \
    "fix 100 all momentum 100 linear 1 1 1 \n" \
    f"fix bias all rhok {hkl[0]} {hkl[1]} {hkl[2]}   {IP_kappa}   {IP_a} \n" \
    f"##fix montecarlo all atom/swap 1000 2 999 {temperature} types 1 2 \n"\
    f"##fix montecarlo all atom/swap 1000 2 123 {temperature} types 3 2 \n"\
    "thermo_style custom step temp pzz pe lz f_bias f_bias[1] f_bias[2] f_bias[3] vol etotal \n" \
    "thermo 50 \n" \
    f"dump dumpXYZ all xyz 100 {location}/traj_pinning.xyz \n" \
    f"run {step} \n" 
    with open(f'{location}/in.pinning_auto' , "w") as f:
        f.write(param_string)
    return param_string


def write_in_setup_liquid(potential : str, location: str, pressure : float, temperature : float, step = 10000, hkl=[6,6,0],
    barostat=1.00, thermostat=0.25, dt=0.001):
    param_string="units           metal \n" \
    "dimension	3  \n" \
    "boundary        p p p  \n" \
    "atom_style  atomic  \n" \
    f"read_data     {location}/data.crystal_end_meaned  \n"\
    "mass          1 26.98  \n" \
    "mass          2 47.867  \n" \
    "mass          3 50.94  \n" \
    f"{potential}"\
    "neighbor        0.3 bin \n" \
    f"timestep        {dt} \n" \
    "run_style	verlet \n" \
    f"fix ensemble all nvt temp {temperature + 1500} {temperature} {thermostat} \n" \
    f"fix bias all rhok {hkl[0]} {hkl[1]} {hkl[2]}  0.0 0.0 \n" \
    "thermo 50 \n" \
    "thermo_style custom step temp press density f_bias[3] vol etotal \n" \
    "#dump dumpXYZ all xyz 500 traj_liquid_melting.xyz \n" \
    f"run {step} \n" \
    f"write_data {location}/data.liquid \n"
    with open(f'{location}/in.setup_liquid_auto' , "w") as f:
        f.write(param_string)
    return param_string

def write_in_liquid(potential : str , location: str, pressure : float, temperature : float, step = 10000, hkl=[6,6,0],
    barostat=1.00, thermostat=0.25, dt=0.001):
    param_string="units           metal \n" \
    "dimension	3  \n" \
    "boundary        p p p  \n" \
    "atom_style  atomic  \n" \
    f"read_data     {location}/data.liquid  \n" \
    "mass          1 26.98  \n" \
    "mass          2 47.867  \n" \
    "mass          3 50.94  \n" \
    f"{potential}" \
    "neighbor        0.3 bin \n" \
    f"timestep        {dt} \n" \
    "run_style	verlet \n" \
    f"velocity all create  {temperature} 1 mom yes rot yes \n" \
    f"fix ensemble all npt temp {temperature} {temperature} {thermostat} z {pressure} {pressure} {barostat} \n" \
    "fix 100 all momentum 100 linear 1 1 1 \n" \
    f"fix bias all rhok {hkl[0]} {hkl[1]} {hkl[2]}  0.0 0.0 \n" \
    "thermo_style custom step temp pzz pe lz f_bias f_bias[1] f_bias[2] f_bias[3] vol etotal enthalpy \n" \
    "thermo 50 \n" \
    f"dump dumpXYZ all xyz 500 {location}/traj_liquid.xyz \n" \
    f"run {step} \n" 
    with open(f'{location}/in.liquid_auto' , "w") as f:
        f.write(param_string)
    return param_string

class ReadLammps():
    #Read lammps output: assumes Step is outputted.
    def __init__(self, filein):
        self.get_dict(filein)
    def get_dict(self,filein):
        with open(filein) as f:
            lines=f.readlines()
        for i, val in enumerate(lines):
            if 'Step' in val:
                break
        start_loc=i+1        
        for i, val in enumerate(lines):
            if 'Loop time' in val:
                break
        end_loc=i

        labels = lines[start_loc-1].split()[0:]
        mydict = {}
        for i in labels:
            mydict[i]=[]
        for i, val in enumerate(lines[start_loc:end_loc]):
                for j, val2 in enumerate(labels):
                    mydict[val2].append(val.split()[j])
        self.dict = mydict
    def average_Q(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['f_bias[3]'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['f_bias[3]'], dtype=float).mean()

    def std_Q(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['f_bias[3]'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['f_bias[3]'], dtype=float).std()


    def average_vol(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Volume'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Volume'], dtype=float).mean()

    def average_density(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Density'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Density'], dtype=float).mean()

    def average_energy(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['TotEng'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['TotEng'], dtype=float).mean()

    def average_potential_pinned(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            total_pot=np.array(self.dict['PotEng'], dtype=float)[-thinned:]
            bias=np.array(self.dict['f_bias'], dtype=float)[-thinned:]
            return (total_pot - bias).mean()
        return (total_pot - bias).mean()

    def std_potential_pinned(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            total_pot=np.array(self.dict['PotEng'], dtype=float)[-thinned:]
            bias=np.array(self.dict['f_bias'], dtype=float)[-thinned:] #not sure if this is contained in PE or not...
            return (total_pot).std()
        return (total_pot - bias).std()

    def average_bias_pot(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['f_bias'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['f_bias'], dtype=float).mean()

    def std_bias_pot(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['f_bias'], dtype=float)[-thinned:].std()
        return np.array(self.dict['f_bias'], dtype=float).std()

    def average_temp(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Temp'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Temp'], dtype=float).mean()

    def average_pressure(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Press'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Press'], dtype=float).mean()

    def average_enthalpy(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Enthalpy'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Enthalpy'], dtype=float).mean()

    def average_lx(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Lx'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Lx'], dtype=float).mean()

    def average_ly(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Ly'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Ly'], dtype=float).mean()

    def average_lz(self, thinned=None):
        if thinned!=None and type(thinned) == int:
            return np.array(self.dict['Lz'], dtype=float)[-thinned:].mean()
        return np.array(self.dict['Lz'], dtype=float).mean()



class IPparams():
    def __init__(self, LAMMPS_RUN_COMMAND, atoms, toploc, potential ,pressure, temperature, step, step_setup, hkl, 
    thermostat, barostat, Nz, dt=0.001 , thinned=30, auto=False, kappa_prefactor=1,
    step_prefactor=1, z_half=None, kappa=None, a=None):
        if auto ==False and kappa==None and a==None:
            print('ERROR: Warning, not using IP.auto, kappa and a need to be set!')
            return
        self.toploc = toploc
        self.location=f'{toploc}/P_{str(pressure).zfill(6)}'
        if not(os.path.exists(self.location)):
            os.mkdir(self.location)
            ase.io.write(f'{self.location}/data.crystal', atoms, format='lammps-data')
        self.potential=potential
        self.pressure=pressure
        self.barostat=barostat
        self.temperature=temperature
        self.thermostat=thermostat
        self.dt = dt
        self.step=step
        self.step_setup=step_setup
        self.hkl=hkl
        if z_half == None:
            self.z_half=np.round(atoms.cell[2][2]/2, 1)
        self.Nz = Nz
        self.a=a
        self.kappa=kappa
        self.thinned=thinned
        self.N = len(atoms)
        self.atoms = atoms
        self.LAMMPS_RUN_COMMAND = LAMMPS_RUN_COMMAND
        self.auto = auto
        self.kappa_prefactor = kappa_prefactor
        self.step_prefactor = step_prefactor
        self.traj_name =  f"{toploc}/P_{str(pressure).zfill(6)}.iptraj"
        return

    def write_all(self):
        self.write_crystal()
        self.write_liquid()
        self.write_coex()
        return
    
    def write_crystal(self):
        write_in_crystal(potential=self.potential, location=self.location, pressure=self.pressure, temperature=self.temperature, step=self.step,
        hkl=self.hkl, thermostat=self.thermostat, barostat=self.barostat, dt=self.dt)
        return

    def write_liquid(self):
        write_in_setup_liquid(potential=self.potential, location=self.location, pressure=self.pressure, temperature=self.temperature, step=self.step_setup,
        hkl=self.hkl, thermostat=self.thermostat, barostat=self.barostat, dt=self.dt)
        write_in_liquid(potential=self.potential, location=self.location, pressure=self.pressure, temperature=self.temperature, step=self.step, hkl=self.hkl,
        thermostat=self.thermostat, barostat=self.barostat, dt=self.dt)
        return
    
    def write_coex(self):
        write_in_setup_pinning(potential=self.potential, location=self.location, pressure=self.pressure, temperature=self.temperature, step=self.step_setup,
        hkl=self.hkl, z_half=self.z_half, thermostat=self.thermostat, barostat=self.barostat, dt=self.dt)
        write_in_pinning(potential=self.potential, location=self.location, pressure=self.pressure, temperature=self.temperature, step=self.step, hkl=self.hkl,
        IP_a=self.a, IP_kappa=self.kappa, thermostat=self.thermostat, barostat=self.barostat, dt=self.dt)
        return

    def update_IP(self, kappa_prefactor=1):
        crystal = ReadLammps(f'{self.location}/crystal_auto.out')
        liquid = ReadLammps(f'{self.location}/liquid_auto.out')
        q_crystal = crystal.average_Q(thinned=self.thinned)
        q_liquid = liquid.average_Q(thinned=self.thinned)
        self.a = q_liquid+ (q_crystal-q_liquid)/2
        self.kappa = kappa_prefactor*kB*self.temperature*self.Nz**2/(q_crystal-q_liquid)**2
        return

    def write_IP_params(self):
        with open(f'{self.toploc}/params.txt' , "w") as f:
            f.write('Paramters used: \n' +\
            f'start pressure: {self.pressure}  \n ' +\
            f'start temperature: {self.temperature}  \n' +\
            f'number of MD steps: {self.step}  \n' +\
            f'number of setup MD steps: {self.step_setup}  \n' +\
            f'hkl: {self.hkl}  \n' +\
            f'thermostat: {self.thermostat}  \n' +\
            f'barostat: {self.barostat}  \n' +\
            f'timestep: {self.dt} ps \n' +\
            f'z_half: {self.z_half}  \n' +\
            f'IP_a: {self.a}  \n' +\
            f'IP_kappa: {self.kappa}  \n' +\
            f'IP_auto: {self.auto}  \n' +\
            f'thinned: {self.thinned}  \n' +\
            f'Number of atoms: {self.N}  \n' +\
            f'Kappa prefactor: {self.kappa_prefactor}  \n' +\
            f'Next T step prefactor: {self.step_prefactor}  \n'
            )
        return

    def next_isobar_start(self, pressure, temperature):
        self.location=f'{self.toploc}/P_{str(pressure).zfill(6)}'
        self.pressure=pressure
        if not(os.path.exists(self.location)):
            os.mkdir(self.location)
            ase.io.write(f'{self.location}/data.crystal', self.atoms, format='lammps-data')
        self.traj_name =  f"{self.toploc}/P_{str(pressure).zfill(6)}.iptraj"
        self.temperature = temperature
        return


    
