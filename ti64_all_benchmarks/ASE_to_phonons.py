import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import ase, ase.build, ase.io
from ase.calculators.castep import Castep
from glob import glob
import os
from quippy.potential import Potential
import warnings
warnings.simplefilter('ignore')




class AtomsToPhonons:
    def __init__(self, primitive_cell, phonon_grid, displacement, kpath, calculator, kpoints=100, castep=False, plusminus=False,diagonal=True):
        self.calculator_string=calculator
        self.get_supercell(primitive_cell, phonon_grid, displacement, plusminus,diagonal)
        if castep==False: self.get_forces_gap()
        if castep==True: self.get_forces_castep()
        self.get_band_struct(kpath, kpoints)   
    def get_supercell(self, primitive_cell, phonon_grid, displacement, plusminus,diagonal):
        self.unitcell = PhonopyAtoms(symbols=primitive_cell.get_chemical_symbols(),
                        cell=primitive_cell.get_cell(),
                        scaled_positions=primitive_cell.get_scaled_positions()
                       )
        self.phonon = Phonopy(self.unitcell,phonon_grid)
        self.phonon.generate_displacements(distance=displacement, is_plusminus=plusminus, is_diagonal=diagonal)
        self.supercells = self.phonon.supercells_with_displacements        
    def get_forces_gap(self):
        potential= self.calculator_string
        forces = []
        self.atoms=[]
        for i, s in enumerate(self.supercells) :
            a=ase.Atoms(symbols=s.get_chemical_symbols(), cell=s.get_cell(), 
                scaled_positions=s.get_scaled_positions(), pbc=True)
            self.atoms.append(a)
            a.calc = potential
            forces.append(a.get_forces())
        self.forces = forces
    def get_forces_castep(self):
        forces=[]
        self.atoms=[]
        for i, s in enumerate(self.calculator_string):
            castep_atoms = ase.io.read(s)
            self.atoms.append(castep_atoms)
            forces.append(castep_atoms.get_forces())
        self.forces = forces    
    def get_band_struct(self, kpath, kpoints):
        self.phonon.set_forces(self.forces)
        self.phonon.produce_force_constants()
        qpoints, connections = get_band_qpoints_and_path_connections(kpath, npoints=kpoints)
        self.phonon.run_band_structure(qpoints, with_eigenvectors=True, path_connections=connections) 
        self.frequencies_array = self.phonon.get_band_structure_dict()['frequencies']
        self.eigenvectors = self.phonon.get_band_structure_dict()['eigenvectors']
        self.frequencies=self.frequencies_array[0]
        xticks=[]
        x=0
        for i, val in enumerate(self.frequencies_array[1:]):
            self.frequencies = np.append(self.frequencies, val, axis=0)
            x+=len(val)
            xticks.append(x)
        n_kpoints = kpoints*len(kpath[0])
        n_trace = int(n_kpoints / (len(kpath[0])))
        self.normal_ticks = [i*n_trace for i in range(len(kpath[0]))]

class AtomsToPDOS:
    def __init__(self, primitive_cell, phonon_grid, displacement, calculator, kpoints=100, castep=False, plusminus=False,diagonal=True):
        self.calculator_string=calculator
        self.get_supercell(primitive_cell, phonon_grid, displacement, plusminus,diagonal)
        if castep==False: self.get_forces_gap()
        if castep==True: self.get_forces_castep()
        self.get_dynamical_matrix()
    def get_supercell(self, primitive_cell, phonon_grid, displacement, plusminus,diagonal):
        self.unitcell = PhonopyAtoms(symbols=primitive_cell.get_chemical_symbols(),
                        cell=primitive_cell.get_cell(),
                        scaled_positions=primitive_cell.get_scaled_positions()
                       )
        self.phonon = Phonopy(self.unitcell,phonon_grid)
        self.phonon.generate_displacements(distance=displacement, is_plusminus=plusminus, is_diagonal=diagonal)
        self.supercells = self.phonon.supercells_with_displacements        
    def get_forces_gap(self):
        potential= self.calculator_string
        forces = []
        self.atoms=[]
        for i, s in enumerate(self.supercells) :
            a=ase.Atoms(symbols=s.get_chemical_symbols(), cell=s.get_cell(), 
                scaled_positions=s.get_scaled_positions(), pbc=True)
            self.atoms.append(a)
            a.calc = potential
            forces.append(a.get_forces())
        self.forces = forces
    def get_forces_castep(self):
        forces=[]
        self.atoms=[]
        for i, s in enumerate(self.calculator_string):
            castep_atoms = ase.io.read(s)
            self.atoms.append(castep_atoms)
            forces.append(castep_atoms.get_forces())
        self.forces = forces   
    def get_dynamical_matrix(self):
        self.phonon.set_forces(self.forces)
        self.phonon.produce_force_constants()
    def get_PDOS(self, qmesh):
        self.phonon.run_mesh(qmesh, with_eigenvectors=True, is_mesh_symmetry=False)
        self.phonon.run_projected_dos()
        self.phonon.run_total_dos()
        self.pdos = self.phonon.get_projected_dos_dict()
    def get_tp(self, qmesh, tmin=0,tmax=2000, tstep=100):
        self.phonon.run_mesh(qmesh)
        self.phonon.run_thermal_properties(t_step=tstep,
                              t_max=tmax,
                              t_min=tmin)
        self.tp_dict = self.phonon.get_thermal_properties_dict()

    def get_DOS(self, qmesh):
        self.phonon.run_mesh(qmesh)
        self.phonon.run_total_dos()
        self.dos = self.phonon.get_total_dos_dict()


# class AtomsToPhonons:
#     def __init__(self, primitive_cell, phonon_grid, displacement, kpath, calculator, kpoints=100, castep=False, plusminus=False,diagonal=True):
#         self.calculator_string=calculator
#         self.get_supercell(primitive_cell, phonon_grid, displacement, plusminus,diagonal)
#         if castep==False: self.get_forces_gap()
#         if castep==True: self.get_forces_castep()
#         self.get_band_struct(kpath, kpoints)   
#     def get_supercell(self, primitive_cell, phonon_grid, displacement, plusminus,diagonal):
#         self.unitcell = PhonopyAtoms(symbols=primitive_cell.get_chemical_symbols(),
#                         cell=primitive_cell.get_cell(),
#                         scaled_positions=primitive_cell.get_scaled_positions()
#                        )
#         self.phonon = Phonopy(self.unitcell,phonon_grid)
#         self.phonon.generate_displacements(distance=displacement, is_plusminus=plusminus, is_diagonal=diagonal)
#         self.supercells = self.phonon.supercells_with_displacements        
#     def get_forces_gap(self):
#         potential= Potential(param_filename=self.calculator_string[0])
#         forces = []
#         self.atoms=[]
#         for i, s in enumerate(self.supercells) :
#             a=ase.Atoms(symbols=s.get_chemical_symbols(), cell=s.get_cell(), 
#                 scaled_positions=s.get_scaled_positions(), pbc=True)
#             self.atoms.append(a)
#             a.calc = potential
#             forces.append(a.get_forces())
#         self.forces = forces
#     def get_forces_castep(self):
#         forces=[]
#         self.atoms=[]
#         for i, s in enumerate(self.calculator_string):
#             castep_atoms = ase.io.read(s)
#             self.atoms.append(castep_atoms)
#             forces.append(castep_atoms.get_forces())
#         self.forces = forces    
#     def get_band_struct(self, kpath, kpoints):
#         self.phonon.set_forces(self.forces)
#         self.phonon.produce_force_constants()
#         qpoints, connections = get_band_qpoints_and_path_connections(kpath, npoints=kpoints)
#         self.phonon.run_band_structure(qpoints, path_connections=connections) 
#         self.frequencies_array = self.phonon.get_band_structure_dict()['frequencies']
#         self.frequencies=self.frequencies_array[0]
#         xticks=[]
#         x=0
#         for i, val in enumerate(self.frequencies_array[1:]):
#             self.frequencies = np.append(self.frequencies, val, axis=0)
#             x+=len(val)
#             xticks.append(x)
#         n_kpoints = kpoints*len(kpath[0])
#         n_trace = int(n_kpoints / (len(kpath[0])))
#         self.normal_ticks = [i*n_trace for i in range(len(kpath[0]))]

# class AtomsToPDOS:
#     def __init__(self, primitive_cell, phonon_grid, displacement, calculator, kpoints=100, castep=False, plusminus=False,diagonal=True):
#         self.calculator_string=calculator
#         self.get_supercell(primitive_cell, phonon_grid, displacement, plusminus,diagonal)
#         if castep==False: self.get_forces_gap()
#         if castep==True: self.get_forces_castep()
#         self.get_dynamical_matrix()
#     def get_supercell(self, primitive_cell, phonon_grid, displacement, plusminus,diagonal):
#         self.unitcell = PhonopyAtoms(symbols=primitive_cell.get_chemical_symbols(),
#                         cell=primitive_cell.get_cell(),
#                         scaled_positions=primitive_cell.get_scaled_positions()
#                        )
#         self.phonon = Phonopy(self.unitcell,phonon_grid)
#         self.phonon.generate_displacements(distance=displacement, is_plusminus=plusminus, is_diagonal=diagonal)
#         self.supercells = self.phonon.supercells_with_displacements        
#     def get_forces_gap(self):
#         potential= Potential(param_filename=self.calculator_string[0])
#         forces = []
#         self.atoms=[]
#         for i, s in enumerate(self.supercells) :
#             a=ase.Atoms(symbols=s.get_chemical_symbols(), cell=s.get_cell(), 
#                 scaled_positions=s.get_scaled_positions(), pbc=True)
#             self.atoms.append(a)
#             a.calc = potential
#             forces.append(a.get_forces())
#         self.forces = forces
#     def get_forces_castep(self):
#         forces=[]
#         self.atoms=[]
#         for i, s in enumerate(self.calculator_string):
#             castep_atoms = ase.io.read(s)
#             self.atoms.append(castep_atoms)
#             forces.append(castep_atoms.get_forces())
#         self.forces = forces   
#     def get_dynamical_matrix(self):
#         self.phonon.set_forces(self.forces)
#         self.phonon.produce_force_constants()
#     def get_PDOS(self, qmesh):
#         self.phonon.run_mesh(qmesh, with_eigenvectors=True, is_mesh_symmetry=False)
#         self.phonon.run_projected_dos()
#         self.pdos = self.phonon.get_projected_dos_dict()
#     def get_tp(self, qmesh, tmin=0,tmax=2000, tstep=100):
#         self.phonon.run_mesh(qmesh)
#         self.phonon.run_thermal_properties(t_step=tstep,
#                               t_max=tmax,
#                               t_min=tmin)
#         self.tp_dict = self.phonon.get_thermal_properties_dict()

#     def get_DOS(self, qmesh):
#         self.phonon.run_mesh(qmesh)
#         self.phonon.run_total_dos()
#         self.dos = self.phonon.get_total_dos_dict()
