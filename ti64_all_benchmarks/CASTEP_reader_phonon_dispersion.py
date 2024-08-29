import numpy as np
import ase, ase.io
import re
import warnings
warnings.simplefilter('ignore')

class PhononFromCastep:
    '''
        Extract phonon frequencies and kpoints from a CASTEP phonon calculation.
        Requires that the calculation used (THz) units. 
        Optional: may rescale phonon despersion axis from 0 to 1 for a specified k-path (must be same as that done in CASTEP). 
    '''
    RESCALE_TOL = 1e-4
    def __init__(self, castep_file :str, kpath_in=None, verbose=False):
        if castep_file:
            try:
                atoms=ase.io.read(castep_file)
            except AttributeError:
                print('Invalid Input Type!')
            self.number_of_branches = len(atoms)*3
            self.filename=castep_file
            self.read_in_file()
            self.get_frequencies()
            self.get_kpath()
            if verbose:
                print(f'Atoms object info: \n{atoms}\n')
                print(self.__dict__.keys(), '\n')
        else: print('Castep File Not Defined')            
        if kpath_in != None:
            self.rescale_xaxis(kpath_in)
            if verbose: print('k-path rescaled')
        elif verbose: print('no k-path re-scaling done')
        delattr(self, 'filelines')
    def __str__(self):
        return f"Phonon Dispersion (THz) from Castep file object"
    
    def read_in_file(self):
        with open(self.filename) as f:
            self.filelines = f.readlines()
    def get_frequencies(self):
        headlines=2 #number of lines before frequency numbers appear
        #find where the index for when (THz) occurs, then get the correct filelines into one list
        thzlist = [self.filelines[i+headlines:i+headlines+self.number_of_branches]
                   for i, val in enumerate(self.filelines) if re.search(f' \(THz\) ', val) != None]
        thzlist = [j for i in thzlist for j in i] #flatten list
        thzlist = [i.split()[2] for i in thzlist] #just get the numbers from string
        frequencies =  np.array(thzlist, dtype=float)    
        self.kpoints = int((len(frequencies)/self.number_of_branches))
        self.frequencies = np.reshape(frequencies, [self.kpoints, self.number_of_branches])        
    def get_kpath(self):
        qptlist = [self.filelines[i]
                   for i, val in enumerate(self.filelines) if re.search(f'q-pt=', val) != None]
        for i,val in enumerate(qptlist):
            temp= val.split()[4:7]
            temp[2] = temp[2].replace(")",  "")
            qptlist[i] = temp
        self.kpath=np.array(qptlist, dtype=float)  
    
    def find_index(self, in_path):
        j=0
        sympoint_idx=[]
        self.kpath_idx=[]
        for i, val in enumerate(self.kpath): #loop to find high symm points!
            if abs(np.linalg.norm(in_path[j]-val)) < self.RESCALE_TOL:
                sympoint_idx.append(i)
                j=j+1
        for i in range(len(sympoint_idx)-1):
            x = [j for j in range(sympoint_idx[i], sympoint_idx[i+1]+1)]
            self.kpath_idx.append(x)
        
    def rescale_xaxis(self, rescale_xaxis):
        #rescale axis to go 0 to 1
        in_path = np.array(rescale_xaxis, dtype=float)
        self.find_index(in_path)
        xsplit = 1/len(self.kpath_idx)
        xscale=[0]
        pos=0
        kpath_cut=[]
        kpath_cut.append(self.kpath_idx[0])
        for i in range(len(self.kpath_idx)-1):
            kpath_cut.append(self.kpath_idx[i+1][1:])
        x_inc = xsplit/(len(kpath_cut[0])-1)
        for j in range(len(kpath_cut[0])-1):
            pos = pos + x_inc
            xscale.append(pos)        
        for i in range(len(kpath_cut)-1):
            x_inc = xsplit/(len(kpath_cut[i+1]))
            for j in range(len(kpath_cut[i+1])):
                pos = pos + x_inc
                xscale.append(pos)
        self.xscale = np.array(xscale)


