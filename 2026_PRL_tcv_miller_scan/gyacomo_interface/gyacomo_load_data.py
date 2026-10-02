import h5py
import numpy as np
from .gyacomo_tools import closest_index, is_convertible_to_float
import copy as cp

def load_data_0D(filename,dname):
    with h5py.File(filename, 'r') as file:
        # Load time data
        time  = file['data/var0d/time']
        time  = time[:]
        var0D = file['data/var0d/'+dname]
        var0D = var0D[:]
    return time, var0D

def load_data_3D_frame(filename,dname,tframe):
    with h5py.File(filename, 'r') as file:
        # load time
        time  = file['data/var3d/time']
        time  = time[:]
        # find frame
        iframe = closest_index(time,tframe)
        tf     = time[iframe]
        # Load data
        try:
            data = file[f'data/var3d/{dname}/{iframe:06d}']
        except:
            g_ = file[f'data/var3d/']
            print('Dataset: '+f'data/var3d/{dname}/{iframe:06d}'+' not found')
            print('Available fields: ')
            msg = ''
            for key in g_:
                msg = msg + key + ', '
            print(msg)
            exit()
        # Select the first species for species dependent fields
        if not (dname == 'phi' or dname == 'psi'):
            data = data[:,:,:,0]
        else:
            data = data[:]
        data = data['real']+1j*data['imaginary'] 
        # Load the grids
        kxgrid = file[f'data/grid/coordkx']
        kygrid = file[f'data/grid/coordky']
        zgrid  = file[f'data/grid/coordz']
    return time,data,tf

def load_data_5D_frame(filename,dname,tframe):
    with h5py.File(filename, 'r') as file:
        # load time
        time  = file['data/var5d/time']
        time  = time[:]
        # find frame
        iframe = closest_index(time,tframe)
        tf     = time[iframe]
        # Load data
        try:
            data = file[f'data/var5d/{dname}/{iframe:06d}']
        except:
            g_ = file[f'data/var5d/']
            print('Dataset: '+f'data/var5d/{dname}/{iframe:06d}'+' not found')
            print('Available fields: ')
            msg = ''
            for key in g_:
                msg = msg + key + ', '
            print(msg)
            exit()
        # Select the first species for species dependent fields
        data = data[:]
        data = data['real']+1j*data['imaginary'] 
        # Load the grids
        kxgrid = file[f'data/grid/coordkx']
        kygrid = file[f'data/grid/coordky']
        zgrid  = file[f'data/grid/coordz']
    return time,data,tf


def load_grids(filename):
    with h5py.File(filename, 'r') as file:
        # Load the grids
        kxgrid = file[f'data/grid/coordkx'][:]
        kygrid = file[f'data/grid/coordky'][:]
        zgrid  = file[f'data/grid/coordz'][:]
        pgrid  = file[f'data/grid/coordp'][:]
        jgrid  = file[f'data/grid/coordj'][:]
        Nx     = kxgrid.size
        Nky    = kygrid.size
        Ny     = 2*(Nky-1)
        Lx     = 2*np.pi/kxgrid[1]
        Ly     = 2*np.pi/kygrid[1]
        xgrid  = np.linspace(-Lx/2,Lx/2,Nx)
        ygrid  = np.linspace(-Ly/2,Ly/2,Ny)

    return xgrid,kxgrid,ygrid,kygrid,zgrid,pgrid,jgrid

def load_group(filename,group):
    data = {}
    with h5py.File(filename, 'r') as file:
        g_  = file[f"data/"+group]
        for key in g_.keys():
            name='data/'+group+'/'+key
            data[key] = file.get(name)[:]
    return data
  
def load_h5path(filename,path):
    with h5py.File(filename, 'r') as file:
        data = file.get(path)[:]
    return data

def load_3Dfield(filename,field):
    data = {}
    with h5py.File(filename, 'r') as file:
        g_  = file[f"data/var3d/"+field]
        for key in g_.keys():
            name='data/var3d/'+field+'/'+key
            data[key] = file.get(name)[:]
    return data

def load_params(filename):
    jobid = filename[-5:-3]
    with h5py.File(filename, 'r') as file:
        nml_str = file[f"files/STDIN."+jobid][0]
        nml_str = nml_str.decode('utf-8')
        params = read_namelist(nml_str)
    return params

# Function to read all namelists from a file
def read_namelist(nml_str):
    Nspecies = 1
    all_namelists = {}
    current_namelist = None
    nml_str = nml_str.split('\n')
    for line in nml_str:
        line = line.split('!')
        line = line[0]
        if line.startswith('&'):
            current_namelist = line[1:].strip()
            if current_namelist == 'SPECIES':
                current_namelist = current_namelist + "_" + str(Nspecies)
                Nspecies = Nspecies + 1
            all_namelists[current_namelist] = {}
        elif line.startswith('/'):
            current_namelist = None
        elif current_namelist:
            parts = line.split('=')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().rstrip(',').strip("'").strip()
                if is_convertible_to_float(value):
                    all_namelists[current_namelist][key] = float(value)
                else:
                    all_namelists[current_namelist][key] = value
    return all_namelists

def read_data_std(filename):
    t_values = []
    Pxi_values = []
    Qxi_values = []
    dict       = {"t":[],"Pxi":[],"Pxe":[],"Qxi":[],"Qxe":[]}
    with open(filename, 'r') as file:
        for line in file:
            a = line.split('|')
            for i in a[1:-1]:
                b = i.split('=')
                dict[b[0].strip()].append(float(b[1]))

    return dict