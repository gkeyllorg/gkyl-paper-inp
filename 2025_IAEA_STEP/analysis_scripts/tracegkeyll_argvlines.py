# coding: utf-8
import numpy as np
import matplotlib  as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import postgkyl as pg
import os
import math
import sys

import scipy.interpolate 
from scipy.interpolate import RegularGridInterpolator, interp1d
import scipy.integrate as sci


def fix_gridvals(grid):
    """Output file grids have cell-edge coordinates by default, but the values are at cell centers.
    This function will return the cell-center coordinate grid.
    Usage: cell_center_grid = fix_gridvals(cell_edge_grid)
    """
    grid = np.array(grid).squeeze()
    grid = (grid[0:-1] + grid[1:])/2
    return grid

# Universal params
mp = 1.67262192e-27
me = 9.1093837e-31
eV = 1.602e-19
mu_0 = 12.56637061435917295385057353311801153679e-7
eps0=8.854187817620389850536563031710750260608e-12

masses = {}
masses["elc"] = me
masses["ion"] = 2.014*mp


charges = {}
charges["elc"] = -eV 
charges["ion"] = eV


sim_dir = "../%s/"%sys.argv[1]
base_name = sim_dir+'hstep27'
bmin, bmax = 0, 8
sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]
sim_labels= ['b%d'%i for i in range(bmin,bmax)]

xidx_list = [0, 15, 15, 7, 7, 0, 15, 15]
zidx_list = [0, 0, 47, 0, 7, 7, 47, 0]


#frame = 1
start= int(sys.argv[2])
stop = int(sys.argv[3])
step = int(sys.argv[4])
end_frame = stop+1
frames = np.arange(start, stop+1, step)


remove_frames = []
for frame in frames:
    filepath = "%s/%s_b0-elc_M0_%d.gkyl"%(sim_dir, base_name, frame)
    if not os.path.exists(filepath):
        remove_frames.append(frame)


frames = frames[~np.isin(frames,remove_frames)]
lrange = len(frames)

num_species=2
density_midout = np.zeros((lrange, num_species))
density_midin = np.zeros((lrange, num_species))
density_downout = np.zeros((lrange, num_species))
density_downin = np.zeros((lrange, num_species))
density_coreout = np.zeros((lrange, num_species))
density_corein = np.zeros((lrange, num_species))

temp_midout = np.zeros((lrange, num_species))
temp_midin = np.zeros((lrange, num_species))
temp_downout = np.zeros((lrange, num_species))
temp_downin = np.zeros((lrange, num_species))
temp_coreout = np.zeros((lrange, num_species))
temp_corein = np.zeros((lrange, num_species))

phi_midout = np.zeros((lrange, 1))
phi_midin = np.zeros((lrange, 1))
phi_downout= np.zeros((lrange, 1))
phi_downin = np.zeros((lrange, 1))
phi_coreout = np.zeros((lrange, 1))
phi_corein = np.zeros((lrange, 1))

#time = np.zeros(end_frame-start)
time = np.zeros(lrange)
iframe=0
for frame in frames:
    print("frame is %d"%frame)
    mom_data_list = []
    for isim, sim_name in enumerate(sim_names):
        mom_data = {}

        ##Get Plate angle information
        #if isim == 1 : 
        #    plate_data = np.genfromtxt("./stepplate_data/highres/osol.txt", delimiter = ",")
        #if isim == 0 : 
        #    plate_data = np.genfromtxt("./stepplate_data/highres/opf.txt", delimiter = ",")
        #if isim == 4 : 
        #    plate_data = np.genfromtxt("./stepplate_data/highres/isol.txt", delimiter = ",")
        #if isim == 5 : 
        #    plate_data = np.genfromtxt("./stepplate_data/highres/ipf.txt", delimiter = ",")

    
        #Load moment data
        for species in ["elc", "ion"]:
            for mom in ["M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp", "BGKM0dot", "BGKM1dot", "BGKM2dot"]:
                mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
                grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
                val = val.squeeze()[xidx_list[isim], zidx_list[isim]]
                mom_data[species+mom] = val
                #x = fix_gridvals(grid[0])
                #z = fix_gridvals(grid[1])
                #val = val.squeeze()
                #if(xidx_list[isim] == 0):
                #    xsel = 0
                #else:
                #    xsel = 1
                #if(zidx_list[isim] == 0):
                #    zsel = 0
                #else:
                #    zsel = 1
                #mom_data[species+mom] = val[xsel,zsel]
 
            # Set interpolated moment data
            mom_data[species+"Temp"] =  (masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tpar"] =  (masses[species]) * (mom_data[species+"M2par"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tperp"] =  (masses[species]/2) * (mom_data[species+"M2perp"])/mom_data[species+"M0"] / eV
            mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])
            mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
        mom_data["Qtot"] =  mom_data["elcQ"]+mom_data["ionQ"]


        # Load the potential
        #mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame), z0 = xidx_list[isim], z1 = xidx_list[isim])
        mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
        grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
        x = fix_gridvals(grid[0])
        z = fix_gridvals(grid[1])
        val = val.squeeze()[xidx_list[isim], zidx_list[isim]]
        mom_data["phi"] = val
        #if(xidx_list[isim] == 0):
        #    xsel = 0
        #else:
        #    xsel = 1
        #if(zidx_list[isim] == 0):
        #    zsel = 0
        #else:
        #    zsel = 1
        #mom_data["phi"] = val[xsel,zsel]
        mom_data["time"] = float(mdata.info()[9:21])

        for species in ["elc", "ion"]:
            mom_data[species+"Qwall"] = mom_data[species+"Q"] + charges[species]*mom_data[species+"M1"]*mom_data["phi"]

        ## Interpolate B ratio at plates
        #if isim in [1,0,4,5]:
        #    Binterpolator = interp1d(plate_data[:,0], plate_data[:,1])
        #    Bratio = Binterpolator(x)
        #    mom_data["Bratio" ] = Bratio

        mom_data_list.append(mom_data)

    for s, species in enumerate(["elc", "ion"]):
        density_midout[iframe, s] = mom_data_list[2][species+"M0"]
        density_midin[iframe, s] = mom_data_list[3][species+"M0"]
        density_downout[iframe, s] = mom_data_list[1][species+"M0"]
        density_downin[iframe, s] = mom_data_list[4][species+"M0"]
        density_coreout[iframe, s] = mom_data_list[6][species+"M0"]
        density_corein[iframe, s] = mom_data_list[7][species+"M0"]
        temp_midout[iframe, s] = mom_data_list[2][species+"Temp"]
        temp_midin[iframe, s] = mom_data_list[3][species+"Temp"]
        temp_downout[iframe, s] = mom_data_list[1][species+"Temp"]
        temp_downin[iframe, s] = mom_data_list[4][species+"Temp"]
        temp_coreout[iframe, s] = mom_data_list[6][species+"Temp"]
        temp_corein[iframe, s] = mom_data_list[7][species+"Temp"]

    phi_midout[iframe, 0] = mom_data_list[2]["phi"]
    phi_midin[iframe, 0] = mom_data_list[3]["phi"]
    phi_downout[iframe, 0] = mom_data_list[1]["phi"]
    phi_downin[iframe, 0] = mom_data_list[4]["phi"]
    phi_coreout[iframe, 0] = mom_data_list[6]["phi"]
    phi_corein[iframe, 0] = mom_data_list[7]["phi"]
    
    time[iframe] = float(mdata.info().split('\n')[0].split("Time: ")[1]) 
    iframe+=1
 

elc_handle = mlines.Line2D([],[], color = 'tab:blue', label = "elc")
ion_handle = mlines.Line2D([],[], color = 'tab:orange', label = "ion")
handles_species = [elc_handle, ion_handle]

fign, axn = plt.subplots(3, 2, figsize = (16,9))
axn[0,0].plot(time, density_corein)
axn[0,0].set_ylabel('n at inner core')
axn[0,1].plot(time, density_coreout)
axn[0,1].set_ylabel('n at outer core')

axn[1,0].plot(time, density_midin)
axn[1,0].set_ylabel('n at IMP')
axn[1,1].plot(time, density_midout)
axn[1,1].set_ylabel('n at OMP')

axn[2,0].plot(time, density_downin)
axn[2,0].set_ylabel('n at Inboard Plate')
axn[2,1].plot(time, density_downout)
axn[2,1].set_ylabel('n at Outboard Plate')
for i in range(3):
    for j in range(2):
        axn[i,j].legend(handles = handles_species)

fign.suptitle(sim_dir)

figt, axt = plt.subplots(3, 2, figsize = (16,9))
axt[0,0].plot(time, temp_corein)
axt[0,0].set_ylabel('T at inner core')
axt[0,1].plot(time, temp_coreout)
axt[0,1].set_ylabel('T at outer core')

axt[1,0].plot(time, temp_midin)
axt[1,0].set_ylabel('T at IMP')
axt[1,1].plot(time, temp_midout)
axt[1,1].set_ylabel('T at OMP')

axt[2,0].plot(time, temp_downin)
axt[2,0].set_ylabel('T at Inboard Plate')
axt[2,1].plot(time, temp_downout)
axt[2,1].set_ylabel('T at Outboard Plate')

for i in range(3):
    for j in range(2):
        axt[i,j].legend(handles = handles_species)

figt.suptitle(sim_dir)

figp, axp = plt.subplots(3, 2, figsize = (16,9))
axp[0,0].plot(time, phi_corein)
axp[0,0].set_ylabel('phi at inner core')
axp[0,1].plot(time, phi_coreout)
axp[0,1].set_ylabel('phi at outer core')

axp[1,0].plot(time, phi_midin)
axp[1,0].set_ylabel('phi at IMP')
axp[1,1].plot(time, phi_midout)
axp[1,1].set_ylabel('phi at OMP')

axp[2,0].plot(time, phi_downin)
axp[2,0].set_ylabel('phi at Inboard Plate')
axp[2,1].plot(time, phi_downout)
axp[2,1].set_ylabel('phi at Outboard Plate')
figp.suptitle(sim_dir)



frames_vline = np.r_[int(sys.argv[5])]
times_vline = frames_vline*1e-5
for axs in [axn,axt,axp]:
    for i in range(3):
        for j in range(2):
            for k in range(len(frames_vline)):
                axs[i,j].axvline(times_vline[k], color='k', linestyle='dashed')
 

