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

def myinterpolate(raw_x,raw_z,x,z,data):
    myinterpolator = RegularGridInterpolator((raw_x,raw_z), data, bounds_error=False, fill_value=None)
    x0,z0 = np.meshgrid(x,z)
    return myinterpolator((x0,z0)).T

num_basis = 2
def basis(x):
    return 1/np.sqrt(2), np.sqrt(3)*x/np.sqrt(2)

def get_interp_grid(grid):
    grid_interp = np.linspace(grid[0], grid[-1], 2*len(grid) -1)
    grid_interp_cc = 0.5*(grid_interp[0:-1] +grid_interp[1:])
    return grid_interp_cc

def interp_surface(coeffs, component):
    coeffs = coeffs[:,component*num_basis:(component+1)*num_basis]
    
    xval = -1/2
    q1 = np.sum(coeffs*basis(xval),axis=-1)
    xval = 1/2
    q2 = np.sum(coeffs*basis(xval),axis=-1)
    
    q = np.zeros(q1.shape[0]*2)
    for ix in range(q.shape[0]):
        q[ix] = q1[ix//2] if ix % 2 == 0 else q2[ix//2]

    return q

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


# Set up names for data loading
sim_dir = '../' + str(sys.argv[1]) + '/'
base_name = sim_dir+'hstep27'
bmin, bmax = 0, 8
sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]

sim_labels= ['b%d'%i for i in range(bmin,bmax)]


start= int(sys.argv[2])
source_frame = int(sys.argv[3])
end_frame = start+1

for frame in range(start,end_frame):
    Rlist = []
    Zlist = []
    half_mom_data_list = []
    raw_half_mom_data_list = []
    source_half_mom_data_list = []
    for isim, sim_name in enumerate(sim_names):
    
        mom_data = {}
        raw_mom_data = {}
    
        #Load geometry
    
        bdata = pg.GData('%s-bmag.gkyl'%sim_name)
        grid,val = pg.data.GInterpModal(bdata,poly_order=1,basis_type='ms').interpolate(0)
        mom_data["B"] = val.squeeze()
        raw_mom_data["B"] = bdata.get_values()[:,:,0]/2
        
        ci = 0
        for c1 in ["x","y","z"]:
            for c2 in ["x","y","z"]:
                if( c1=="y" and c2 =="x"):
                    continue
                if( c1=="z" and c2 !="z"):
                    continue
                gdata = pg.GData('%s-g_ij.gkyl'%sim_name)
                grid,val = pg.data.GInterpModal(gdata,poly_order=1,basis_type='ms').interpolate(ci)
                mom_data["g_%s%s"%(c1,c2)] = val.squeeze()
                raw_mom_data["g_%s%s"%(c1,c2)] = gdata.get_values()[:,:,0]/2
                ci+=1
        ci = 0
        for c1 in ["x","y","z"]:
            for c2 in ["x","y","z"]:
                if( c1=="y" and c2 =="x"):
                    continue
                if( c1=="z" and c2 !="z"):
                    continue
                gdata = pg.GData('%s-gij.gkyl'%sim_name)
                grid,val = pg.data.GInterpModal(gdata,poly_order=1,basis_type='ms').interpolate(ci)
                mom_data["g%s%s"%(c1,c2)] = val.squeeze()
                raw_mom_data["g%s%s"%(c1,c2)] = gdata.get_values()[:,:,0]/2
                ci+=1
        
        jdata = pg.GData('%s-jacobgeo.gkyl'%sim_name)
        grid,val = pg.data.GInterpModal(jdata,poly_order=1,basis_type='ms').interpolate(0)
        mom_data["J"] = val.squeeze()
        raw_mom_data["J"] = jdata.get_values()[:,:,0]/2
        raw_grid = jdata.get_grid()
        geo_fac = 1/mom_data["J"]/mom_data["B"]
    
        #Load grid data
        mc2pdata = pg.GData(sim_name + "-mapc2p_deflated.gkyl")
        _ ,R = pg.data.GInterpModal(mc2pdata,poly_order=1,basis_type='ms').interpolate(0)
        _ ,Z = pg.data.GInterpModal(mc2pdata,poly_order=1,basis_type='ms').interpolate(1)

        mom_data["Ri"] = R.squeeze()
        mom_data["Zi"] = Z.squeeze()
    

        raw_Ri = (mc2pdata.get_values()[:,:,0]/2).squeeze()
        raw_Zi = (mc2pdata.get_values()[:,:,4]/2).squeeze()
        raw_mom_data["Ri"] = raw_Ri
        raw_mom_data["Zi"] = raw_Zi

        #Get Plate angle information
        if isim == 1 : 
            plate_data = np.genfromtxt(sim_dir+"stepplate_data/highres/osol.txt", delimiter = ",")
            bimpactangledata = pg.GData(sim_dir+'%s-bimpactangle_dir1.gkyl'%sim_name).get_values()[:,0,0:2]
        if isim == 0 : 
            plate_data = np.genfromtxt(sim_dir+"stepplate_data/highres/opf.txt", delimiter = ",")
            bimpactangledata = pg.GData(sim_dir+'%s-bimpactangle_dir1.gkyl'%sim_name).get_values()[:,0,0:2]
        if isim == 4 : 
            plate_data = np.genfromtxt(sim_dir+"stepplate_data/highres/isol.txt", delimiter = ",")
            bimpactangledata = pg.GData(sim_dir+'%s-bimpactangle_dir1.gkyl'%sim_name).get_values()[:,-1,0:2]
        if isim == 5 : 
            plate_data = np.genfromtxt(sim_dir+"stepplate_data/highres/ipf.txt", delimiter = ",")
            bimpactangledata = pg.GData(sim_dir+'%s-bimpactangle_dir1.gkyl'%sim_name).get_values()[:,-1,0:2]

    
        #Load bflux data
        for species in ["elc", "ion"]:
            #for bdry in ["xlower", "xupper", "ylower", "yupper"]:
            for bdry in ["ylower", "yupper"]:
                if isim in [0,1,4,5]:
                    bflux_file = '%s-%s_bflux_%s_HamiltonianMoments_%d.gkyl'%(sim_name, species, bdry, frame)
                    if os.path.exists(bflux_file):
                        mdata = pg.GData(bflux_file)
                        coeffs = mdata.get_values()

                        lenrdata = pg.GData('%s-lenr_dir1.gkyl'%sim_name).get_values()
                        if bdry == "ylower":
                            lenrdata = lenrdata[:, 0, 0:2]
                        if bdry == "yupper":
                            lenrdata = lenrdata[:, -1, 0:2]
                        mom_data["lenr"] = interp_surface(lenrdata, 0)
                        #Factor of 2 is because of some error in the code
                        mom_data[species+'Q'+bdry] = interp_surface(coeffs, 2)/interp_surface(lenrdata, 0)/2
                        mom_data[species+'M1'+bdry] = interp_surface(coeffs, 0)/interp_surface(lenrdata, 0)/2

    
        #Load moment data
        for species in ["elc", "ion"]:
            for mom in ["M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp", "BGKM0dot", "BGKM1dot", "BGKM2dot"]:
                mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
                raw_mom_data[species+mom] = mdata.get_values()[:,:,0]/2
                raw_grid = mdata.get_grid()
                raw_x = fix_gridvals(raw_grid[0])
                raw_z = fix_gridvals(raw_grid[1])
                grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
                x = fix_gridvals(grid[0])
                z = fix_gridvals(grid[1])
                val = val.squeeze()
                mom_data[species+mom] = val
 
            #Set cell avg moment data
            raw_mom_data[species+"Temp"] =  (masses[species]/3) * (raw_mom_data[species+"M2"] - raw_mom_data[species+"M1"]**2 / raw_mom_data[species+"M0"])/raw_mom_data[species+"M0"] / eV
            raw_mom_data[species+"Tpar"] =  (masses[species]) * (raw_mom_data[species+"M2par"] - raw_mom_data[species+"M1"]**2 / raw_mom_data[species+"M0"])/raw_mom_data[species+"M0"] / eV
            raw_mom_data[species+"Tperp"] =  (masses[species]/2) * (raw_mom_data[species+"M2perp"])/raw_mom_data[species+"M0"] / eV
            raw_mom_data[species+"Q"] =  masses[species]/2 * (raw_mom_data[species+"M3par"] + raw_mom_data[species+"M3perp"])
            raw_mom_data[species+"Upar"] =  raw_mom_data[species+"M1"]/raw_mom_data[species+"M0"]

            # Set interpolated moment data
            mom_data[species+"Temp"] =  (masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tpar"] =  (masses[species]) * (mom_data[species+"M2par"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tperp"] =  (masses[species]/2) * (mom_data[species+"M2perp"])/mom_data[species+"M0"] / eV
            mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])
            mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
        mom_data["Qtot"] =  mom_data["elcQ"]+mom_data["ionQ"]
        raw_mom_data["Qtot"] =  raw_mom_data["elcQ"]+raw_mom_data["ionQ"]


        # Calculate sound speed for ion species
        for species in ["ion" ]:
            mom_data[species+"cs"] = np.sqrt( (mom_data["elcTemp"]*eV + mom_data[species+"Temp"]*eV) / masses[species])
            raw_mom_data[species+"cs"] = np.sqrt( (raw_mom_data["elcTemp"]*eV + raw_mom_data[species+"Temp"]*eV) / masses[species])
        
        for species in ["ion", "elc"]:
            mom_data[species+"normUpar"] = mom_data[species+"Upar"]/mom_data["ioncs"]
            raw_mom_data[species+"normUpar"] = raw_mom_data[species+"Upar"]/raw_mom_data["ioncs"]


        # Load the potential
        mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
        grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
        raw_mom_data["phi"] = mdata.get_values()[:,:,0]/2
        x = fix_gridvals(grid[0])
        z = fix_gridvals(grid[1])
        val = val.squeeze()
        mom_data["phi"] = val
        mom_data["time"] = float(mdata.info()[9:21])

        for species in ["elc", "ion"]:
            mom_data[species+"Qwall"] = mom_data[species+"Q"] + charges[species]*mom_data[species+"M1"]*mom_data["phi"]
            raw_mom_data[species+"Qwall"] = raw_mom_data[species+"Q"] + charges[species]*raw_mom_data[species+"M1"]*raw_mom_data["phi"]



        # Interpolate B ratio at plates
        if isim in [1,0,4,5]:
            #Binterpolator = interp1d(plate_data[:,0], plate_data[:,1])
            #Bratio = Binterpolator(x)
            #mom_data["Bratio" ] = Bratio
            mom_data["Bratio" ] = interp_surface(bimpactangledata, 0)


        #Save grids
        mom_data["x"] = x
        mom_data["z"] = z

        raw_mom_data["x"] = raw_x
        raw_mom_data["z"] = raw_z

        raw_mom_data["rawx"] = raw_grid[0]
        raw_mom_data["rawz"] = raw_grid[1]

        #Load source moment data
        source_mom_data = {}
        if (isim in [6,7]):
            for species in ["elc", "ion"]:
                for mom in ["M0", "M1", "M2", "M2par", "M2perp"]:
                    source_path = '%s-%s_source_%s_%d.gkyl'%(sim_name, species,mom,frame)
                    if not os.path.exists(source_path):
                        source_path = '%s-%s_source_%s_%d.gkyl'%(sim_name, species,mom,source_frame)
                    mdata = pg.GData(source_path)
                    grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
                    x = fix_gridvals(grid[0])
                    z = fix_gridvals(grid[1])
                    val = val.squeeze()
                    source_mom_data[species+mom] = val
                # Set interpolated moment data
                source_mom_data[species+"Temp"] =  (masses[species]/3) * (source_mom_data[species+"M2"] - source_mom_data[species+"M1"]**2 / source_mom_data[species+"M0"])/source_mom_data[species+"M0"] / eV
                source_mom_data[species+"Tpar"] =  (masses[species]) * (source_mom_data[species+"M2par"] - source_mom_data[species+"M1"]**2 / source_mom_data[species+"M0"])/source_mom_data[species+"M0"] / eV
                source_mom_data[species+"Tperp"] =  (masses[species]/2) * (source_mom_data[species+"M2perp"])/source_mom_data[species+"M0"] / eV
                source_mom_data[species+"Upar"] =  source_mom_data[species+"M1"]/source_mom_data[species+"M0"]
    

            ion_source_integrand = masses["ion"]/2 *(source_mom_data["ionM2"]) * mom_data["J"]
            elc_source_integrand = masses["elc"]/2 *(source_mom_data["elcM2"]) * mom_data["J"]

            ion_sp = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            elc_sp = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            source_mom_data["ionsourcepower"] = ion_sp
            source_mom_data["elcsourcepower"] = elc_sp

            ion_source_integrand = (source_mom_data["ionM0"]) * mom_data["J"]
            elc_source_integrand = (source_mom_data["elcM0"]) * mom_data["J"]

            ion_sn = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            elc_sn = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
            source_mom_data["ionsourcedensity"] = ion_sn
            source_mom_data["elcsourcedensity"] = elc_sn

        #Calculated integrated sources from Eirene
        ion_source_integrand = masses["ion"]/2 *(mom_data["ionBGKM2dot"]) * mom_data["J"]
        elc_source_integrand = masses["elc"]/2 *(mom_data["elcBGKM2dot"]) * mom_data["J"]

        ion_sp = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        elc_sp = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        source_mom_data["BGKionsourcepower"] = ion_sp
        source_mom_data["BGKelcsourcepower"] = elc_sp

        ion_source_integrand = (mom_data["ionBGKM0dot"]) * mom_data["J"]
        elc_source_integrand = (mom_data["elcBGKM0dot"]) * mom_data["J"]

        ion_sn = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        elc_sn = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        source_mom_data["BGKionsourcedensity"] = ion_sn
        source_mom_data["BGKelcsourcedensity"] = elc_sn

        ion_source_integrand = masses["ion"] *(mom_data["ionBGKM1dot"]) * mom_data["J"]
        ion_sm = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        source_mom_data["BGKionsourcemomentum"] = ion_sm
    
    
    
        # Append data
        half_mom_data_list.append(mom_data)
        raw_half_mom_data_list.append(raw_mom_data)
        source_half_mom_data_list.append(source_mom_data)

    total_ion_bgk_sp = 0
    total_elc_bgk_sp = 0
    total_ion_bgk_sn = 0
    total_elc_bgk_sn = 0
    total_ion_bgk_sm = 0
    for ib in range(8):
        total_ion_bgk_sp += source_half_mom_data_list[ib]["BGKionsourcepower"]
        total_elc_bgk_sp += source_half_mom_data_list[ib]["BGKelcsourcepower"]
        total_ion_bgk_sn += source_half_mom_data_list[ib]["BGKionsourcedensity"]
        total_elc_bgk_sn += source_half_mom_data_list[ib]["BGKelcsourcedensity"]
        total_ion_bgk_sm += source_half_mom_data_list[ib]["BGKionsourcemomentum"]

    total_sp = (source_half_mom_data_list[6]["ionsourcepower"] + source_half_mom_data_list[6]["elcsourcepower"] + source_half_mom_data_list[7]["ionsourcepower"] + source_half_mom_data_list[7]["elcsourcepower"])*2
    total_sn = (source_half_mom_data_list[6]["ionsourcedensity"] + source_half_mom_data_list[6]["elcsourcedensity"] + source_half_mom_data_list[7]["ionsourcedensity"] + source_half_mom_data_list[7]["elcsourcedensity"])*2
    print(" Total (doubled) source power = MW", total_sp/1e6)
    print(" Total (doubled) source density = ", total_sn)

    # Construct the 12 block data
    mom_data_list = [None]*12
    raw_mom_data_list = [None]*12
    even_keys = [["elc", "ion" ][j] + ["M0", "M2", "Temp", "Tpar", "Tperp", "BGKM0dot", "BGKM1dot", "BGKM2dot"][i] for i in range(8) for j in range(2)] + ["phi" , "gxx", "gzz", "Ri", "J", "B"]
    odd_keys = [["elc", "ion"][j] + ["M1", "Upar", "normUpar", "Q", "Qwall"][i] for i in range(5) for j in range(2)] + ["Zi", "Qtot"]
    bflux_keys = []
    for species in ["elc", "ion"]:
        for bdry in ["ylower", "yupper"]:
            bflux_keys.append(species+'Q'+bdry)
            bflux_keys.append(species+'M1'+bdry)
    bflux_keys.append("lenr")

    #Unmodified but renumbered blocks
    mom_data_list[0] = half_mom_data_list[0]
    mom_data_list[1] = half_mom_data_list[1]
    mom_data_list[8] = half_mom_data_list[4]
    mom_data_list[9] = half_mom_data_list[5]

    raw_mom_data_list[0] = raw_half_mom_data_list[0]
    raw_mom_data_list[1] = raw_half_mom_data_list[1]
    raw_mom_data_list[8] = raw_half_mom_data_list[4]
    raw_mom_data_list[9] = raw_half_mom_data_list[5]
    
    
    # Reflected blocks
    mom_data_list[3] = {}
    mom_data_list[6] = {}
    mom_data_list[4] = {}
    mom_data_list[5] = {}
    raw_mom_data_list[3] = {}
    raw_mom_data_list[6] = {}
    raw_mom_data_list[4] = {}
    raw_mom_data_list[5] = {}
    #Do Bratio myself manually
    mom_data_list[3]["Bratio"] = mom_data_list[1]["Bratio"]
    mom_data_list[6]["Bratio"] = mom_data_list[8]["Bratio"]
    mom_data_list[4]["Bratio"] = mom_data_list[0]["Bratio"]
    mom_data_list[5]["Bratio"] = mom_data_list[9]["Bratio"]
    raw_mom_data_list[3]["Bratio"] = mom_data_list[1]["Bratio"]
    raw_mom_data_list[6]["Bratio"] = mom_data_list[8]["Bratio"]
    raw_mom_data_list[4]["Bratio"] = mom_data_list[0]["Bratio"]
    raw_mom_data_list[5]["Bratio"] = mom_data_list[9]["Bratio"]
    
    
    
    # Doubled blocks
    mom_data_list[2] = {}
    mom_data_list[7] = {}
    mom_data_list[10] = {}
    mom_data_list[11] = {}
    raw_mom_data_list[2] = {}
    raw_mom_data_list[7] = {}
    raw_mom_data_list[10] = {}
    raw_mom_data_list[11] = {}
    for key in even_keys:
        mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1]))
        mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1]))
        mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1]))
        mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1]))

        raw_mom_data_list[2][key] = np.zeros((raw_half_mom_data_list[2]["ionM0"].shape[0], 2*raw_half_mom_data_list[2]["ionM0"].shape[1]))
        raw_mom_data_list[7][key] = np.zeros((raw_half_mom_data_list[3]["ionM0"].shape[0], 2*raw_half_mom_data_list[3]["ionM0"].shape[1]))
        raw_mom_data_list[10][key] = np.zeros((raw_half_mom_data_list[6]["ionM0"].shape[0], 2*raw_half_mom_data_list[6]["ionM0"].shape[1]))
        raw_mom_data_list[11][key] = np.zeros((raw_half_mom_data_list[7]["ionM0"].shape[0], 2*raw_half_mom_data_list[7]["ionM0"].shape[1]))
    
    for key in odd_keys:
        mom_data_list[2][key] = np.zeros((half_mom_data_list[2]["ionM0"].shape[0], 2*half_mom_data_list[2]["ionM0"].shape[1]))
        mom_data_list[7][key] = np.zeros((half_mom_data_list[3]["ionM0"].shape[0], 2*half_mom_data_list[3]["ionM0"].shape[1]))
        mom_data_list[10][key] = np.zeros((half_mom_data_list[6]["ionM0"].shape[0], 2*half_mom_data_list[6]["ionM0"].shape[1]))
        mom_data_list[11][key] = np.zeros((half_mom_data_list[7]["ionM0"].shape[0], 2*half_mom_data_list[7]["ionM0"].shape[1]))

        raw_mom_data_list[2][key] = np.zeros((raw_half_mom_data_list[2]["ionM0"].shape[0], 2*raw_half_mom_data_list[2]["ionM0"].shape[1]))
        raw_mom_data_list[7][key] = np.zeros((raw_half_mom_data_list[3]["ionM0"].shape[0], 2*raw_half_mom_data_list[3]["ionM0"].shape[1]))
        raw_mom_data_list[10][key] = np.zeros((raw_half_mom_data_list[6]["ionM0"].shape[0], 2*raw_half_mom_data_list[6]["ionM0"].shape[1]))
        raw_mom_data_list[11][key] = np.zeros((raw_half_mom_data_list[7]["ionM0"].shape[0], 2*raw_half_mom_data_list[7]["ionM0"].shape[1]))

    
    #Do x myself manually
    mom_data_list[3]["x"] = mom_data_list[1]["x"]
    mom_data_list[6]["x"] = mom_data_list[8]["x"]
    mom_data_list[4]["x"] = mom_data_list[0]["x"]
    mom_data_list[5]["x"] = mom_data_list[9]["x"]
    raw_mom_data_list[3]["x"] = raw_mom_data_list[1]["x"]
    raw_mom_data_list[6]["x"] = raw_mom_data_list[8]["x"]
    raw_mom_data_list[4]["x"] = raw_mom_data_list[0]["x"]
    raw_mom_data_list[5]["x"] = raw_mom_data_list[9]["x"]
    
    
    mom_data_list[2]["x"] = mom_data_list[1]["x"]
    mom_data_list[7]["x"] = mom_data_list[8]["x"]
    mom_data_list[10]["x"] = half_mom_data_list[6]["x"]
    mom_data_list[11]["x"] = half_mom_data_list[7]["x"]
    raw_mom_data_list[2]["x"] = raw_mom_data_list[1]["x"]
    raw_mom_data_list[7]["x"] = raw_mom_data_list[8]["x"]
    raw_mom_data_list[10]["x"] = raw_half_mom_data_list[6]["x"]
    raw_mom_data_list[11]["x"] = raw_half_mom_data_list[7]["x"]

    #Do z myself manually
    mom_data_list[3]["z"] = -np.flip(mom_data_list[1]["z"])
    mom_data_list[6]["z"] = -np.flip(mom_data_list[8]["z"])
    mom_data_list[4]["z"] = np.flip(mom_data_list[0]["z"])
    mom_data_list[5]["z"] = np.flip(mom_data_list[9]["z"])

    raw_mom_data_list[3]["z"] = -np.flip(raw_mom_data_list[1]["z"])
    raw_mom_data_list[6]["z"] = -np.flip(raw_mom_data_list[8]["z"])
    raw_mom_data_list[4]["z"] = np.flip(raw_mom_data_list[0]["z"])
    raw_mom_data_list[5]["z"] = np.flip(raw_mom_data_list[9]["z"])
    
    mom_data_list[2]["z"] = np.r_[half_mom_data_list[2]["z"], -np.flip(half_mom_data_list[2]["z"])]
    mom_data_list[7]["z"] = np.r_[-np.flip(half_mom_data_list[3]["z"]), half_mom_data_list[3]["z"]]
    mom_data_list[10]["z"] = np.array([half_mom_data_list[6]["z"][0] + np.diff(half_mom_data_list[6]["z"])[0]*i for i in range(2*len(half_mom_data_list[6]["z"]))])
    mom_data_list[11]["z"] = np.flip(np.array([half_mom_data_list[7]["z"][-1] - np.diff(half_mom_data_list[7]["z"])[0]*i for i in range(2*len(half_mom_data_list[7]["z"]))]))

    raw_mom_data_list[2]["z"] = np.r_[raw_half_mom_data_list[2]["z"], -np.flip(raw_half_mom_data_list[2]["z"])]
    raw_mom_data_list[7]["z"] = np.r_[-np.flip(raw_half_mom_data_list[3]["z"]), raw_half_mom_data_list[3]["z"]]
    raw_mom_data_list[10]["z"] = np.array([raw_half_mom_data_list[6]["z"][0] + np.diff(raw_half_mom_data_list[6]["z"])[0]*i for i in range(2*len(raw_half_mom_data_list[6]["z"]))])
    raw_mom_data_list[11]["z"] = np.flip(np.array([raw_half_mom_data_list[7]["z"][-1] - np.diff(raw_half_mom_data_list[7]["z"])[0]*i for i in range(2*len(raw_half_mom_data_list[7]["z"]))]))
    
    
    
    # Flip some and double/reflect some
    for key in even_keys:
        #Upper SOL
        mom_data_list[3][key] = np.flip(mom_data_list[1][key], axis=-1)
        mom_data_list[6][key] = np.flip(mom_data_list[8][key], axis=-1)
        raw_mom_data_list[3][key] = np.flip(raw_mom_data_list[1][key], axis=-1)
        raw_mom_data_list[6][key] = np.flip(raw_mom_data_list[8][key], axis=-1)
        #Upper PF
        mom_data_list[4][key] = np.flip(mom_data_list[0][key], axis=-1)
        mom_data_list[5][key] = np.flip(mom_data_list[9][key], axis=-1)
        raw_mom_data_list[4][key] = np.flip(raw_mom_data_list[0][key], axis=-1)
        raw_mom_data_list[5][key] = np.flip(raw_mom_data_list[9][key], axis=-1)
    
        #Outer Middle
        mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
        mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = np.flip(half_mom_data_list[2][key], axis=-1)
        raw_mom_data_list[2][key][:, 0:raw_mom_data_list[2][key].shape[1]//2] = raw_half_mom_data_list[2][key] 
        raw_mom_data_list[2][key][:, raw_mom_data_list[2][key].shape[1]//2:] = np.flip(raw_half_mom_data_list[2][key], axis=-1)
    
        mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
        mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = np.flip(half_mom_data_list[6][key], axis=-1)
        raw_mom_data_list[10][key][:, 0:raw_mom_data_list[10][key].shape[1]//2] = raw_half_mom_data_list[6][key] 
        raw_mom_data_list[10][key][:, raw_mom_data_list[10][key].shape[1]//2:] = np.flip(raw_half_mom_data_list[6][key], axis=-1)
    
        #Inner Middle
        mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = np.flip(half_mom_data_list[3][key], axis=-1) 
        mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
        raw_mom_data_list[7][key][:, 0:raw_mom_data_list[7][key].shape[1]//2] = np.flip(raw_half_mom_data_list[3][key], axis=-1) 
        raw_mom_data_list[7][key][:, raw_mom_data_list[7][key].shape[1]//2:] = raw_half_mom_data_list[3][key]
    
        mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = np.flip(half_mom_data_list[7][key], axis=-1) 
        mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
        raw_mom_data_list[11][key][:, 0:raw_mom_data_list[11][key].shape[1]//2] = np.flip(raw_half_mom_data_list[7][key], axis=-1) 
        raw_mom_data_list[11][key][:, raw_mom_data_list[11][key].shape[1]//2:] = raw_half_mom_data_list[7][key]
    
    for key in odd_keys:
        #Upper SOL
        mom_data_list[3][key] = -np.flip(mom_data_list[1][key], axis=-1)
        mom_data_list[6][key] = -np.flip(mom_data_list[8][key], axis=-1)
        raw_mom_data_list[3][key] = -np.flip(raw_mom_data_list[1][key], axis=-1)
        raw_mom_data_list[6][key] = -np.flip(raw_mom_data_list[8][key], axis=-1)
        #Upper PF
        mom_data_list[4][key] = -np.flip(mom_data_list[0][key], axis=-1)
        mom_data_list[5][key] = -np.flip(mom_data_list[9][key], axis=-1)
        raw_mom_data_list[4][key] = -np.flip(raw_mom_data_list[0][key], axis=-1)
        raw_mom_data_list[5][key] = -np.flip(raw_mom_data_list[9][key], axis=-1)
    
        #Outer Middle
        mom_data_list[2][key][:, 0:mom_data_list[2][key].shape[1]//2] = half_mom_data_list[2][key] 
        mom_data_list[2][key][:, mom_data_list[2][key].shape[1]//2:] = -np.flip(half_mom_data_list[2][key], axis=-1)
        raw_mom_data_list[2][key][:, 0:raw_mom_data_list[2][key].shape[1]//2] = raw_half_mom_data_list[2][key] 
        raw_mom_data_list[2][key][:, raw_mom_data_list[2][key].shape[1]//2:] = -np.flip(raw_half_mom_data_list[2][key], axis=-1)
    
        mom_data_list[10][key][:, 0:mom_data_list[10][key].shape[1]//2] = half_mom_data_list[6][key] 
        mom_data_list[10][key][:, mom_data_list[10][key].shape[1]//2:] = -np.flip(half_mom_data_list[6][key], axis=-1)
        raw_mom_data_list[10][key][:, 0:raw_mom_data_list[10][key].shape[1]//2] = raw_half_mom_data_list[6][key] 
        raw_mom_data_list[10][key][:, raw_mom_data_list[10][key].shape[1]//2:] = -np.flip(raw_half_mom_data_list[6][key], axis=-1)
    
        #Inner Middle
        mom_data_list[7][key][:, 0:mom_data_list[7][key].shape[1]//2] = -np.flip(half_mom_data_list[3][key], axis=-1) 
        mom_data_list[7][key][:, mom_data_list[7][key].shape[1]//2:] = half_mom_data_list[3][key]
        raw_mom_data_list[7][key][:, 0:raw_mom_data_list[7][key].shape[1]//2] = -np.flip(raw_half_mom_data_list[3][key], axis=-1) 
        raw_mom_data_list[7][key][:, raw_mom_data_list[7][key].shape[1]//2:] = raw_half_mom_data_list[3][key]
    
        mom_data_list[11][key][:, 0:mom_data_list[11][key].shape[1]//2] = -np.flip(half_mom_data_list[7][key], axis=-1) 
        mom_data_list[11][key][:, mom_data_list[11][key].shape[1]//2:] = half_mom_data_list[7][key]
        raw_mom_data_list[11][key][:, 0:raw_mom_data_list[11][key].shape[1]//2] = -np.flip(raw_half_mom_data_list[7][key], axis=-1) 
        raw_mom_data_list[11][key][:, raw_mom_data_list[11][key].shape[1]//2:] = raw_half_mom_data_list[7][key]


    #Flip bflux blocks:
    for key in bflux_keys:
        if 'lower' in key:
            new_key = key.replace('lower','upper')
        elif 'upper' in key:
            new_key = key.replace('upper','lower')
        else:
            new_key = key
        #Upper SOL
        mom_data_list[3][new_key] = mom_data_list[1][key] if key in mom_data_list[1].keys() else None
        mom_data_list[6][new_key] = mom_data_list[8][key] if key in mom_data_list[8].keys() else None
        #Upper PF
        mom_data_list[4][new_key] = mom_data_list[0][key] if key in mom_data_list[0].keys() else None
        mom_data_list[5][new_key] = mom_data_list[9][key] if key in mom_data_list[9].keys() else None
    
    xidx = -1
    xidx_core=0
    zidx = [mom_data_list[i]["Zi"].shape[1]//2 for i in range(12)]
    zidx_xpt = [0,0,0]
    Rmid_out = mom_data_list[2]["Ri"][:,zidx[2]]
    Rmid_in= mom_data_list[7]["Ri"][:,zidx[7]]
    
    #Now let us set up data for plotting
    raw_keys = [["elc", "ion"][j] + ["M0", "Temp", "Tpar", "Tperp", "Q", "normUpar", "M1"][i] for i in range(7) for j in range(2)] + ["phi"]
    keys = ["z"]

    # Construct the concatenated parallel data
    raw_outboard_data = {}
    outboard_data = {}
    for raw_key in raw_keys : 
        raw_outboard_data[raw_key] = np.column_stack((raw_mom_data_list[1][raw_key], raw_mom_data_list[2][raw_key], raw_mom_data_list[3][raw_key]))
    for key in keys : 
        outboard_data[key] = np.concatenate((mom_data_list[1][key], mom_data_list[2][key], mom_data_list[3][key]))
        raw_outboard_data[key] = np.concatenate((raw_mom_data_list[1][key], raw_mom_data_list[2][key], raw_mom_data_list[3][key]))
    for ikey in raw_keys:
        outboard_data[ikey] = myinterpolate(raw_mom_data_list[1]["x"], raw_outboard_data["z"], mom_data_list[1]["x"], outboard_data["z"], raw_outboard_data[ikey])
    outboard_data["Zi"] = np.column_stack((mom_data_list[1]["Zi"], mom_data_list[2]["Zi"], mom_data_list[3]["Zi"]))

    raw_inboard_data = {}
    inboard_data = {}
    for raw_key in raw_keys : 
        raw_inboard_data[raw_key] = np.column_stack((raw_mom_data_list[6][raw_key], raw_mom_data_list[7][raw_key], raw_mom_data_list[8][raw_key]))
    for key in keys : 
        inboard_data[key] = np.concatenate((mom_data_list[6][key], mom_data_list[7][key], mom_data_list[8][key]))
        raw_inboard_data[key] = np.concatenate((raw_mom_data_list[6][key], raw_mom_data_list[7][key], raw_mom_data_list[8][key]))
    for ikey in raw_keys:
        inboard_data[ikey] = myinterpolate(raw_mom_data_list[6]["x"], raw_inboard_data["z"], mom_data_list[6]["x"], inboard_data["z"], raw_inboard_data[ikey])
    inboard_data["Zi"] = np.column_stack((mom_data_list[6]["Zi"], mom_data_list[7]["Zi"], mom_data_list[8]["Zi"]))

    raw_core_data = {}
    core_data = {}
    for raw_key in raw_keys : 
        raw_core_data[raw_key] = np.column_stack((raw_mom_data_list[10][raw_key], raw_mom_data_list[11][raw_key]))
    for key in keys : 
        core_data[key] = np.concatenate((mom_data_list[10][key], mom_data_list[11][key]))
        raw_core_data[key] = np.concatenate((raw_mom_data_list[10][key], raw_mom_data_list[11][key]))
    for ikey in raw_keys:
        core_data[ikey] = myinterpolate(raw_mom_data_list[10]["x"], raw_core_data["z"], mom_data_list[10]["x"], core_data["z"], raw_core_data[ikey])
        #core_data[ikey] = np.column_stack((mom_data_list[10][ikey], mom_data_list[11][ikey]))
    core_data["Zi"] = np.column_stack((mom_data_list[10]["Zi"], mom_data_list[11]["Zi"]))

    # Set Up flux surface mapping to OMP for outboard leg plotting
    OMPinterpolator = interp1d(np.r_[mom_data_list[2]["x"], mom_data_list[10]["x"]], np.r_[mom_data_list[2]["Ri"][:, zidx[2]], mom_data_list[10]["Ri"][:, zidx[10]]])

    IMPinterpolator = interp1d(np.r_[mom_data_list[7]["x"], mom_data_list[11]["x"]], np.r_[mom_data_list[7]["Ri"][:, zidx[7]], mom_data_list[11]["Ri"][:, zidx[11]]])




    sim_data = {}
    sim_data['name'] = str(sys.argv[1])
    sim_data['R_OMP_sep'] = mom_data_list[2]["Ri"][-1, zidx[2]]
    sim_data['R_OP_sep'] = mom_data_list[3]["Ri"][-1, -1]

    sim_data['R_OMP'] = np.r_[mom_data_list[2]["Ri"][:, zidx[2]],mom_data_list[10]["Ri"][:, zidx[10]]]
    sim_data['Ti_OMP'] = np.r_[mom_data_list[2]["ionTemp"][:, zidx[2]],mom_data_list[10]["ionTemp"][:, zidx[10]]]
    sim_data['Te_OMP'] = np.r_[mom_data_list[2]["elcTemp"][:, zidx[2]],mom_data_list[10]["elcTemp"][:, zidx[10]]]
    sim_data['ne_OMP'] = np.r_[mom_data_list[2]["elcM0"][:, zidx[2]],mom_data_list[10]["elcM0"][:, zidx[10]]]
    sim_data['ni_OMP'] = np.r_[mom_data_list[2]["ionM0"][:, zidx[2]],mom_data_list[10]["ionM0"][:, zidx[10]]]

    sim_data['R_OP'] = np.r_[mom_data_list[3]["Ri"][:, -1],mom_data_list[4]["Ri"][:, -1]]
    sim_data['Ti_OP'] = np.r_[mom_data_list[3]["ionTemp"][:, -1],mom_data_list[4]["ionTemp"][:, -1]]
    sim_data['Te_OP'] = np.r_[mom_data_list[3]["elcTemp"][:, -1],mom_data_list[4]["elcTemp"][:, -1]]
    sim_data['ne_OP'] = np.r_[mom_data_list[3]["elcM0"][:, -1],mom_data_list[4]["elcM0"][:, -1]]
    sim_data['ni_OP'] = np.r_[mom_data_list[3]["ionM0"][:, -1],mom_data_list[4]["ionM0"][:, -1]]
    sim_data['Qe_OP'] = np.r_[mom_data_list[3]["elcQyupper"],mom_data_list[4]["elcQyupper"]]
    sim_data['Qi_OP'] = np.r_[mom_data_list[3]["ionQyupper"],mom_data_list[4]["ionQyupper"]]


    sim_data['Z_IP'] = np.r_[mom_data_list[8]["Zi"][:, -1],mom_data_list[9]["Zi"][:, -1]]
    sim_data['Z_IP_sep'] = mom_data_list[8]["Zi"][-1, -1]
    sim_data['Ti_IP'] = np.r_[mom_data_list[8]["ionTemp"][:, -1],mom_data_list[9]["ionTemp"][:, -1]]
    sim_data['Te_IP'] = np.r_[mom_data_list[8]["elcTemp"][:, -1],mom_data_list[9]["elcTemp"][:, -1]]
    sim_data['ne_IP'] = np.r_[mom_data_list[8]["elcM0"][:, -1],mom_data_list[9]["elcM0"][:, -1]]
    sim_data['ni_IP'] = np.r_[mom_data_list[8]["ionM0"][:, -1],mom_data_list[9]["ionM0"][:, -1]]
    sim_data['Qe_IP'] = np.r_[mom_data_list[8]["elcQyupper"],mom_data_list[9]["elcQyupper"]]
    sim_data['Qi_IP'] = np.r_[mom_data_list[8]["ionQyupper"],mom_data_list[9]["ionQyupper"]]

