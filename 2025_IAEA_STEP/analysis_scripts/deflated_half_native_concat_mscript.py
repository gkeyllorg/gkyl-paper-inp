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
masses["molecule"] = 4.028*mp


charges = {}
charges["elc"] = -eV 
charges["ion"] = eV
charges["molecule"] = eV


plot_divertor = True
plot_leg = False
# Set up names for data loading
sim_dir = "./"
#sim_dir = "./simdata/hstep26/"
base_name = sim_dir+'hstep26'
bmin, bmax = 0, 8
sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]

sim_labels= ['b%d'%i for i in range(bmin,bmax)]


#frame = 1
start= int(sys.argv[1])
neut_frame = start
end_frame = start+1
#end_frame=95

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
            plate_data = np.genfromtxt("./stepplate_data/highres/osol.txt", delimiter = ",")
        if isim == 0 : 
            plate_data = np.genfromtxt("./stepplate_data/highres/opf.txt", delimiter = ",")
        if isim == 4 : 
            plate_data = np.genfromtxt("./stepplate_data/highres/isol.txt", delimiter = ",")
        if isim == 5 : 
            plate_data = np.genfromtxt("./stepplate_data/highres/ipf.txt", delimiter = ",")

    
        #Load bflux data
        for species in ["elc", "ion", "molecule"]:
            #for bdry in ["xlower", "xupper", "ylower", "yupper"]:
            for bdry in ["ylower", "yupper"]:
                if isim in [0,1,4,5]:
                    bflux_file = '%s-%s_bflux_%s_HamiltonianMoments_%d.gkyl'%(sim_name, species, bdry, frame)
                    if os.path.exists(bflux_file):
                        mdata = pg.GData(bflux_file)
                        coeffs = mdata.get_values()
                        mom_data[species+'Q'+bdry] = interp_surface(coeffs, 2)
                        mom_data[species+'M1'+bdry] = interp_surface(coeffs, 0)

    
        #Load moment data
        for species in ["elc", "ion", "molecule"]:
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

            #Try interpolated M1 and Q
            #M1interpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"M1"], bounds_error=False, fill_value=None)
            #x0,z0 = np.meshgrid(x,z)
            #mom_data[species+"M1"] = M1interpolator((x0,z0)).T
            #Qinterpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"Q"], bounds_error=False, fill_value=None)
            #x0,z0 = np.meshgrid(x,z)
            #mom_data[species+"Q"] = Qinterpolator((x0,z0)).T
            #mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])


            #Try interpolated M0
            #M0interpolator = RegularGridInterpolator((raw_x,raw_z), raw_mom_data[species+"M0"], bounds_error=False, fill_value=None)
            #x0,z0 = np.meshgrid(x,z)
            #mom_data[species+"M0"] = M0interpolator((x0,z0)).T

            #use interpolating function
            #mom_data[species+"M1"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"M1"])
            #mom_data[species+"M0"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"M0"])
            #mom_data[species+"Q"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Q"])
            #mom_data[species+"Temp"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Temp"])
            #mom_data[species+"Tpar"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Tpar"])
            #mom_data[species+"Tperp"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Tperp"])
            #mom_data[species+"Upar"] = myinterpolate(raw_x,raw_z,x,z,raw_mom_data[species+"Upar"])

            # Set interpolated moment data
            mom_data[species+"Temp"] =  (masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tpar"] =  (masses[species]) * (mom_data[species+"M2par"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
            mom_data[species+"Tperp"] =  (masses[species]/2) * (mom_data[species+"M2perp"])/mom_data[species+"M0"] / eV
            mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])
            mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
        
        mom_data["Qtot"] =  mom_data["elcQ"]+mom_data["ionQ"]+mom_data["moleculeQ"]
        raw_mom_data["Qtot"] =  raw_mom_data["elcQ"]+raw_mom_data["ionQ"]+raw_mom_data["moleculeQ"]


        # Calculate sound speed for ion species
        for species in ["ion" ]:
            mom_data[species+"cs"] = np.sqrt( (mom_data["elcTemp"]*eV + mom_data[species+"Temp"]*eV) / masses[species])
            raw_mom_data[species+"cs"] = np.sqrt( (raw_mom_data["elcTemp"]*eV + raw_mom_data[species+"Temp"]*eV) / masses[species])
        
        for species in ["ion", "elc", "molecule" ]:
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

        for species in ["elc", "ion", "molecule"]:
            mom_data[species+"Qwall"] = mom_data[species+"Q"] + charges[species]*mom_data[species+"M1"]*mom_data["phi"]
            raw_mom_data[species+"Qwall"] = raw_mom_data[species+"Q"] + charges[species]*raw_mom_data[species+"M1"]*raw_mom_data["phi"]



        # Interpolate B ratio at plates
        if isim in [1,0,4,5]:
            Binterpolator = interp1d(plate_data[:,0], plate_data[:,1])
            Bratio = Binterpolator(x)
            mom_data["Bratio" ] = Bratio


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
                        source_path = '%s-%s_source_%s_%d.gkyl'%(sim_name, species,mom,1196)
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
        molecule_source_integrand = masses["molecule"]/2 *(mom_data["moleculeBGKM2dot"]) * mom_data["J"]
        elc_source_integrand = masses["elc"]/2 *(mom_data["elcBGKM2dot"]) * mom_data["J"]

        ion_sp = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        molecule_sp = np.sum(molecule_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        elc_sp = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        source_mom_data["BGKionsourcepower"] = ion_sp
        source_mom_data["BGKmoleculesourcepower"] = molecule_sp
        source_mom_data["BGKelcsourcepower"] = elc_sp

        ion_source_integrand = (mom_data["ionBGKM0dot"]) * mom_data["J"]
        molecule_source_integrand = (mom_data["moleculeBGKM0dot"]) * mom_data["J"]
        elc_source_integrand = (mom_data["elcBGKM0dot"]) * mom_data["J"]

        ion_sn = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        molecule_sn = np.sum(molecule_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        elc_sn = np.sum(elc_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        source_mom_data["BGKionsourcedensity"] = ion_sn
        source_mom_data["BGKmoleculesourcedensity"] = molecule_sn
        source_mom_data["BGKelcsourcedensity"] = elc_sn

        ion_source_integrand = masses["ion"] *(mom_data["ionBGKM1dot"]) * mom_data["J"]
        ion_sm = np.sum(ion_source_integrand*np.diff(x)[0]*np.diff(z)[0]) * 2 * np.pi
        source_mom_data["BGKionsourcemomentum"] = ion_sm
    
    
    
        # Append data
        half_mom_data_list.append(mom_data)
        raw_half_mom_data_list.append(raw_mom_data)
        source_half_mom_data_list.append(source_mom_data)

    total_ion_bgk_sp = 0
    total_molecule_bgk_sp = 0
    total_elc_bgk_sp = 0
    total_ion_bgk_sn = 0
    total_molecule_bgk_sn = 0
    total_elc_bgk_sn = 0
    total_ion_bgk_sm = 0
    for ib in range(8):
        total_ion_bgk_sp += source_half_mom_data_list[ib]["BGKionsourcepower"]
        total_molecule_bgk_sp += source_half_mom_data_list[ib]["BGKmoleculesourcepower"]
        total_elc_bgk_sp += source_half_mom_data_list[ib]["BGKelcsourcepower"]
        total_ion_bgk_sn += source_half_mom_data_list[ib]["BGKionsourcedensity"]
        total_molecule_bgk_sn += source_half_mom_data_list[ib]["BGKmoleculesourcedensity"]
        total_elc_bgk_sn += source_half_mom_data_list[ib]["BGKelcsourcedensity"]
        total_ion_bgk_sm += source_half_mom_data_list[ib]["BGKionsourcemomentum"]

    total_sp = (source_half_mom_data_list[6]["ionsourcepower"] + source_half_mom_data_list[6]["elcsourcepower"] + source_half_mom_data_list[7]["ionsourcepower"] + source_half_mom_data_list[7]["elcsourcepower"])*2
    total_sn = (source_half_mom_data_list[6]["ionsourcedensity"] + source_half_mom_data_list[6]["elcsourcedensity"] + source_half_mom_data_list[7]["ionsourcedensity"] + source_half_mom_data_list[7]["elcsourcedensity"])*2
    print(" Total (doubled) source power = MW", total_sp/1e6)
    print(" Total (doubled) source density = ", total_sn)

    # Construct the 12 block data
    mom_data_list = [None]*12
    raw_mom_data_list = [None]*12
    even_keys = [["elc", "ion", "molecule"][j] + ["M0", "M2", "Temp", "Tpar", "Tperp", "BGKM0dot", "BGKM1dot", "BGKM2dot"][i] for i in range(8) for j in range(3)] + ["phi" , "gxx", "gzz", "Ri", "J", "B"]
    odd_keys = [["elc", "ion", "molecule"][j] + ["M1", "Upar", "normUpar", "Q", "Qwall"][i] for i in range(5) for j in range(3)] + ["Zi", "Qtot"]
    bflux_keys = []
    for species in ["elc", "ion", "molecule"]:
        for bdry in ["ylower", "yupper"]:
            bflux_keys.append(species+'Q'+bdry)
            bflux_keys.append(species+'M1'+bdry)

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
        if 'upper' in key:
            new_key = key.replace('upper','lower')
        #Upper SOL
        mom_data_list[3][new_key] = mom_data_list[1][key] if key in mom_data_list[1].keys() else None
        mom_data_list[6][new_key] = mom_data_list[8][key] if key in mom_data_list[8].keys() else None
        #Upper PF
        mom_data_list[4][new_key] = mom_data_list[0][key] if key in mom_data_list[0].keys() else None
        mom_data_list[5][new_key] = mom_data_list[9][key] if key in mom_data_list[9].keys() else None
    






    
    #Print input power and density stats:
    #print("ion source power = %e, elc source power = %e"%(source_mom_data_list[10]["ionsourcepower"] + source_mom_data_list[11]["ionsourcepower"], source_mom_data_list[10]["elcsourcepower"]+ source_mom_data_list[11]["elcsourcepower"]))
    #print("ion source density = %e, elc source density = %e"%(source_mom_data_list[10]["ionsourcedensity"] + source_mom_data_list[11]["ionsourcedensity"], source_mom_data_list[10]["elcsourcedensity"]+ source_mom_data_list[11]["elcsourcedensity"]))

    #xidx = Ri.shape[0]//2
    xidx = -1
    xidx_core=0
    zidx = [mom_data_list[i]["Zi"].shape[1]//2 for i in range(12)]
    zidx_xpt = [0,0,0]
    Rmid_out = mom_data_list[2]["Ri"][:,zidx[2]]
    Rmid_in= mom_data_list[7]["Ri"][:,zidx[7]]

    #raw_zidx = [raw_mom_data_list[i]["Ri"].shape[1]//2 for i in range(12)]
    #raw_Rmid_out = raw_mom_data_list[2]["Ri"][:,raw_zidx[2]]
    #raw_Rmid_in= raw_mom_data_list[7]["Ri"][:,raw_zidx[7]]

    # Now look at heat to side wall from diffusion
    D=0.22
    # Use surface method
    id=10
    diff_power_surf = {}
    for species in ["elc", "ion"]:
        diff_power_surf[species] = 0.0
        for edge_ind in [0]:
            dM2dx = np.gradient(mom_data_list[id][species+"M2"], mom_data_list[id]["x"], axis=0, edge_order=2)
            
            inner = mom_data_list[id]["gxx"]*mom_data_list[id]["J"]*D* masses[species]/2 * dM2dx
            diffintegrand = inner
            diff_power_surf[species]  += np.abs(np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[id]["z"])[0]) * 2 * np.pi/1e6)
    print("outboard diff_power_surf [MW] = ", diff_power_surf)

    id=11
    diff_power_surf = {}
    for species in ["elc", "ion"]:
        diff_power_surf[species] = 0.0
        for edge_ind in [0]:
            dM2dx = np.gradient(mom_data_list[id][species+"M2"], mom_data_list[id]["x"], axis=0, edge_order=2)
            
            inner = mom_data_list[id]["gxx"]*mom_data_list[id]["J"]*D* masses[species]/2 * dM2dx
            diffintegrand = inner
            diff_power_surf[species]  += np.abs(np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[id]["z"])[0]) * 2 * np.pi/1e6)
    print("inboard diff_power_surf [MW] = ", diff_power_surf)

    ## Now look at Particle Flux to Side Wall
    ## Use surface method
    total_wall_pflux = 0.0
    #edge_inds = [0, -1,-1,-1, 0,0, -1,-1,-1, 0]
    #signs = [1, -1,-1,-1, 1,1, -1,-1,-1, 1]
    edge_inds = [-1, 0,0,0, -1,-1, 0,0,0, -1]
    signs = [-1, 1,1,1, -1,-1, 1,1,1, -1]
    for bi in [0,1,2,3,4,5,6,7,8,9]:
        for species in ["elc", "ion"]:
            edge_ind = edge_inds[bi]
            sign = signs[bi]
            dM0dx = np.gradient(mom_data_list[bi][species+"M0"], mom_data_list[bi]["x"], axis=0, edge_order=2)
            
            inner = mom_data_list[bi]["gxx"]*mom_data_list[bi]["J"]*D * dM0dx
            diffintegrand = inner
            total_wall_pflux  += sign*np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[bi]["z"])[0]) * 2 * np.pi

    ## Now look at Heat Flux to Side Wall
    ## Use surface method
    total_wall_hflux = 0.0
    #edge_inds = [0, -1,-1,-1, 0,0, -1,-1,-1, 0]
    #signs = [1, -1,-1,-1, 1,1, -1,-1,-1, 1]
    edge_inds = [-1, 0,0,0, -1,-1, 0,0,0, -1]
    signs = [-1, 1,1,1, -1,-1, 1,1,1, -1]
    for bi in [0,1,2,3,4,5,6,7,8,9]:
        for species in ["elc", "ion"]:
            edge_ind = edge_inds[bi]
            sign = signs[bi]
            dM2dx = np.gradient(mom_data_list[bi][species+"M2"], mom_data_list[bi]["x"], axis=0, edge_order=2)
            
            inner = mom_data_list[bi]["gxx"]*mom_data_list[bi]["J"]*D* masses[species]/2 * dM2dx
            diffintegrand = inner
            total_wall_hflux  += sign*np.sum(diffintegrand[edge_ind,:]*np.diff(mom_data_list[bi]["z"])[0]) * 2 * np.pi



    # Calculate Gradient scale lengths at midplane
    for species in ["elc", "ion" ]:
        mom_data_list[2][species + "M0_L"] = np.abs(mom_data_list[2][species+"M0"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"M0"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2))
        mom_data_list[2][species + "Temp_L"] = np.abs(mom_data_list[2][species+"Temp"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Temp"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2))
        mom_data_list[2][species + "Tpar_L"] = np.abs(mom_data_list[2][species+"Tpar"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Tpar"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2) )
        mom_data_list[2][species + "Tperp_L"] = np.abs(mom_data_list[2][species+"Tperp"][:, zidx[2] ]/np.gradient(mom_data_list[2][species+"Tperp"][:, zidx[2] ], mom_data_list[2]["Ri"][:, zidx[2]], axis =0, edge_order=2))



    #Calculate Particle Flux to SE
    total_pflux = 0
    zedge = [0, 0, -1, -1, 0, 0, -1, -1]
    for bi, bidx in enumerate([0, 1, 3,4, 5, 6, 8,9]):
        TotalM1 = np.zeros(mom_data_list[bidx]["elcM1"].shape)
        for species in ["elc", "ion"]:
            TotalM1 += mom_data_list[bidx][species+"M1"]
        integrand = TotalM1/mom_data_list[bidx]["B"]

        pflux = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
        if(bi==1):
            print("block 1 pflux = %e\n"%pflux)
        if zedge[bi] == 0 :
            total_pflux -= pflux
        else:
            total_pflux+=pflux

    #Calculate Heat to SE
    total_heat = 0
    total_outboard_heat = 0
    total_inboard_heat = 0
    zedge = [0, 0, -1, -1, 0, 0, -1, -1]
    for bi, bidx in enumerate([0, 1, 3, 4, 5, 6, 8, 9]):
        TotalQ = np.zeros(mom_data_list[bidx]["elcQ"].shape)
        for species in ["elc", "ion"]:
            TotalQ += mom_data_list[bidx][species+"Q"]
        integrand = TotalQ/mom_data_list[bidx]["B"]

        heat = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
        if zedge[bi] == 0 :
            total_heat-= heat
            if bidx in [5,6,8,9]:
                total_inboard_heat -= heat
            else:
                total_outboard_heat -= heat
        else:
            total_heat+= heat
            if bidx in [5,6,8,9]:
                total_inboard_heat += heat
            else:
                total_outboard_heat += heat

    #Calculate ES heat to se
    total_esheat = 0
    total_outboard_esheat = 0
    total_inboard_esheat = 0
    zedge = [0, 0, -1, -1, 0, 0, -1, -1]
    for bi, bidx in enumerate([0, 1, 3,4, 5, 6, 8,9]):
        TotalQes = np.zeros(mom_data_list[bidx]["elcQ"].shape)
        for species in ["elc", "ion"]:
            TotalQes +=  charges[species]*mom_data_list[bidx][species+"M1"]
        integrand = mom_data_list[bidx]["phi"]/mom_data_list[bidx]["B"]*TotalQes
        es_heat = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
        if zedge[bi] == 0 :
            total_esheat-= es_heat
            if bidx in [5,6,8,9]:
                total_inboard_esheat-=es_heat
            else:
                total_outboard_esheat-=es_heat
        else:
            total_esheat+= es_heat
            if bidx in [5,6,8,9]:
                total_inboard_esheat+=es_heat
            else:
                total_outboard_esheat+=es_heat



    ##Calculate ES to plate
    #zedge = [0,-1,0,-1]
    #for bi, bidx in enumerate([1,3,6,8]):
    #    TotalES = np.zeros(mom_data_list[bidx]["elcM1"].shape)
    #    for species in ["elc", "ion"]:
    #        TotalES += charges[species]*mom_data_list[bidx][species+"M1"]*mom_data_list[bidx]["phi"]
    #    integrand = TotalES/mom_data_list[bidx]["B"]

    #    esheat = np.sum(integrand[:,zedge[bi]]*np.diff(mom_data_list[bidx]["x"])[0]) * 2 * np.pi
    #    print("bidx = %d, esheat = %e\n"%( bidx, esheat));

    print("Total Heat to SE = %g MW"%(total_heat/1e6))
    print("Total ES Heat to Wall = %g MW"%(total_esheat/1e6))
    print("Total Heat to Wall = %g Mw"%((total_heat + total_esheat)/1e6))
    print("Total pflux to SE = %g "%(total_pflux))

    print("Total Outboard Heat to SE = %g MW"%(total_outboard_heat/1e6))
    print("Total Inboard Heat to SE = %g MW"%(total_inboard_heat/1e6))

    print("Total Outboard ES Heat to SE = %g MW"%(total_outboard_esheat/1e6))
    print("Total Inboard ES Heat to SE = %g MW"%(total_inboard_esheat/1e6))

    print("Total Heat flux to side walls = %g MW"%(total_wall_hflux/1e6))
    print("Total Particle flux to side walls = %g"%total_wall_pflux)

    
    
    
    
    
    #Now let us set up data for plotting
    raw_keys = [["elc", "ion", "molecule"][j] + ["M0", "Temp", "Tpar", "Tperp", "Q", "normUpar", "M1"][i] for i in range(7) for j in range(3)] + ["phi"]
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


    fig,ax = plt.subplots(nrows=2, ncols=4, figsize=(16,9))
    
    linestyles = ['-' , '-' , '-' , '--']
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    
    mb_handle = mlines.Line2D([],[], color = 'k', linestyle = '-', label = "MB")
    sb_handle = mlines.Line2D([],[], color = 'k', linestyle = '--', label = "SB")
    handles = [mb_handle, sb_handle]
    
    elc_handle = mlines.Line2D([],[], color = 'tab:blue', label = "elc")
    ion_handle = mlines.Line2D([],[], color = 'tab:orange', label = "ion")
    molecule_handle = mlines.Line2D([],[], color = 'tab:red', label = "molecule")
    out_handle = mlines.Line2D([],[], color = 'k', label = "outboard")
    in_handle = mlines.Line2D([],[], color = 'k', label = "inboard", linestyle="dashed")
    handles_species = [elc_handle, ion_handle, molecule_handle]

    special_elc_handle = mlines.Line2D([],[], color = 'tab:blue', label = "e-")
    special_ion_handle = mlines.Line2D([],[], color = 'tab:orange', label = "D+")
    special_molecule_handle = mlines.Line2D([],[], color = 'tab:red', label = "D2+")
    special_handles_species = [special_elc_handle, special_ion_handle, special_molecule_handle]


    t_handle = mlines.Line2D([],[], color = 'tab:blue', label = r'$T$')
    tpar_handle = mlines.Line2D([],[], color = 'tab:orange', label = r'$T_\parallel$')
    tperp_handle = mlines.Line2D([],[], color = 'tab:green', label = r'$T_\perp$')
    temp_handles = [t_handle, tpar_handle, tperp_handle]
    temp_colors = ["tab:blue", "tab:orange", "tab:green"]
    elctemp_handle = mlines.Line2D([],[], color = 'k', label = "elc")
    iontemp_handle = mlines.Line2D([],[], color = 'k', label = "ion", linestyle="dashed")
    moleculetemp_handle = mlines.Line2D([],[], color = 'k', label = "molecule", linestyle="dotted")
    
    handles_all = [mb_handle, sb_handle, elc_handle, ion_handle]
    
    # Parallel
    # DENSITY
    for s, species in enumerate(["elc", "ion", "molecule"]):
        ax[0,0].plot(outboard_data["Zi"][xidx], outboard_data[species+"M0"][xidx], color=colors[s])

    for s, species in enumerate(["elc", "ion", "molecule"]):
        ax[0,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"M0"][xidx], color=colors[s], linestyle = "dashed")
    
    ax[0,0].set_xlabel('Z [m]')
    ax[0,0].set_ylabel(r'$n\, [m^{-3}]$')
    ax[0,0].legend(handles = handles_species + [out_handle, in_handle] )

    #PHI 
    ax[0,2].plot(outboard_data["Zi"][xidx], outboard_data["phi"][xidx], color='tab:blue')
    ax[0,2].plot(inboard_data["Zi"][xidx], inboard_data["phi"][xidx], color='tab:blue', linestyle="dashed")
    ax[0,2].set_xlabel('Z [m]')
    ax[0,2].set_ylabel(r'$\phi\, [V]$')
    ax[0,2].legend(handles = [out_handle, in_handle])


    # Outboard Temp 
    for s, species in enumerate(["elc"]):
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Temp"][xidx,:], color = "tab:blue")
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Tpar"][xidx,:], color = "tab:orange")
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Tperp"][xidx,:], color = "tab:green")
    for s, species in enumerate(["ion"]):
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Temp"][xidx,:], color = "tab:blue", linestyle="dashed")
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Tpar"][xidx,:], color = "tab:orange", linestyle="dashed")
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Tperp"][xidx,:], color = "tab:green", linestyle="dashed")
    for s, species in enumerate(["molecule"]):
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Temp"][xidx,:], color = "tab:blue", linestyle="dotted")
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Tpar"][xidx,:], color = "tab:orange", linestyle="dotted")
        ax[1,1].plot(outboard_data["Zi"][xidx], outboard_data[species+"Tperp"][xidx,:], color = "tab:green", linestyle="dotted")



    ax[1,1].set_xlabel('Z [m]')
    ax[1,1].set_ylabel(r'$T\, [eV]$')
    ax[1,1].legend(handles=temp_handles + [elctemp_handle, iontemp_handle, moleculetemp_handle])
    ax[1,1].set_title("Outboard")
    
    # Inboard Temp 
    for s, species in enumerate(["elc"]):
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Temp"][xidx,:], color = "tab:blue")
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Tpar"][xidx,:], color = "tab:orange")
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Tperp"][xidx,:], color = "tab:green")
    for s, species in enumerate(["ion"]):
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Temp"][xidx,:], color = "tab:blue", linestyle="dashed")
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Tpar"][xidx,:], color = "tab:orange", linestyle="dashed")
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Tperp"][xidx,:], color = "tab:green", linestyle="dashed")
    for s, species in enumerate(["molecule"]):
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Temp"][xidx,:], color = "tab:blue", linestyle="dotted")
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Tpar"][xidx,:], color = "tab:orange", linestyle="dotted")
        ax[1,0].plot(inboard_data["Zi"][xidx], inboard_data[species+"Tperp"][xidx,:], color = "tab:green", linestyle="dotted")



    ax[1,0].set_xlabel('Z [m]')
    ax[1,0].set_ylabel(r'$T\, [eV]$')
    ax[1,0].legend(handles=temp_handles + [elctemp_handle, iontemp_handle, moleculetemp_handle])
    ax[1,0].set_title("Inboard")
    

    # Upar 
    for s, species in enumerate(["elc", "ion", "molecule"]):
        ax[1,2].plot(outboard_data["Zi"][xidx], outboard_data[species+"normUpar"][xidx,:], color = colors[s])
    for s, species in enumerate(["elc", "ion", "molecule"]):
        ax[1,2].plot(inboard_data["Zi"][xidx], inboard_data[species+"normUpar"][xidx,:], color = colors[s], linestyle="dashed")
    
    ax[1,2].set_xlabel('Z [m]')
    ax[1,2].set_ylabel(r'$u_\parallel/c_s$')
    ax[1,2].legend(handles = handles_species + [out_handle , in_handle])

    # M1
    for s, species in enumerate(["elc", "ion"]):
        ax[1,3].plot(outboard_data["Zi"][xidx], outboard_data[species+"M1"][xidx,:], color = colors[s])
    for s, species in enumerate(["elc", "ion"]):
        ax[1,3].plot(inboard_data["Zi"][xidx], inboard_data[species+"M1"][xidx,:], color = colors[s], linestyle="dashed")
    
    ax[1,3].set_xlabel('Z [m]')
    ax[1,3].set_ylabel(r'$\Gamma\, [m^{-2}s^{-1}]$')
    ax[1,3].legend(handles = handles_species + [out_handle , in_handle])
    
    # DENSITY
    for s, species in enumerate(["elc", "ion", "molecule"]):
        ax[0,1].plot(core_data["z"], core_data[species+"M0"][xidx_core,:], color = colors[s])
    
    ax[0,1].set_xlabel(r'$\theta$')
    ax[0,1].set_ylabel(r'$n\, [m^{-3}]$')
    ax[0,1].set_title("Core")
    ax[0,1].legend(handles = handles_species)
    

    
    #PHI 
    ax[0,3].plot(core_data["z"], core_data["phi"][xidx_core,:], color = 'tab:blue')
    
    ax[0,3].set_xlabel(r'$\theta$')
    ax[0,3].set_ylabel(r'$\phi\, [V]$')
    ax[0,3].set_title("Core")



    fig.suptitle(sim_name + ", frame = %d"%frame + ", t = %1.6f ms"%(mom_data_list[0]["time"]/1e-3))
    fig.tight_layout()
    fig.savefig('movies/parplot_%03d'%frame);
    #plt.show()
    
    
    # Radial
    psifig,psiax = plt.subplots(nrows=2, ncols=4, figsize=(16,9))
    
    # Density at midplane
    for i in [7, 11]:
        for s, species in enumerate(["elc", "ion", "molecule"]):
            psiax[0,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"M0"][:,zidx[i]], color = colors[s])
    
    psiax[0,0].set_xlabel('R [m]')
    psiax[0,0].set_ylabel(r'$n\, [m^{-3}]$')
    psiax[0,0].set_title("At inboard midplane")
    psiax[0,0].legend(handles = handles_species)
    for i in [2, 10]:
        for s, species in enumerate(["elc", "ion", "molecule"]):
            psiax[0,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"M0"][:,zidx[i]], color = colors[s])
    
    psiax[0,1].set_xlabel('R [m]')
    psiax[0,1].set_ylabel(r'$n\, [m^{-3}]$')
    psiax[0,1].set_title("At outboard midplane")
    psiax[0,1].legend(handles = handles_species)

    # Temp at outboard midplane
    for i in [2,10]:
        for s, species in enumerate(["elc"]):
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0])
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1])
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2])
    for i in [2,10]:
        for s, species in enumerate(["ion"]):
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dashed")
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dashed")
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dashed")
    for i in [2,10]:
        for s, species in enumerate(["molecule"]):
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dotted")
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dotted")
            psiax[1,1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dotted")


    psiax[1,1].set_xlabel('R [m]')
    psiax[1,1].set_ylabel(r'$T \, [eV]$')
    psiax[1,1].set_title("At outboard midplane")
    psiax[1,1].legend(handles = temp_handles + [elctemp_handle, iontemp_handle])

    # Temp at inboard midplane
    for i in [7,11]:
        for s, species in enumerate(["elc"]):
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0])
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1])
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2])
    for i in [7,11]:
        for s, species in enumerate(["ion"]):
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dashed")
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dashed")
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dashed")
    for i in [7,11]:
        for s, species in enumerate(["molecule"]):
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Temp"][:,zidx[i]], color = temp_colors[0], linestyle="dotted")
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tpar"][:,zidx[i]], color=temp_colors[1], linestyle="dotted")
            psiax[1,0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tperp"][:,zidx[i]], color =temp_colors[2], linestyle="dotted")
    
    psiax[1,0].set_xlabel('R [m]')
    psiax[1,0].set_ylabel(r'$T \, [eV]$')
    psiax[1,0].set_title("At inboard midplane")
    psiax[1,0].legend(handles = temp_handles + [elctemp_handle, iontemp_handle])

    # Normal M1 at plate vs flux
    for i in [8, 9]:
        for s, species in enumerate(["elc", "ion", "molecule"]):
            #psiax[1,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M1"+"yupper"]*mom_data_list[i]["Bratio"],color = colors[s], linestyle='dashed')
            psiax[1,2].plot(IMPinterpolator(mom_data_list[i]["x"]), mom_data_list[i][species+"M1"+"yupper"]*mom_data_list[i]["Bratio"],color = colors[s], linestyle='dashed')
    
    #psiax[1,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    psiax[1,2].set_xlabel(r'$R\, [m]$' + ' mapped to IMP')
    psiax[1,2].set_ylabel(r'$\Gamma_{normal}\, [1/m^2]$')
    psiax[1,2].set_title("At inboard plate")
    psiax[1,2].legend(handles = handles_species)
    for i in [3, 4]:
        for s, species in enumerate(["elc", "ion", "molecule"]):
            #psiax[1,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"M1"+"yupper"]*mom_data_list[i]["Bratio"]*0.31,color = colors[s], linestyle='dashed')
            psiax[1,3].plot(OMPinterpolator(mom_data_list[i]["x"]), mom_data_list[i][species+"M1"+"yupper"]*mom_data_list[i]["Bratio"]*0.31,color = colors[s], linestyle='dashed')
    
    #psiax[1,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    psiax[1,3].set_xlabel(r'$R\, [m]$' + ' mapped to OMP')
    psiax[1,3].set_ylabel(r'$\Gamma_{normal}\, [1/m^2]$')
    psiax[1,3].set_title("At outboard plate")
    psiax[1,3].legend(handles = handles_species)

    #Normal Heat Flux at plate vs mapped R
    for i in [8, 9]:
        for s, species in enumerate(["elc", "ion", "molecule"]):
            #psiax[0,2].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Q"+"yupper"]/1e6*mom_data_list[i]["Bratio"],color = colors[s], linestyle='dashed')
            psiax[0,2].plot(IMPinterpolator(mom_data_list[i]["x"]), mom_data_list[i][species+"Q"+"yupper"]/1e6*mom_data_list[i]["Bratio"],color = colors[s], linestyle='dashed')
    
    #psiax[0,2].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    psiax[0,2].set_xlabel(r'$R\, [m]$' + ' mapped to IMP')
    psiax[0,2].set_ylabel(r'$Q_{normal}\, [MW/m^2]$')
    psiax[0,2].set_title("At Inboard Plate")
    psiax[0,2].legend(handles = handles_species)
    for i in [3, 4]:
        for s, species in enumerate(["elc", "ion", "molecule"]):
            #psiax[0,3].plot(mom_data_list[i]["x"], mom_data_list[i][species+"Q"+"yupper"]/1e6*mom_data_list[i]["Bratio"]*0.31,color = colors[s], linestyle='dashed')
            psiax[0,3].plot(OMPinterpolator(mom_data_list[i]["x"]), mom_data_list[i][species+"Q"+"yupper"]/1e6*mom_data_list[i]["Bratio"]*0.31,color = colors[s], linestyle='dashed')
    
    #psiax[0,3].set_xlabel(r'$\psi\,\, [Wb/rad]$')
    psiax[0,3].set_xlabel(r'$R\, [m]$' + ' mapped to OMP')
    psiax[0,3].set_ylabel(r'$Q_{normal}\, [MW/m^2]$')
    psiax[0,3].set_title("At Outboard Plate")
    psiax[0,3].legend(handles = handles_species)


    #Divertor Plate Plots
    if plot_divertor:
        ifig,iax = plt.subplots(1)
        #Connected Temp at plate vs phyiscal coordinate
        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [8, 9]:
                xplot = np.r_[xplot, mom_data_list[i]["Zi"][:,-1]]
                yplot = np.r_[yplot, mom_data_list[i][species+"Temp"][:,-1]]
            xplot = xplot - mom_data_list[8]["Zi"][-1,-1]
            iax.plot(xplot, yplot, color = colors[s])
        iax.set_xlabel(r'$Z - Z_{sep}\, [m]$', fontsize=16)
        iax.set_ylabel(r'$T\, [eV]$', fontsize=16)
        iax.set_title("Inboard Plate", fontsize=16)
        iax.legend(handles = special_handles_species)
        plt.show()
        ifig.savefig('movies/if1_%03d'%frame);

        ifig,iax = plt.subplots(1)
        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [3, 4]:
                xplot = np.r_[xplot, mom_data_list[i]["Ri"][:,-1]]
                yplot = np.r_[yplot, mom_data_list[i][species+"Temp"][:,-1]]
            xplot = xplot - mom_data_list[3]["Ri"][-1,-1]
            iax.plot(xplot, yplot, color = colors[s])
        iax.set_xlabel(r'$R - R_{sep}\, [m]$', fontsize=16)
        iax.set_ylabel(r'$T\, [eV]$', fontsize=16)
        iax.set_title("Outboard Plate", fontsize=16)
        iax.legend(handles = special_handles_species)
        plt.show()
        ifig.savefig('movies/if2_%03d'%frame);

        #Connected Poloidal Heat Flux at plate vs phyiscal coordinate
        ifig,iax = plt.subplots(1)
        for s, species in enumerate(["elc", "ion"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [8, 9]:
                xplot = np.r_[xplot, mom_data_list[i]["Zi"][:,-1]]
                yplot = np.r_[yplot, (mom_data_list[i]["elcQ"+"yupper"]+ mom_data_list[i]["ionQ"+"yupper"]+ mom_data_list[i]["moleculeQ"+"yupper"])/1e6*mom_data_list[i]["Bratio"]]
            xplot = xplot - mom_data_list[8]["Zi"][-1,-1]
            iax.plot(xplot, yplot, color = colors[0])
        iax.set_xlabel(r'$Z - Z_{sep}\, [m]$', fontsize=16)
        iax.set_ylabel(r'$Q_{normal}\, [MW/m^2]$', fontsize=16)
        iax.set_title("Inboard Plate", fontsize=16)
        plt.show()
        ifig.savefig('movies/if3_%03d'%frame);

        ifig,iax = plt.subplots(1)
        xplot = np.array([])
        yplot = np.array([])
        for i in [3, 4]:
            xplot = np.r_[xplot, mom_data_list[i]["Ri"][:,-1]]
            yplot = np.r_[yplot, (mom_data_list[i]["elcQ"+"yupper"]+ mom_data_list[i]["ionQ"+"yupper"]+ mom_data_list[i]["moleculeQ"+"yupper"])/1e6*mom_data_list[i]["Bratio"]*0.31]
        xplot = xplot - mom_data_list[3]["Ri"][-1,-1]
        iax.plot(xplot, yplot, color = colors[0])
        iax.set_xlabel(r'$R - R_{sep}\, [m]$', fontsize=16)
        iax.set_ylabel(r'$Q_{normal}\, [MW/m^2]$', fontsize=16)
        iax.set_title("Outboard Plate", fontsize=16)
        plt.show()
        ifig.savefig('movies/if4_%03d'%frame);

    #Leg Plots
    if plot_leg:
        ifig,iax = plt.subplots(2, 5, figsize=(16,9))
        #Inboard
        zidx_ileg = -2
        #Connected density at leg vs phyiscal coordinate
        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [8, 9]:
                xplot = np.r_[xplot, IMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"M0"][:,zidx_ileg]]
            xplot = xplot - mom_data_list[7]["Ri"][-1,zidx[7]]
            iax[0,0].plot(xplot, yplot, color = colors[s])
        iax[0,0].set_xlabel(r'$R - R_{sep}\, [m]$' + ' mapped to IMP', fontsize=16)
        iax[0,0].set_ylabel(r'$n\, [m^{-3}]$', fontsize=16)
        iax[0,0].set_title("Inboard Leg", fontsize=16)
        iax[0,0].legend(handles = special_handles_species)

        #Connected Temp at leg vs phyiscal coordinate
        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [8, 9]:
                xplot = np.r_[xplot, IMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Temp"][:,zidx_ileg]]
            xplot = xplot - mom_data_list[7]["Ri"][-1,zidx[7]]
            iax[0,1].plot(xplot, yplot, color = colors[s])
        iax[0,1].set_xlabel(r'$R - R_{sep}\, [m]$' + ' mapped to IMP', fontsize=16)
        iax[0,1].set_ylabel(r'$T\, [eV]$', fontsize=16)
        iax[0,1].set_title("Inboard Leg", fontsize=16)
        iax[0,1].legend(handles = special_handles_species)

        #Connected Tpar at leg vs phyiscal coordinate
        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [8, 9]:
                xplot = np.r_[xplot, IMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Tpar"][:,zidx_ileg]]
            xplot = xplot - mom_data_list[7]["Ri"][-1,zidx[7]]
            iax[0,2].plot(xplot, yplot, color = colors[s])
        iax[0,2].set_xlabel(r'$R - R_{sep}\, [m]$' + ' mapped to IMP', fontsize=16)
        iax[0,2].set_ylabel(r'$T_\parallel\, [eV]$', fontsize=16)
        iax[0,2].set_title("Inboard Leg", fontsize=16)
        iax[0,2].legend(handles = special_handles_species)

        #Connected Tperp at leg vs phyiscal coordinate
        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [8, 9]:
                xplot = np.r_[xplot, IMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Tperp"][:,zidx_ileg]]
            xplot = xplot - mom_data_list[7]["Ri"][-1,zidx[7]]
            iax[0,3].plot(xplot, yplot, color = colors[s])
        iax[0,3].set_xlabel(r'$R - R_{sep}\, [m]$' + ' mapped to IMP', fontsize=16)
        iax[0,3].set_ylabel(r'$T_\perp\, [eV]$', fontsize=16)
        iax[0,3].set_title("Inboard Leg", fontsize=16)
        iax[0,3].legend(handles = special_handles_species)

        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [8, 9]:
                xplot = np.r_[xplot, IMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Upar"][:,zidx_ileg] / np.sqrt(mom_data_list[i][species+"Temp"][:,zidx_ileg]*eV/masses[species])]
            xplot = xplot - mom_data_list[7]["Ri"][-1,zidx[7]]
            iax[0,4].plot(xplot, yplot, color = colors[s])
        iax[0,4].set_xlabel(r'$R - R_{sep}\, [m]$' + ' mapped to IMP', fontsize=16)
        iax[0,4].set_ylabel(r'$u_\parallel/v_{th}$', fontsize=16)
        iax[0,4].set_title("Inboard Leg", fontsize=16)
        iax[0,4].legend(handles = special_handles_species)

        #Outboard 
        zidx_oleg = -2
        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [3, 4]:
                xplot = np.r_[xplot, OMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"M0"][:,zidx_oleg]]
            xplot = xplot - mom_data_list[2]["Ri"][-1,zidx[2]]
            xplot*=1000
            iax[1,0].plot(xplot, yplot, color = colors[s])
        iax[1,0].set_xlabel(r'$R - R_{sep}\, [mm]$' + ' mapped to OMP', fontsize=16)
        iax[1,0].set_ylabel(r'$n\, [m^{-3}]$', fontsize=16)
        iax[1,0].set_title("Outboard Leg", fontsize=16)
        iax[1,0].legend(handles = special_handles_species)

        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [3, 4]:
                xplot = np.r_[xplot, OMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Temp"][:,zidx_oleg]]
            xplot = xplot - mom_data_list[2]["Ri"][-1,zidx[2]]
            xplot*=1000
            iax[1,1].plot(xplot, yplot, color = colors[s])
        iax[1,1].set_xlabel(r'$R - R_{sep}\, [mm]$' + ' mapped to OMP', fontsize=16)
        iax[1,1].set_ylabel(r'$T\, [eV]$', fontsize=16)
        iax[1,1].set_title("Outboard Leg", fontsize=16)
        iax[1,1].legend(handles = special_handles_species)

        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [3, 4]:
                xplot = np.r_[xplot, OMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Tpar"][:,zidx_oleg]]
            xplot = xplot - mom_data_list[2]["Ri"][-1,zidx[2]]
            xplot*=1000
            iax[1,2].plot(xplot, yplot, color = colors[s])
        iax[1,2].set_xlabel(r'$R - R_{sep}\, [mm]$' + ' mapped to OMP', fontsize=16)
        iax[1,2].set_ylabel(r'$T_\parallel\, [eV]$', fontsize=16)
        iax[1,2].set_title("Outboard Leg", fontsize=16)
        iax[1,2].legend(handles = special_handles_species)

        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [3, 4]:
                xplot = np.r_[xplot, OMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Tperp"][:,zidx_oleg]]
            xplot = xplot - mom_data_list[2]["Ri"][-1,zidx[2]]
            xplot*=1000
            iax[1,3].plot(xplot, yplot, color = colors[s])
        iax[1,3].set_xlabel(r'$R - R_{sep}\, [mm]$' + ' mapped to OMP', fontsize=16)
        iax[1,3].set_ylabel(r'$T_\perp\, [eV]$', fontsize=16)
        iax[1,3].set_title("Outboard Leg", fontsize=16)
        iax[1,3].legend(handles = special_handles_species)

        for s, species in enumerate(["elc", "ion", "molecule"]):
            xplot = np.array([])
            yplot = np.array([])
            for i in [3, 4]:
                xplot = np.r_[xplot, OMPinterpolator(mom_data_list[i]["x"])]
                yplot = np.r_[yplot, mom_data_list[i][species+"Upar"][:,zidx_oleg]/ np.sqrt(mom_data_list[i][species+"Temp"][:,zidx_oleg]*eV/masses[species])]
            xplot = xplot - mom_data_list[2]["Ri"][-1,zidx[2]]
            xplot*=1000
            iax[1,4].plot(xplot, yplot, color = colors[s])
        iax[1,4].set_xlabel(r'$R - R_{sep}\, [mm]$' + ' mapped to OMP', fontsize=16)
        iax[1,4].set_ylabel(r'$u_\parallel/v_{th}$', fontsize=16)
        iax[1,4].set_title("Outboard Leg", fontsize=16)
        iax[1,4].legend(handles = special_handles_species)

        ifig.tight_layout()
        plt.show()



    
    psifig.suptitle(sim_name + ", frame = %d"%frame + ", t = %1.6f ms"%(mom_data_list[0]["time"]/1e-3))
    psifig.tight_layout()
    psifig.savefig('movies/radplot_%03d'%frame);
    #plt.show()

    #plt.close('all')

ni= mom_data_list[2]["ionM0"][xidx,zidx[2]]
ne= mom_data_list[2]["elcM0"][xidx,zidx[2]]
Ti= mom_data_list[2]["ionTemp"][xidx,zidx[2]]
Te= mom_data_list[2]["elcTemp"][xidx,zidx[2]]
clogi=30 - np.log(np.sqrt(ni)/Ti**1.5)
cloge=31.3 - np.log(np.sqrt(ne)/Te**1.5)
R=1
a=0.5
invA=a/R
q=10
nustari = 4.9e-18*q*R*ni*clogi/(Ti**2*invA**1.5)
nustare = 6.921e-18*q*R*ne*cloge/(Te**2*invA**1.5)
print("nustari = ", nustari, "nustare=", nustare)


#Calculate ion mfp and lt
#Ions
# mfp
nuFrac = 1.0
n0 = 1.0e18
Ti0 = 10000 *  eV
logLambdaIon = 6.6 - 0.5*np.log(n0/1e20) + 1.5*np.log(Ti0/eV)
nuIon = nuFrac*logLambdaIon*(eV**4)*n0/(12*np.pi**(3/2)*(eps0**2)*np.sqrt(2.014*mp)*(Ti0)**(3/2))
#nuIon = nuIon/nuFrac # remove the nuFrac so we can do the target coll freq
#vti0 = np.sqrt(Ti0/masses["ion"])
#normNu = nuIon/n0 * (2*vti0**2)**1.5
#vti = np.sqrt(mom_data_list[2]['ionTemp']*eV/masses['ion'])
#nuIon = normNu * mom_data_list[2]['ionM0']/(2*vti**2)**1.5
#lambda_mfp_i = vti/nuIon

Te0 = 6000 * eV
logLambdaElc= 6.6 - 0.5*np.log(n0/1e20) + 1.5*np.log(Te0/eV)
nuElc = nuFrac*logLambdaElc*(eV**4)*n0/(6.0*np.sqrt(2.0)*np.pi**(3/2)*(eps0**2)*np.sqrt(masses["elc"])*Te0**(3/2))
##Lt
#geo_fac = 1/mom_data_list[1]["J"]/mom_data_list[1]["B"]
#L_ti = (mom_data_list[1]['ionTemp'][xidx,:]*eV/(geo_fac[xidx,:]*np.gradient(mom_data_list[1]['ionTemp'][xidx,:]*eV, mom_data_list[1]["z"], edge_order=2))).clip(-20,20) #Not sure about Lc for hl2a
#mfp_norm_i = np.abs(lambda_mfp_i[xidx,:]/L_ti)
#figmfp,axmfp = plt.subplots(1)
#axmfp.plot(mom_data_list[1]["z"],mfp_norm_i)
#axmfp.set_xlabel(r'Normalized Arc Length')
#axmfp.set_ylabel(r'$\lambda_{mfp,i}/L_{T,i} (\psi = %1.3f)$'%x[xidx])
#axmfp.set_title("Normalized Deuterium Mean Free Path Along Field Line")
#plt.show()

#Lc is 26m at psi = -0.10

#figmfp,axmfp = plt.subplots(1)
#axmfp.plot(mom_data_list[2]["z"],lambda_mfp_i[xidx]/vti[xidx])
#axmfp.set_xlabel(r'Normalized Arc Length')
#axmfp.set_ylabel(r'$\tau_i (\psi = %1.3f)$'%mom_data_list[2]["x"][xidx])
#axmfp.set_title("Ion  collision time Along Field Line")
#plt.show()

#Calculate heat flux width
qmid_out = mom_data_list[3]["Qtot"][:, -1]
qmid_out_cum = sci.cumulative_trapezoid(np.flip(qmid_out),x=np.flip(Rmid_out), initial=0)
xcut = np.argwhere(qmid_out_cum/qmid_out_cum.max() > 0.63)[0]
width = (np.flip(Rmid_out)[xcut] - np.flip(Rmid_out)[0])
print("Rcut = ", np.flip(Rmid_out)[xcut])
print("Outboard heat flux width = %g mm"%(width[0]/1e-3))

#qmid_in = mom_data_list[8]["Qtot"][:, -1]
#qmid_in_cum = -sci.cumulative_trapezoid(qmid_in,x=Rmid_in, initial=0)
#xcut = np.argwhere(qmid_in_cum/qmid_in_cum.max() > 0.63)[0]
#width = np.abs(Rmid_in[xcut] - Rmid_in[0])
#print("Rcut = ", Rmid_in[xcut])
#print("Inboard heat flux width = %g mm"%(width[0]/1e-3))
#
#
##Calculate actual width
#Rmid = mom_data_list[2]["Ri"][:, zidx[2]]
#nimid = mom_data_list[2]["ionM0"][:, zidx[2]]
#nimid_cum = sci.cumulative_trapezoid(nimid,x=Rmid, initial=0)
##print(nimid_cum/nimid_cum.max())
#xcut = np.argwhere(nimid_cum/nimid_cum.max() > 0.63)[0]
#width = (Rmid[xcut] - Rmid[0])
#
##Caclulate expected width
#Lc = 26 # 26m from midplane to plate
#t_transit = Lc/vti[xidx,zidx[2]]
#t_coll = (lambda_mfp_i/vti)[xidx,zidx[2]]
#l_transit = (D*t_transit)**0.5
#l_coll = (D*t_coll)**0.5
#
##Calculate ratio of B and then trapped fraction
#Bmax = mom_data_list[2]["B"][xidx,-1]
#Bmin = mom_data_list[2]["B"][xidx,zidx[2]]
#frac_trap = np.sqrt(1 - 1/(Bmax/Bmin))
#frac_pass = 1 - frac_trap
#
##Use a  weighted sum to get an expected width
#Rtest = np.linspace(0,21e-3,100)
#y_trap = frac_trap*np.exp(-Rtest/l_coll)
#y_pass = frac_pass*np.exp(-Rtest/l_transit)
#y_tot = y_trap+y_pass
#ycum = sci.cumulative_trapezoid(y_tot, x=Rtest, initial = 0)
#exp_xcut = np.argwhere(ycum/ycum.max() > 0.63)[0]
#exp_width = (Rtest[exp_xcut] - Rtest[0])
#
#print("width  = %g mm, expected width = %g mm"%(width[0]/1e-3, exp_width[0]/1e-3))

#nfig,nax = plt.subplots(1)
#n0i = mom_data_list[2]["ionM0"][xidx,zidx[2]]
#exp_ni = n0i*frac_pass*np.exp(-(Rmid-Rmid[0])/l_transit) + n0i*frac_trap*np.exp(-(Rmid-Rmid[0])/l_coll)
#nax.plot(Rmid, mom_data_list[2]["ionM0"][:, zidx[2]], label = "actual")
#nax.plot(Rmid, exp_ni, label = "two decay lengths")
#nax.legend()
#nax.set_xlabel("R [m]")
#nax.set_ylabel(r'$n_i\, [m^{-3}]$')

#Plot scale lengths

# Electron scale lengths at midplane
#Lfig, Lax = plt.subplots(1,3, figsize=(16,9))
#for i in [2]:
#    for s, species in enumerate(["elc"]):
#        Lax[0].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"M0_L"], label = r'$L_n = |n/(dn/dR)|$')
#        Lax[1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Temp_L"], label = r'$L_T = |T/(dT/dR)|$')
#        Lax[2].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tpar_L"], label = r'$L_{T\parallel} = |T_\parallel/(dT_\parallel/dR)|$')
#        Lax[1].plot(mom_data_list[i]["Ri"][:,zidx[i]], mom_data_list[i][species+"Tperp_L"], label = r'$L_{T\perp} = |T_\perp/(dT_\perp/dR)|$')
#
#Lax[0].set_xlabel('R [m]')
#Lax[1].set_xlabel('R [m]')
#Lax[2].set_xlabel('R [m]')
#Lax[0].set_ylabel('Electron Density Scale Lengths L [m]')
#Lax[1].set_ylabel('Electron Temperature Scale Lengths L [m]')
#Lax[2].set_ylabel('Electron Temperature Scale Lengths L [m]')
#Lax[0].set_title("At midplane")
#Lax[1].set_title("At midplane")
#Lax[2].set_title("At midplane")
#Lax[0].legend()
#Lax[1].legend()
#Lax[2].legend()
#Lfig.tight_layout()
#Lfig.savefig('movies/scaleplot_%03d'%frame);

def plot_bgk(species, mom):
    bmin=0
    bmax=12
    minval = 1.0e50
    maxval = -1.0e50
    for i in range(bmin,bmax):
        nvals = mom_data_list[i][species+"BGK%sdot"%mom]
        minval = min(minval, np.min(nvals))
        maxval = max(maxval, np.max(nvals))

    fig,ax = plt.subplots(figsize = (6,8))
    norm=mpl.colors.SymLogNorm(vmin=minval, vmax=maxval, linthresh=max(np.abs(minval)/1e3, maxval)/1e3)
    for i in range(bmin,bmax):
        im=ax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], mom_data_list[i][species+"BGK%sdot"%mom], cmap='inferno', norm = norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    #cbar.set_label(r'$\dot{M0}_{%s} \, [m{^-3}s^{-1}]$'%species, fontsize=20)
    cbar.set_label(r'$\dot{%s}_{%s}$'%(mom,species), fontsize=20)
    ax.set_xlabel('R [m]', fontsize=20)
    ax.set_ylabel('Z [m]', fontsize=20)
    #ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles='dashed')
    fig.tight_layout()

def plot_mom(species, mom):
    bmin=0
    bmax=12
    minval = 1.0e50
    maxval = -1.0e50
    for i in range(bmin,bmax):
        nvals = mom_data_list[i][species+mom]
        minval = min(minval, np.min(nvals))
        maxval = max(maxval, np.max(nvals))

    fig,ax = plt.subplots(figsize = (6,8))
    #norm=mpl.colors.SymLogNorm(vmin=minval, vmax=maxval, linthresh=max(np.abs(minval)/1e3, maxval)/1e3)
    norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
    for i in range(bmin,bmax):
        im=ax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], mom_data_list[i][species+mom], cmap='inferno', norm = norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    #cbar.set_label(r'$\dot{M0}_{%s} \, [m{^-3}s^{-1}]$'%species, fontsize=20)
    cbar.set_label(r'$\dot{%s}_{%s}$'%(mom,species), fontsize=20)
    ax.set_xlabel('R [m]', fontsize=20)
    ax.set_ylabel('Z [m]', fontsize=20)
    #ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles='dashed')
    fig.tight_layout()

def plot_bgk_ratio(species, mom):
    bmin=0
    bmax=12
    minval = 1.0e50
    maxval = -1.0e50
    for i in range(bmin,bmax):
        nvals = mom_data_list[i][species+"BGK%sdot"%mom]/mom_data_list[i][species+mom]
        minval = min(minval, np.min(nvals))
        maxval = max(maxval, np.max(nvals))

    fig,ax = plt.subplots(figsize = (6,8))
    norm=mpl.colors.SymLogNorm(vmin=minval, vmax=maxval, linthresh=max(np.abs(minval)/1e3, maxval)/1e3)
    for i in range(bmin,bmax):
        im=ax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], mom_data_list[i][species+"BGK%sdot"%mom]/mom_data_list[i][species+mom], cmap='inferno', norm = norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    #cbar.set_label(r'$\dot{M0}_{%s} \, [m{^-3}s^{-1}]$'%species, fontsize=20)
    cbar.set_label(r'$\dot{%s}_{%s}$'%(mom,species), fontsize=20)
    ax.set_xlabel('R [m]', fontsize=20)
    ax.set_ylabel('Z [m]', fontsize=20)
    #ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles='dashed')
    fig.tight_layout()


def calc_coll_freq(bidx):
    nuIon = nuFrac*logLambdaIon*(eV**4)*mom_data_list[bidx]["ionM0"]/(12*np.pi**(3/2)*(eps0**2)*np.sqrt(masses["ion"])*(mom_data_list[bidx]["ionTemp"]*eV)**(3/2))
    nuElc = nuFrac*logLambdaElc*(eV**4)*mom_data_list[bidx]["elcM0"]/(6.0*np.sqrt(2.0)*np.pi**(3/2)*(eps0**2)*np.sqrt(masses["elc"])*(mom_data_list[bidx]["elcTemp"]*eV)**(3/2))
    return nuElc, nuIon



# Calculated integrated diagnostics from native stuff
bflux_frame= -1#(frame-306)*100
#First do parallel
qidata = []
qedata = []
qmdata = []
for bi in [0,1]:
    qidata.append(pg.GData('%s_b%d-ion_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qedata.append(pg.GData('%s_b%d-elc_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qmdata.append(pg.GData('%s_b%d-molecule_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)

for bi in [4,5]:
    qidata.append(pg.GData('%s_b%d-ion_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qedata.append(pg.GData('%s_b%d-elc_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qmdata.append(pg.GData('%s_b%d-molecule_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)

nidata = []
nedata = []
nmdata = []
for bi in [0,1]:
    nidata.append(pg.GData('%s_b%d-ion_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nedata.append(pg.GData('%s_b%d-elc_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nmdata.append(pg.GData('%s_b%d-molecule_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)

for bi in [4,5]:
    nidata.append(pg.GData('%s_b%d-ion_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nedata.append(pg.GData('%s_b%d-elc_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nmdata.append(pg.GData('%s_b%d-molecule_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)


midata = []
medata = []
mmdata = []
for bi in [0,1]:
    midata.append(pg.GData('%s_b%d-ion_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    medata.append(pg.GData('%s_b%d-elc_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    mmdata.append(pg.GData('%s_b%d-molecule_bflux_ylower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)

for bi in [4,5]:
    midata.append(pg.GData('%s_b%d-ion_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    medata.append(pg.GData('%s_b%d-elc_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    mmdata.append(pg.GData('%s_b%d-molecule_bflux_yupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)

#Second do perp
qidata_perp = []
qedata_perp = []
qmdata_perp = []
for bi in [1,2,3,4]:
    qidata_perp.append(pg.GData('%s_b%d-ion_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qedata_perp.append(pg.GData('%s_b%d-elc_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qmdata_perp.append(pg.GData('%s_b%d-molecule_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)

for bi in [0,5]:
    qidata_perp.append(pg.GData('%s_b%d-ion_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qedata_perp.append(pg.GData('%s_b%d-elc_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)
    qmdata_perp.append(pg.GData('%s_b%d-molecule_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,2]/1e6*2*np.pi)

nidata_perp = []
nedata_perp = []
nmdata_perp = []
for bi in [1,2,3,4]:
    nidata_perp.append(pg.GData('%s_b%d-ion_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nedata_perp.append(pg.GData('%s_b%d-elc_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nmdata_perp.append(pg.GData('%s_b%d-molecule_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)

for bi in [0,5]:
    nidata_perp.append(pg.GData('%s_b%d-ion_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nedata_perp.append(pg.GData('%s_b%d-elc_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)
    nmdata_perp.append(pg.GData('%s_b%d-molecule_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,0]*2*np.pi)

midata_perp = []
medata_perp = []
mmdata_perp = []
for bi in [1,2,3,4]:
    midata_perp.append(pg.GData('%s_b%d-ion_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    medata_perp.append(pg.GData('%s_b%d-elc_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    mmdata_perp.append(pg.GData('%s_b%d-molecule_bflux_xlower_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)

for bi in [0,5]:
    midata_perp.append(pg.GData('%s_b%d-ion_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    medata_perp.append(pg.GData('%s_b%d-elc_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)
    mmdata_perp.append(pg.GData('%s_b%d-molecule_bflux_xupper_integrated_HamiltonianMoments.gkyl'%(base_name,bi)).get_values()[bflux_frame,1]*2*np.pi)


#PARALLEL
total_native_heat = np.sum(qidata)*2+np.sum(qedata)*2 + np.sum(qmdata)*2
total_native_pflux = np.sum(nidata)*2+np.sum(nedata)*2 + np.sum(nmdata)*2
total_native_mflux = np.sum(midata)*2+np.sum(medata)*2 + np.sum(mmdata)*2
#PERP
total_native_heat_perp = np.sum(qidata_perp)*2+np.sum(qedata_perp)*2+np.sum(qmdata_perp)*2
total_native_pflux_perp = np.sum(nidata_perp)*2+np.sum(nedata_perp)*2+np.sum(nmdata_perp)*2
total_native_mflux_perp = np.sum(midata_perp)*2+np.sum(medata_perp)*2+np.sum(mmdata_perp)*2
print("\n NATIVE DIAGNOSTICS")
print("Total Heat to Plate = %g MW"%(total_native_heat))
print("Total Particles to Plate = %12e"%total_native_pflux)
print("Total Momentum to Plate = %12e"%total_native_mflux)
print("\n Total Heat to Walls = %g MW"%(total_native_heat_perp))
print("Total Particles to Walls = %12e"%total_native_pflux_perp)
print("Total Momentum to Walls = %12e"%total_native_mflux_perp)


# Look at integrated moments
integrated_mom_data_list = []
bflux_integrated_mom_data_list = []
source_integrated_mom_data_list = []
for isim, sim_name in enumerate(sim_names):
    integrated_mom_data= {}
    bflux_integrated_mom_data= {}
    source_integrated_mom_data= {}
    for species in ["elc", "ion", "molecule"]:
         itime = pg.GData('%s-%s_integrated_moms.gkyl'%(sim_name,species)).get_grid()[0]
         integrated_mom_data[species] = pg.GData('%s-%s_integrated_moms.gkyl'%(sim_name,species)).get_values()*2*np.pi
         if isim in [6,7] and species!="molecule":
            source_integrated_mom_data[species] = pg.GData('%s-%s_source_integrated_moms.gkyl'%(sim_name,species)).get_values()*2*np.pi
         else:
            source_integrated_mom_data[species] = None
         for bdry in ["xlower", "xupper", "ylower", "yupper"]:
            bflux_fname = '%s-%s_bflux_%s_integrated_HamiltonianMoments.gkyl'%(sim_name,species, bdry)
            if os.path.exists(bflux_fname):
                bflux_integrated_mom_data[species+bdry] = pg.GData(bflux_fname).get_values()*2*np.pi
            else:
                bflux_integrated_mom_data[species+bdry] = None 

    integrated_mom_data_list.append(integrated_mom_data)
    bflux_integrated_mom_data_list.append(bflux_integrated_mom_data)
    source_integrated_mom_data_list.append(source_integrated_mom_data)

# Let's make some plots to see if parallel losses are ambipolar
parallel_losses = {}
for species in ["elc", "ion", "molecule"]:
    for bi in [0,1]:
        key = species
        val = bflux_integrated_mom_data_list[bi][f"{species}ylower"][:,0]
        if key not in parallel_losses:
            parallel_losses[key] = val
        else:
            parallel_losses[key] += val

    for bi in [4,5]:
        key = species
        val = bflux_integrated_mom_data_list[bi][f"{species}yupper"][:,0]
        if key not in parallel_losses:
            parallel_losses[key] = val
        else:
            parallel_losses[key] += val

parallel_heat_losses = {}
for species in ["elc", "ion", "molecule"]:
    for bi in [0,1]:
        key = species
        val = bflux_integrated_mom_data_list[bi][f"{species}ylower"][:,2]
        if key not in parallel_heat_losses:
            parallel_heat_losses[key] = val
        else:
            parallel_heat_losses[key] += val

    for bi in [4,5]:
        key = species
        val = bflux_integrated_mom_data_list[bi][f"{species}yupper"][:,2]
        if key not in parallel_heat_losses:
            parallel_heat_losses[key] = val
        else:
            parallel_heat_losses[key] += val

ifig, iax = plt.subplots(2)
tstart=0

iax[0].plot(itime[tstart:], parallel_losses["elc"][tstart:]*2, label = "elc")
iax[0].plot(itime[tstart:], parallel_losses["ion"][tstart:]*2, label = "ion")
iax[0].plot(itime[tstart:], parallel_losses["molecule"][tstart:]*2, label = "molecule")
iax[0].plot(itime[tstart:], (parallel_losses["elc"][tstart:] + parallel_losses["ion"][tstart:] + parallel_losses["molecule"][tstart:])*2, label = "total")
iax[0].set_xlabel(r'$t\,[s]$')
iax[0].set_ylabel('Parallel Particle Loss: ' + r'$\dot{n}\,[m^3s^{-1}]$')
iax[0].legend()

#rel_error = (2*np.abs(parallel_losses["ion"] + parallel_losses["molecule"] - parallel_losses["elc"])/(parallel_losses["ion"] +parallel_losses["molecule"] + parallel_losses["elc"]))
#iax[1].semilogy(itime[tstart:], rel_error[tstart:])
#iax[1].set_xlabel(r'$t\,[s]$')
#iax[1].set_ylabel('Relative Error: ' + r'$ 2\frac{|\dot{n_i} - \dot{n_e}|}{\dot{n_i} + \dot{n_e}} $')

iax[1].plot(itime[tstart:], parallel_heat_losses["elc"][tstart:]/1e6*2, label = "elc")
iax[1].plot(itime[tstart:], parallel_heat_losses["ion"][tstart:]/1e6*2, label = "ion")
iax[1].plot(itime[tstart:], parallel_heat_losses["molecule"][tstart:]/1e6*2, label = "molecule")
iax[1].plot(itime[tstart:],  (parallel_heat_losses["elc"][tstart:] +  parallel_heat_losses["ion"][tstart:] + parallel_heat_losses["molecule"][tstart:])/1e6*2, label = "total")
iax[1].set_xlabel(r'$t\,[s]$')
iax[1].set_ylabel('Parallel Heat Loss' + r'$[MW]$')
iax[1].legend()
iax[1].grid()

def plot_source_and_loss(bi,species, bdry):
    plt.figure()
    plt.plot(source_integrated_mom_data_list[bi][species][:,0])
    plt.plot(bflux_integrated_mom_data_list[bi][species+bdry][:,0], linestyle='dashed')

def plot_loss(bi,species, bdry):
    plt.figure()
    plt.plot(bflux_integrated_mom_data_list[bi][species+bdry][:,0], linestyle='dashed')



    
