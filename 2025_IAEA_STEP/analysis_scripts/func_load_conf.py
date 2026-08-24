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


import sys
import os
import yaml
import scipy.constants as pyconst
from scipy.ndimage import median_filter

from reactionrates import HydrogenEffectiveRates


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



def load_eirene(sim_name, frame, gkeyll_simulation_name='hstep26'):
    # Extract paths from config
    target_dir = '../gkapp-main3/gkylsoft/GK-Neutral_coupling/common/'
    sys.path.insert(0, target_dir)
    import B2IO as b2
    import eireneIO as eirene
    import triangle_mesh as tri_mesh
    
    # Set paths from config
    gkeyll_data_path = sim_dir = '../' + sim_name + '/'
    eirene_data_path = sim_dir = '../' + sim_name + '/eirene_history/' + str(frame) + '/'

    # Block range
    bmin = 0
    bmax = 8
     
    # Helper Function for de-noising
    def despike_source(data, kernel_size=3):
        """
        Applies a median filter to remove single-pixel outliers (Monte Carlo noise).
        A 3x3 kernel is usually sufficient to kill spikes without blurring features.
        """
        # Check for NaNs just in case and replace with 0
        data = np.nan_to_num(data, nan=0.0)
        return median_filter(data, size=kernel_size)
    
    
    edat = eirene.eirene(eirene_data_path)
    edat.load_extra_forts(eirene_data_path)
    edat.triangle_mesh.calc_incenter()
    eR = edat.triangle_mesh.incenter[:,0]
    eZ = edat.triangle_mesh.incenter[:,1]
    atoms = edat.fort46['pdena']
    atoms_source = edat.sources["particle"]['D']["SUM"]/eV*1e6
    atoms_temp =  edat.fort46['edena']/atoms/eV

    iz_source = edat.loaded_sources.sources["particle"]["atom-plasma"]["ELECTRONS"]["SUM"]/eV*1e6

    # Calculations
    mass_ion = 3.34e-27
    mass_elc = 9.11e-31

    ion = "D+"
    
    pisource = edat.sources["particle"][ion]["SUM"]
    misource = edat.sources["momentum"][ion]["SUM"]
    eisource = edat.sources["energy"][ion]["SUM"]
    
    pesource = edat.sources["particle"]["ELECTRONS"]["SUM"]
    eesource = edat.sources["energy"]["ELECTRONS"]["SUM"]
    
    
    # Step 1: load the Gkeyll grid information
    #     same as process_eirene_output.py's Step 1
    simNames = ['%s_b%d'%(gkeyll_data_path+gkeyll_simulation_name,i) for i in range(bmin,bmax)]
    Rlist = []
    Zlist = []
    nodal_grid_list = []
    jlist = []
    for i, simName in enumerate(simNames):
        data = pg.GData(simName+"-nodesint.gkyl")
        vals = data.get_values()
        R = vals[:,:,0]
        Z = vals[:,:,1]
        phi = vals[:,:,2]
        
        Rlist.append(R)
        Zlist.append(Z)
    
    
    # Step 2: Fill Nodal Gkeyll data by finding closest point from Eirene
    # Changed to use data calculated/loaded from Eirene
    M0D_list = []
    M0D2_list = []
    M0D_source_list = []
    M0D2_source_list = []
    M2D_list = []
    M2D2_list = []
    iz_list = []

    M0i_list = []
    M1i_list = []
    M2i_list = []
    
    M0e_list = []
    M2e_list = []
    

    for i, simName in enumerate(simNames):
        nx, nz = Rlist[i].shape
        M0D = np.zeros((nx,nz))
        M0D2 = np.zeros((nx,nz))
        M0D_source = np.zeros((nx,nz))
        M0D2_source = np.zeros((nx,nz))
        M2D = np.zeros((nx,nz))
        M2D2 = np.zeros((nx,nz))
        izi = np.zeros((nx,nz))

        M0i = np.zeros((nx,nz))
        M1i = np.zeros((nx,nz))
        M2i = np.zeros((nx,nz))

        M0e = np.zeros((nx,nz))
        M2e = np.zeros((nx,nz))


        for ix in range(nx):
            for iz in range(nz):
                lindist = np.sqrt((Rlist[i][ix,iz] - eR)**2 + (Zlist[i][ix,iz] - eZ)**2)
                linidx = np.argmin(lindist)
                M0D[ix,iz] = atoms[linidx]
                M0D2[ix,iz] = molecules[linidx]
                M0D_source[ix,iz] = atoms_source[linidx]
                M0D2_source[ix,iz] = molecules_source[linidx]
                M2D[ix,iz] = atoms_temp[linidx]
                M2D2[ix,iz] = molecules_temp[linidx]
                izi[ix,iz] = iz_source[linidx]

                # M0 source calculation
                M0i[ix,iz] = pisource[linidx]/eV*1e6
                M0e[ix,iz] = pesource[linidx]/eV*1e6

                # M1 source calculation
                M1i[ix,iz] = misource[linidx]*10/mass_ion/eV

                # M2 source Calculation
                M2i[ix,iz] = eisource[linidx]*1e6/mass_ion*2.0
                M2e[ix,iz] = eesource[linidx]*1e6/mass_elc*2.0
    
    
    
        # Append the clipped/smoothed data
        M0D_list.append(M0D)
        M0D2_list.append(M0D2)
        M0D_source_list.append(M0D_source)
        M0D2_source_list.append(M0D2_source)
        M2D_list.append(M2D)
        M2D2_list.append(M2D2)
        iz_list.append(izi)

        # Smooth data
        M1i_smoothed = despike_source(M1i, kernel_size=3)

        # Append the clipped/smoothed data
        M0i_list.append(M0i)
        M1i_list.append(M1i_smoothed)
        M2i_list.append(M2i)
        M0e_list.append(M0e)
        M2e_list.append(M2e)

    eirene_data = {}
    eirene_data['M0D'] = M0D_list
    eirene_data['M0D2'] = M0D2_list
    eirene_data['M0D_source'] = M0D_source_list
    eirene_data['M0D2_source'] = M0D2_source_list
    eirene_data['M2D'] = M2D_list
    eirene_data['M2D2'] = M2D2_list
    eirene_data['M0i'] = M0i_list
    eirene_data['M1i'] = M1i_list
    eirene_data['M2i'] = M2i_list
    eirene_data['M0e'] = M0e_list
    eirene_data['M2e'] = M2e_list
    eirene_data['iz_source'] = iz_list

    #Calculate Balance of D2
    atemp = (edat.fort46['volumes'].flatten() * edat.sources["particle"]['D2']["SUM"].flatten())
    atemp[np.isnan(atemp)] = 0.0
    d2vol = np.sum(atemp) * 1e6/eV
    
    d2volsource = np.sum(atemp[atemp>0]) * 1e6/eV
    d2volsink = np.sum(atemp[atemp<0]) * 1e6/eV
    
    d2flux = edat.fort44['wld']['wldpm'][:,:,0].sum()
    d2pump = edat.fort44['wld']['wlpump(M)'].sum()
    
    total_d2_source = d2flux
    total_d2_sink = d2pump - d2volsource
    
    #Calculate balance of D
    dflux = edat.fort44['wld']['wldpa'][:,:,0].sum()
    dpump = edat.fort44['wld']['wlpump(A)'].sum()
    
    atemp = (edat.fort46['volumes'].flatten() * edat.sources["particle"]['D']["SUM"].flatten())
    atemp[np.isnan(atemp)] = 0.0
    dvol = np.sum(atemp) * 1e6/eV
    dvolsource = np.sum(atemp[atemp>0]) * 1e6/eV
    dvolsink = np.sum(atemp[atemp<0]) * 1e6/eV
    
    total_d_source = dflux + dvolsource
    total_d_sink = dpump
    
    #Calculate D+ source rate
    atemp = (edat.fort46['volumes'].flatten() * edat.sources["particle"]['D+']["SUM"].flatten())
    atemp[np.isnan(atemp)] = 0.0
    dpvol = np.sum(atemp) * 1e6/eV
    
    #Calculate D2+ source rate
    atemp = (edat.fort46['volumes'].flatten() * edat.sources["particle"]['D2+']["SUM"].flatten())
    atemp[np.isnan(atemp)] = 0.0
    d2pvol = np.sum(atemp) * 1e6/eV

    #Impinging d2p ions
    dpimp = edat.fort44['wld']['wldpp'][:,0,0].sum()

    eirene_stats = {}
    eirene_stats["Demit"] = dflux
    eirene_stats["Dcreate"] = dvolsource
    eirene_stats["Dlost"] = dvolsink
    eirene_stats["Dpump"] = dpump
    eirene_stats["Dpercent"] = dpump/(dvolsource+dflux)

    eirene_stats["D2emit"] = d2flux
    eirene_stats["D2create"] = d2volsource
    eirene_stats["D2lost"] = d2volsink
    eirene_stats["D2pump"] = d2pump
    eirene_stats["D2percent"] = d2pump/(d2volsource+d2flux)

    eirene_stats["D+create"] = dpvol
    eirene_stats["D2+create"] = d2pvol

    eirene_stats["D+imp"] = dpimp

    return eirene_data, eirene_stats



def load_data(sim_name, frame, gkeyll_simulation_name='hstep26', source_frame=0):
    # Set up names for data loading
    sim_dir = '../' + sim_name + '/'
    base_name = sim_dir+gkeyll_simulation_name
    bmin, bmax = 0, 8
    sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]
    
    sim_labels= ['b%d'%i for i in range(bmin,bmax)]


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
        if isim == 0 : 
            plate_data = np.genfromtxt(sim_dir+"stepplate_data/highres/opf.txt", delimiter = ",")
        if isim == 4 : 
            plate_data = np.genfromtxt(sim_dir+"stepplate_data/highres/isol.txt", delimiter = ",")
        if isim == 5 : 
            plate_data = np.genfromtxt(sim_dir+"stepplate_data/highres/ipf.txt", delimiter = ",")

    
        #Load bflux data
        for species in ["elc", "ion"]:
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
        source_mom_data["J"] = mom_data["J"]
        source_mom_data["x"] = mom_data["x"]
        source_mom_data["z"] = mom_data["z"]
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
    even_keys = [["elc", "ion"][j] + ["M0", "M2", "Temp", "Tpar", "Tperp", "BGKM0dot", "BGKM1dot", "BGKM2dot"][i] for i in range(8) for j in range(2)] + ["phi" , "gxx", "gzz", "Ri", "J", "B"]
    odd_keys = [["elc", "ion"][j] + ["M1", "Upar", "normUpar", "Q", "Qwall"][i] for i in range(5) for j in range(2)] + ["Zi", "Qtot"]
    bflux_keys = []
    for species in ["elc", "ion"]:
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
    mom_data_list[4]["z"] = -np.flip(mom_data_list[0]["z"])
    mom_data_list[5]["z"] = -np.flip(mom_data_list[9]["z"])

    raw_mom_data_list[3]["z"] = -np.flip(raw_mom_data_list[1]["z"])
    raw_mom_data_list[6]["z"] = -np.flip(raw_mom_data_list[8]["z"])
    raw_mom_data_list[4]["z"] = -np.flip(raw_mom_data_list[0]["z"])
    raw_mom_data_list[5]["z"] = -np.flip(raw_mom_data_list[9]["z"])
    
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

    ##Reaction Rates
    #rates = HydrogenEffectiveRates()
    #for ib in range(12):
    #    mom_data_list[ib]["rate13"] = rates.reaction_13(mom_data_list[ib]["elcM0"], mom_data_list[ib]["elcTemp"]).T
    #    mom_data_list[ib]["rate14"] = rates.reaction_14(mom_data_list[ib]["elcM0"], mom_data_list[ib]["elcTemp"]).T
    #    mom_data_list[ib]["dissociation_rate"] =  mom_data_list[ib]["elcM0"]*mom_data_list[ib]["moleculeM0"]*mom_data_list[ib]["rate13"] +  mom_data_list[ib]["elcM0"]*mom_data_list[ib]["moleculeM0"]*mom_data_list[ib]["rate14"]



    sim_data = {}
    sim_data['name'] = sim_name
    sim_data['R_OMP_sep'] = mom_data_list[2]["Ri"][-1, zidx[2]]
    sim_data['R_OP_sep'] = mom_data_list[3]["Ri"][-1, -1]

    sim_data['R_OMP'] = np.r_[mom_data_list[2]["Ri"][:, zidx[2]],mom_data_list[10]["Ri"][:, zidx[10]]]
    sim_data['Ti_OMP'] = np.r_[mom_data_list[2]["ionTemp"][:, zidx[2]],mom_data_list[10]["ionTemp"][:, zidx[10]]]
    sim_data['Te_OMP'] = np.r_[mom_data_list[2]["elcTemp"][:, zidx[2]],mom_data_list[10]["elcTemp"][:, zidx[10]]]
    sim_data['ne_OMP'] = np.r_[mom_data_list[2]["elcM0"][:, zidx[2]],mom_data_list[10]["elcM0"][:, zidx[10]]]
    sim_data['ni_OMP'] = np.r_[mom_data_list[2]["ionM0"][:, zidx[2]],mom_data_list[10]["ionM0"][:, zidx[10]]]
    sim_data['ue_OMP'] = np.r_[mom_data_list[2]["elcUpar"][:, zidx[2]],mom_data_list[10]["elcUpar"][:, zidx[10]]]
    sim_data['ui_OMP'] = np.r_[mom_data_list[2]["ionUpar"][:, zidx[2]],mom_data_list[10]["ionUpar"][:, zidx[10]]]

    sim_data['R_OP'] = np.r_[mom_data_list[3]["Ri"][:, -1],mom_data_list[4]["Ri"][:, -1]]
    sim_data['Ti_OP'] = np.r_[mom_data_list[3]["ionTemp"][:, -1],mom_data_list[4]["ionTemp"][:, -1]]
    sim_data['Te_OP'] = np.r_[mom_data_list[3]["elcTemp"][:, -1],mom_data_list[4]["elcTemp"][:, -1]]
    sim_data['ne_OP'] = np.r_[mom_data_list[3]["elcM0"][:, -1],mom_data_list[4]["elcM0"][:, -1]]
    sim_data['ni_OP'] = np.r_[mom_data_list[3]["ionM0"][:, -1],mom_data_list[4]["ionM0"][:, -1]]
    sim_data['ue_OP'] = np.r_[mom_data_list[3]["elcUpar"][:, -1],mom_data_list[4]["elcUpar"][:, -1]]
    sim_data['ui_OP'] = np.r_[mom_data_list[3]["ionUpar"][:, -1],mom_data_list[4]["ionUpar"][:, -1]]

    return mom_data_list, source_half_mom_data_list, sim_data


def integrate_over_vol(mom_data_list, key):
    num_blocks = len(mom_data_list)
    total = 0.0
    for ib in range(num_blocks):
        if key in mom_data_list[ib]:
            integrand = mom_data_list[ib][key] * mom_data_list[ib]["J"]
            block_total = np.sum(integrand*np.diff(mom_data_list[ib]["x"])[0]*np.diff(mom_data_list[ib]["z"])[0]) * 2 * np.pi
            total+=block_total
            print("For block %d total was %g"%(ib, block_total))
        else:
            continue
    print("Total was %g over %d blocks\n"%(total, num_blocks))
    return total

def plot_mom(mom_data_list, mom):
    bmin=0
    bmax=12
    minval = 1.0e50
    maxval = -1.0e50
    for i in range(bmin,bmax):
        nvals = mom_data_list[i][mom]
        minval = min(minval, np.min(nvals))
        maxval = max(maxval, np.max(nvals))

    fig,ax = plt.subplots(figsize = (6,8))
    #norm=mpl.colors.SymLogNorm(vmin=minval, vmax=maxval, linthresh=max(np.abs(minval)/1e3, maxval)/1e3)
    norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
    for i in range(bmin,bmax):
        im=ax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], mom_data_list[i][mom], cmap='inferno', norm = norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    #cbar.set_label(r'$\dot{M0}_{%s} \, [m{^-3}s^{-1}]$'%species, fontsize=20)
    cbar.set_label(mom, fontsize=20)
    ax.set_xlabel('R [m]', fontsize=20)
    ax.set_ylabel('Z [m]', fontsize=20)
    #ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles='dashed')
    fig.tight_layout()

def plot_log_mom(mom_data_list, mom):
    bmin=0
    bmax=12
    minval = 1.0e50
    maxval = -1.0e50
    for i in range(bmin,bmax):
        nvals = mom_data_list[i][mom]
        minval = min(minval, np.min(nvals))
        maxval = max(maxval, np.max(nvals))

    fig,ax = plt.subplots(figsize = (6,8))
    norm=mpl.colors.SymLogNorm(vmin=minval, vmax=maxval, linthresh=max(np.abs(minval)/1e3, maxval)/1e3)
    for i in range(bmin,bmax):
        im=ax.pcolor(mom_data_list[i]["Ri"], mom_data_list[i]["Zi"], mom_data_list[i][mom], cmap='inferno', norm = norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    #cbar.set_label(r'$\dot{M0}_{%s} \, [m{^-3}s^{-1}]$'%species, fontsize=20)
    cbar.set_label(mom, fontsize=20)
    ax.set_xlabel('R [m]', fontsize=20)
    ax.set_ylabel('Z [m]', fontsize=20)
    #ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles='dashed')
    fig.tight_layout()
