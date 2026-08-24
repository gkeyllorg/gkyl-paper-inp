import numpy as np
import matplotlib.pyplot as plt
import postgkyl as pg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

import matplotlib as mpl


import scipy.interpolate 
from scipy.interpolate import RegularGridInterpolator
import scipy.integrate as sci

plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'



def fix_gridvals(grid):
    """Output file grids have cell-edge coordinates by default, but the values are at cell centers.
    This function will return the cell-center coordinate grid.
    Usage: cell_center_grid = fix_gridvals(cell_edge_grid)
    """
    grid = np.array(grid).squeeze()
    grid = (grid[0:-1] + grid[1:])/2
    return grid




frame = int(sys.argv[1])
mom_data = {}
raw_mom_data = {}

mp = 1.67262192e-27
me = 9.1093837e-31
eV = 1.602e-19
B0=2.51

masses = {}
masses["elc"] = me
masses["ion"] = 2.014*mp
masses["molecule"] = 2.0*2.014*mp

# Set up names for data loading
#sim_dir = './simdata/step14_try1/'
sim_dir = './'
base_name = sim_dir+'hstep26'
bmin, bmax = 0, 8
sim_names = ['%s_b%d'%(base_name,i) for i in range(bmin,bmax)]

sim_labels= ['b%d'%i for i in range(bmin,bmax)]

species_list = ["elc", "ion", "molecule"]
n_species = len(species_list)

Rlist = []
Zlist = []
mom_data_list = []
dist_data_list = []
for isim, sim_name in enumerate(sim_names):

    mom_data = {}
    dist_data= {}

    #Load geometry

    bdata = pg.GData('%s-bmag.gkyl'%sim_name)
    grid,val = pg.data.GInterpModal(bdata,poly_order=1,basis_type='ms').interpolate(0)
    mom_data["B"] = val.squeeze()
    
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
            ci+=1
    
    jdata = pg.GData('%s-jacobgeo.gkyl'%sim_name)
    grid,val = pg.data.GInterpModal(jdata,poly_order=1,basis_type='ms').interpolate(0)
    mom_data["J"] = val.squeeze()
    geo_fac = 1/mom_data["J"]/mom_data["B"]

    #Load grid data
    node_data = pg.GData(sim_name+"-nodes.gkyl")
    vals = node_data.get_values()
    R = vals[:,:,0]
    Z = vals[:,:,1]
    PHI = vals[:,:,2]

    temp_nodal_grid = node_data.get_grid()
    nodal_grid = []
    for d in range(0,len(temp_nodal_grid)):
        nodal_grid.append( np.linspace(temp_nodal_grid[d][0], temp_nodal_grid[d][-1], len(temp_nodal_grid[d])-1) )

    Rinterpolator = RegularGridInterpolator((nodal_grid[0], nodal_grid[1]), R)
    Zinterpolator = RegularGridInterpolator((nodal_grid[0], nodal_grid[1]), Z)
    g0, g1 = np.meshgrid(grid[0], grid[1])
    R = Rinterpolator((g0,g1))
    Z = Zinterpolator((g0, g1))

    g0i, g1i = np.meshgrid(fix_gridvals(grid[0]), fix_gridvals(grid[1]))
    Ri = Rinterpolator((g0i,g1i))
    Zi = Zinterpolator((g0i, g1i))
    mom_data["Ri"] = Ri.T
    mom_data["Zi"] = Zi.T

    Rlist.append(R.T)
    Zlist.append(Z.T)

    #Load moment data
    for species in ["elc", "ion", "molecule"]:
        for mom in ["M0", "M1", "M2", "M2par", "M2perp", "M3par", "M3perp"]:
            mdata = pg.GData('%s-%s_%s_%d.gkyl'%(sim_name, species,mom,frame))
            grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
            x = fix_gridvals(grid[0])
            z = fix_gridvals(grid[1])
            val = val.squeeze()
            mom_data[species+mom] = val
        # Set interpolated moment data
        mom_data[species+"Temp"] =  (masses[species]/3) * (mom_data[species+"M2"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
        mom_data[species+"Tpar"] =  (masses[species]) * (mom_data[species+"M2par"] - mom_data[species+"M1"]**2 / mom_data[species+"M0"])/mom_data[species+"M0"] / eV
        mom_data[species+"Tperp"] =  (masses[species]/2) * (mom_data[species+"M2perp"])/mom_data[species+"M0"] / eV
        mom_data[species+"Q"] =  masses[species]/2 * (mom_data[species+"M3par"] + mom_data[species+"M3perp"])
        mom_data[species+"Upar"] =  mom_data[species+"M1"]/mom_data[species+"M0"]
    
    mom_data["Qtot"] =  mom_data["elcQ"]+mom_data["ionQ"]
    
    # Calculate sound speed for ion species
    for species in ["ion" ]:
        mom_data[species+"cs"] = np.sqrt( (mom_data["elcTemp"]*eV + mom_data[species+"Temp"]*eV) / masses[species])
    
    for species in ["ion", "elc" ]:
        mom_data[species+"normUpar"] = mom_data[species+"Upar"]/mom_data["ioncs"]


    # Load the potential
    mdata = pg.GData('%s-field_%d.gkyl'%(sim_name, frame))
    grid,val = pg.data.GInterpModal(mdata,poly_order=1,basis_type='ms').interpolate(0)
    x = fix_gridvals(grid[0])
    z = fix_gridvals(grid[1])
    val = val.squeeze()
    mom_data["phi"] = val

    #Save grids
    mom_data["x"] = x
    mom_data["z"] = z



    # Append data
    mom_data_list.append(mom_data)

    #Distribution Functions
    for species in species_list:
        jvdata = pg.GData('%s-%s_jacobvel.gkyl'%(sim_name, species), mapc2p_vel_name='%s-%s_mapc2p_vel.gkyl'%(sim_name, species))
        ddata = pg.GData('%s-%s_%d.gkyl'%(sim_name, species,frame), mapc2p_vel_name='%s-%s_mapc2p_vel.gkyl'%(sim_name, species))
        ddata._values = ddata.get_values()/jvdata.get_values()
        grid,val = pg.data.GInterpModal(ddata,poly_order=1,basis_type='gkhyb').interpolate()
        val = val.squeeze()
        dist_data[species] = val/mom_data["J"][:,:,None,None]
        #_, _, dist_data[species +"vpar"], dist_data[species +"mu"] = fix_gridvals(grid)
        dist_data[species +"vpar"], dist_data[species +"mu"] = grid[2], grid[3]
        dist_data[species+"mu"][0] = 0.0

    dist_data_list.append(dist_data)


#Concatenate some of the data so we can take derivatives:
omp_data = {}
for species in species_list:
    omp_data[species] = np.concatenate((dist_data_list[6][species], dist_data_list[2][species]), axis=0)
    omp_data["B"] = np.concatenate((mom_data_list[6]["B"], mom_data_list[2]["B"]), axis=0)
    omp_data["x"] = np.concatenate((mom_data_list[6]["x"], mom_data_list[2]["x"]), axis=0)
    #omp_data[species + "dfdx"] = np.gradient(omp_data[species], omp_data["x"], axis = 0, edge_order=2)
    omp_data[species + "dfdx"] = np.diff(omp_data[species], axis = 0)/np.diff( omp_data["x"])[:,None,None,None]
    omp_data[species+"mu"] = dist_data_list[2][species+"mu"]
    omp_data[species+"vpar"] = dist_data_list[2][species+"vpar"]
omp_data["zmid"] = omp_data["elc"].shape[1]-1
omp_data["xsep"] = dist_data_list[6][species].shape[0]

imp_data = {}
for species in species_list:
    imp_data[species] = np.concatenate((dist_data_list[7][species], dist_data_list[3][species]), axis=0)
    imp_data["B"] = np.concatenate((mom_data_list[7]["B"], mom_data_list[3]["B"]), axis=0)
    imp_data["x"] = np.concatenate((mom_data_list[7]["x"], mom_data_list[3]["x"]), axis=0)
    #imp_data[species + "dfdx"] = np.gradient(imp_data[species], imp_data["x"], axis = 0, edge_order=2)
    imp_data[species + "dfdx"] = np.diff(imp_data[species], axis = 0)/np.diff( imp_data["x"])[:,None,None,None]
    imp_data[species+"mu"] = dist_data_list[3][species+"mu"]
    imp_data[species+"vpar"] = dist_data_list[3][species+"vpar"]
imp_data["zmid"] = 0
imp_data["xsep"] = dist_data_list[7][species].shape[0]

#oleg_data = {}
#for species in species_list:
#    oleg_data[species] = np.concatenate((dist_data_list[4][species], dist_data_list[3][species]), axis=0)
#    oleg_data["B"] = np.concatenate((mom_data_list[4]["B"], mom_data_list[3]["B"]), axis=0)
#    oleg_data["x"] = np.concatenate((mom_data_list[4]["x"], mom_data_list[3]["x"]), axis=0)
#    #oleg_data[species + "dfdx"] = np.gradient(oleg_data[species], oleg_data["x"], axis = 0, edge_order=2)
#    oleg_data[species + "dfdx"] = np.diff(oleg_data[species], axis = 0)/np.diff( oleg_data["x"])[:,None,None,None]
#    oleg_data[species+"mu"] = dist_data_list[3][species+"mu"]
#    oleg_data[species+"vpar"] = dist_data_list[3][species+"vpar"]
#oleg_data["zmid"] = oleg_data["elc"].shape[1]//2
#oleg_data["xsep"] = dist_data_list[4][species].shape[0]
#
#ileg_data = {}
#for species in species_list:
#    ileg_data[species] = np.concatenate((dist_data_list[9][species], dist_data_list[8][species]), axis=0)
#    ileg_data["B"] = np.concatenate((mom_data_list[9]["B"], mom_data_list[8]["B"]), axis=0)
#    ileg_data["x"] = np.concatenate((mom_data_list[9]["x"], mom_data_list[8]["x"]), axis=0)
#    #ileg_data[species + "dfdx"] = np.gradient(ileg_data[species], ileg_data["x"], axis = 0, edge_order=2)
#    ileg_data[species + "dfdx"] = np.diff(ileg_data[species], axis = 0)/np.diff( ileg_data["x"])[:,None,None,None]
#    ileg_data[species+"mu"] = dist_data_list[8][species+"mu"]
#    ileg_data[species+"vpar"] = dist_data_list[8][species+"vpar"]
#ileg_data["zmid"] = oleg_data["elc"].shape[1]//2
#ileg_data["xsep"] = dist_data_list[9][species].shape[0]


#def plot_grad(species, xind_plot, zind_plot):
def plot_grad(species, cat_data):
    xind_plot, zind_plot = cat_data["xsep"]-1, cat_data["zmid"]
    fig, ax = plt.subplots(1, figsize = (8,3))
    # Select the B for mu normalization
    Bplot = cat_data["B"][xind_plot, zind_plot]
    xaxis = cat_data[species+"vpar"]
    yaxis = np.sqrt(cat_data[species+"mu"]*2*Bplot/masses[species])
    xaxis_label = r'$v_\parallel\, [m/s]$'
    im = ax.pcolor(xaxis, yaxis, cat_data[species + "dfdx"][xind_plot, zind_plot,:,:].T, cmap='inferno', shading="gourard")
    ax.set_xlabel(xaxis_label, fontsize=20)
    ax.set_ylabel(r'$v_\perp\, [m/s]$', fontsize = 20)
    ax.axis("equal")
    #ax.set_ylim(yaxis.min(), yaxis.max())
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    dist_label =  r'$df_{%s}/dx$'%(species)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', label = dist_label)
    cbar.set_label(label = dist_label, fontsize=20)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, figsize = (8,3))
    ax.plot(fix_gridvals(xaxis),  cat_data[species + "dfdx"][xind_plot, zind_plot,:,0])
    ax.set_xlabel(xaxis_label, fontsize=20)
    ax.set_ylabel(dist_label + ' at ' + r'$\mu=0$', fontsize=20)
    fig.tight_layout()




# Want to plot temps on same plot and densities on same plot
xidx = 0 
#zidx = [mom_data_list[i]["elcM0"].shape[1]//2 for i in range(bmin,bmax)]
zidx = [0,0,-1,0,-1,-1,-1,0]
Nz = [mom_data_list[i]["elcM0"].shape[1] for i in range(bmin,bmax)]
zidx_xpt = [-1,-1,0,-1,0,0,0,-1]
zidx_down = -4

#xinds = [xidx, xidx+5]
#zind_mid, zind_xpt, zind_down, zind_bt = zidx, zidx+32, zidx+46, zidx+22

xinds = [xidx, xidx+2] # 1.705 and 1.706
#zind_mid, zind_xpt, zind_down = zidx, zidx_xpt, zidx_down




def plot_species(species, bidx, xind, zind_plot):
    zind_mid = zidx[bidx]
    zind_xpt = zidx_xpt[bidx]
    if zind_plot==zind_mid:
        zind_label = 'Midplane'
    elif zind_plot==zind_xpt:
        zind_label = 'X-point'

    if xind==xinds[0]:
        xind_label = 'Inner Edge'
    elif xind == xinds[1]:
        xind_label = 'Farther Out'

    fig, ax = plt.subplots(1, figsize = (8,3))


    # Select the B for mu normalization
    Bplot = mom_data_list[bidx]["B"][xind, zind_plot]
    Bmax = mom_data_list[bidx]["B"][xind, zind_xpt]
    print("Bmax, Bmin, ratio = ", Bmax, Bplot, Bmax/Bplot)
    # Select the delta phi for plotting
    #delta_phi = mom_data['phi'][xinds[j],zind_plot]*eV - mom_data['phi'][xinds[j], zind_plot]*eV
    phiplot = mom_data_list[bidx]["phi"][xind, zind_plot]
    phimax= mom_data_list[bidx]["phi"][xind, zind_xpt]
    
    # Do the electrons
    #if zind_plot == zind_mid:
    xaxis = dist_data_list[bidx][species+"vpar"]
    yaxis = np.sqrt(dist_data_list[bidx][species+"mu"]*2*Bplot/masses[species])
    xaxis_label = r'$v_\parallel\, [m/s]$'
    #maxwellian_line_label = r'$v_\perp = \sqrt{2T/m - v_\parallel^2}$'
    #maxwellian_line_label = 'Contour of a Maxwellian with the same T'
    #else:
    #    xaxis =  dist_data[species+"vpar"]- mom_data[species+'Upar'][xind, zind_plot]
    #    yaxis = np.sqrt(dist_data[species+"mu"]*2*Bplot/masses[species])
    #    xaxis_label = r'$v_\parallel - u_{\parallel}$'
    #    maxwellian_line_label = r'$v_\perp = \sqrt{2T/m - (v_\parallel - u_{\parallel})^2}$'

    im = ax.pcolor(xaxis, yaxis, dist_data_list[bidx][species][xind, zind_plot,:,:].T, cmap='inferno', shading="gourard")
    lin_x = np.linspace(xaxis.min(), xaxis.max(), len(xaxis)*1000)
    if zind_plot == zind_mid:
        maxwellian_line = np.sqrt(2*mom_data_list[bidx][species+'Temp'][xind, zind_plot]*eV/masses[species] - (np.abs(lin_x) - mom_data_list[bidx][species+'Upar'][xind, zind_plot])**2)

        bimaxwellian_line = np.sqrt((2/masses[species] - (lin_x- mom_data_list[bidx][species+'Upar'][xind, zind_plot])**2/(mom_data_list[bidx][species+'Tpar'][xind, zind_plot]*eV)) * mom_data_list[bidx][species+'Tperp'][xind, zind_plot]*eV)
        trapping_line1 = lin_x / np.sqrt(Bmax/Bplot - 1)
        trapping_line2 = lin_x / -np.sqrt(Bmax/Bplot - 1)
        #ax.plot(lin_x, trapping_line1, color='red')
        #ax.plot(lin_x, trapping_line2, color='red')
        print("about to plot trapping line")
        if(species in ["ion", "molecule"]):
            phi_trapping_line1 = np.sqrt( (lin_x**2 + 2*eV/masses[species] * (phiplot - phimax)) / ( Bmax/Bplot - 1) )
        elif(species=="elc"):
            phi_trapping_line1 = np.sqrt( (lin_x**2 - 2*eV/masses["elc"] * (phiplot - phimax)) / ( Bmax/Bplot - 1) )
        #phi_trapping_line2 = lin_x / -np.sqrt(Bmax/Bplot - 1)
        ax.plot(lin_x, phi_trapping_line1, color='red')
        #ax.plot(lin_x, trapping_line2, color='red')
        ax.plot(lin_x, maxwellian_line, color='green', label = "Contour of a Maxwellian with the same T")
        ax.plot(lin_x, bimaxwellian_line, color='blue', label = "Contour of a bi-Maxwellian with the same Tpar and Tperp ")
    else:
        maxwellian_line = np.sqrt(2*mom_data_list[bidx][species+'Temp'][xind, zind_plot]*eV/masses[species] - (lin_x- mom_data_list[bidx][species+'Upar'][xind, zind_plot])**2)

        #bimaxwellian_line = np.sqrt(2/masses[species] - (lin_x- mom_data_list[bidx][species+'Upar'][xind, zind_plot])**2/(mom_data_list[bidx][species+'Tpar'][xind, zind_plot]*eV) * mom_data_list[bidx][species+'Tperp'][xind, zind_plot]*eV)

        ax.plot(lin_x, maxwellian_line, color='green', label = "Contour of a Maxwellian with the same T and " + r'$u_{drift}$')
        #ax.plot(lin_x, bimaxwellian_line, color='blue', label = "Contour of a bi-Maxwellian with the same Tpar and Tperp " + r'$u_{drift}$')
    #ax.plot(xaxis, maxwellian_line, label = maxwellian_line_label, color='green')
    #if zind_plot!=zind_down:
    ax.set_xlabel(xaxis_label, fontsize=20)
    ax.set_ylabel(r'$v_\perp\, [m/s]$', fontsize = 20)
    #else:
    #    ax.set_xlabel(xaxis_label, fontsize=20)
    #    ax.set_ylabel(r'$v_\perp\, [m/s]$', fontsize = 20)
    #ax.legend( loc = "upper left", fontsize=14, bbox_to_anchor=(0.2, 1.2))
    #ax.set_ylim(yaxis.min(), ax.get_ylim()[1])
    ax.axis("equal")
    ax.set_ylim(yaxis.min(), yaxis.max())
    if zind_plot == zind_mid:
        ax.set_xlim(xaxis.min()/2.0, xaxis.max()/2.0)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    #dist_label =  r'$f_{%s}(\psi = %1.3f, z=%1.1f)$'%(species, x[xind], z[zind_plot])
    dist_label =  r'$f_{%s}$'%(species)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', label = dist_label)
    #if zind_plot!=zind_down:
    cbar.set_label(label = dist_label, fontsize=20)
    #else:
    #    cbar.set_label(label = dist_label, fontsize=20)

    #fig.suptitle("Distribution functions at %s for %s"%(zind_label, sim_name))
    fig.tight_layout()
    plt.show()
    #fig.savefig('dist_figures/%s/specialfigures/%s_x%s_z%s.png'%(sim_name,species, xind_label,zind_label))


    fig, ax = plt.subplots(1, figsize = (8,3))
    ax.plot(fix_gridvals(xaxis), dist_data_list[bidx][species][xind, zind_plot,:,0])
    ax.set_xlabel(xaxis_label, fontsize=20)
    ax.set_ylabel(dist_label + ' at ' + r'$\mu=0$', fontsize=20)
    fig.tight_layout()

    return xaxis, yaxis
