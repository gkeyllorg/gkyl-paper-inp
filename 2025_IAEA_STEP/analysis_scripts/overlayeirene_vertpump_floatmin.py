import filecmp
import importlib
import numpy as np
import matplotlib  as mpl
import matplotlib.pyplot  as plt
import sys
import os
target_dir = "/pscratch/sd/a/akshukla/GK-Neutral_coupling/common/"
sys.path.insert(0, target_dir)
import eireneIO
import B2IO as b2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import postgkyl as pg

sim_dir = '../' + str(sys.argv[1]) + '/'
frame = str(sys.argv[2])
filepath = sim_dir+'eirene_history/'+frame
edat = eireneIO.eirene(filepath)
edat.triangle_mesh.calc_incenter()
import matplotlib as mpl
atoms = edat.fort46['pdena']
molecules = edat.fort46['pdenm']
ions = edat.fort46['pdeni']


ion_energy = edat.fort46['edeni']
ion_momentum = edat.fort46['vxdeni']


eR = edat.triangle_mesh.incenter[:,0]
eZ = edat.triangle_mesh.incenter[:,1]


fig1, ax1 = plt.subplots(figsize = (5,5))
norm=mpl.colors.LogNorm(vmin=molecules.min(), vmax=molecules.max())
im = ax1.scatter(eR,eZ,c=molecules,cmap='inferno', norm=norm)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.0)
cbar = fig1.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label(r'$n_{D2}\, [m^{-3}]$', fontsize=16)
ax1.axis("tight")
ax1.axis("image")
ax1.set_xlabel("R [m]", fontsize=16)
ax1.set_ylabel("Z [m]", fontsize=16)
ax1.set_xlim(1.2,6.4)
ax1.set_ylim(-8.9,0.0)
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize = (5,5))
norm=mpl.colors.LogNorm(vmin=atoms.min(), vmax=atoms.max())
im = ax2.scatter(eR,eZ,c=atoms,cmap='inferno', norm=norm)
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.0)
cbar = fig2.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label(r'$n_{D}\, [m^{-3}]$', fontsize=16)
ax2.axis("tight")
ax2.axis("image")
ax2.set_xlabel("R [m]", fontsize=16)
ax2.set_ylabel("Z [m]", fontsize=16)
ax2.set_xlim(1.2,6.4)
ax2.set_ylim(-8.9,0.0)
fig2.tight_layout()

#Plot eirene wall
#filepath = "./step_full_device_D+D2_walled_off/gkeyll_w_2D2/"
edat = eireneIO.eirene(filepath)

wallgeo = edat.fort44['wld']['wall_geometry'][:,:]
R1, Z1, R2, Z2 = wallgeo[0], wallgeo[1], wallgeo[2], wallgeo[3]
artifact_indices1 = np.where(
    ((R1 < 1.64) &  (Z1 < -6.3)) | 
    ((R1 < 1.8) &  (Z1 > 7.0)) | 
    ((R1 > 5.84) & (Z1 > 8.)) |
    ((R1 > 5.84) & (Z1 < -8.)) | 
    (Z1 < -8.57)
)[0]

artifact_indices2 = np.where(
    ((R2 < 1.64) &  (Z2 < -6.3)) | 
    ((R2 < 1.8) &  (Z2 > 7.0)) | 
    ((R2 > 5.84) & (Z2 > 8.)) |
    ((R2 > 5.84) & (Z2 < -8.)) | 
    (Z1 < -8.57)
)[0]

artifact_indices = np.union1d(artifact_indices1, artifact_indices2)

R1 = np.delete(R1, artifact_indices)
Z1 = np.delete(Z1, artifact_indices)
R2 = np.delete(R2, artifact_indices)
Z2 = np.delete(Z2, artifact_indices)
# Iterate through each segment and plot
for i in range(len(R1)):
    ax1.plot([R1[i], R2[i]], [Z1[i], Z2[i]], 'tab:green', lw=1.5)
    ax2.plot([R1[i], R2[i]], [Z1[i], Z2[i]], 'tab:green', lw=1.5)

#Pump
for i in [16]:
    ax1.plot([R1[i], R2[i]], [Z1[i], Z2[i]], 'b', lw=1.5)
    ax2.plot([R1[i], R2[i]], [Z1[i], Z2[i]], 'b', lw=1.5)

#Plot Gkeyll bounds
baseName = sim_dir+'hstep27'
bmin = 0
bmax = 8
simNames = ['%s_b%d'%(baseName,i) for i in range(bmin,bmax)]
Rlist = []
Zlist = []
for i, simName in enumerate(simNames):
    data = pg.GData(simName+"-nodes.gkyl")
    vals = data.get_values()
    R1 = vals[:,:,0]
    Z1 = vals[:,:,1]
    phi = vals[:,:,2]

    mc2pdata = pg.GData(simName+ "-mapc2p_deflated.gkyl")
    _ ,R2 = pg.data.GInterpModal(mc2pdata,poly_order=1,basis_type='ms').interpolate(0)
    _ ,Z2 = pg.data.GInterpModal(mc2pdata,poly_order=1,basis_type='ms').interpolate(1)
    R2 = R2.squeeze()
    Z2 = Z2.squeeze()

    if i in [0,5,6,7]:
        xidx = -1
    else:
        xidx = 0

    if i in [0,1]:
        R3 = R1[:, 0]
        Z3 = Z1[:, 0]
    if i in [4,5]:
        R3 = R1[:, -1]
        Z3 = Z1[:, -1]

    R1 = R1[xidx]
    Z1 = Z1[xidx]
    R2 = R2[xidx]
    Z2 = Z2[xidx]

    Rall = np.r_[R1[0], R2, R1[-1]]
    Zall = np.r_[Z1[0], Z2, Z1[-1]]

    Rall = np.empty(len(R1) + len(R2), dtype=R1.dtype)
    Rall[0::3] = R1
    mask = np.ones(len(Rall), dtype=bool)
    mask[0::3] = False
    Rall[mask] = R2

    Zall = np.empty(len(Z1) + len(Z2), dtype=Z1.dtype)
    Zall[0::3] = Z1
    mask = np.ones(len(Zall), dtype=bool)
    mask[0::3] = False
    Zall[mask] = Z2

    ax1.plot(Rall, Zall, color='k', linewidth=0.7)
    ax2.plot(Rall, Zall, color='k', linewidth=0.7)

    if i in [0,1,4,5]:
        ax1.plot(R3,Z3, color = 'r')
        ax2.plot(R3,Z3, color = 'r')


#Plot separatrix
psid = pg.GData(sim_dir+"step_psi.gkyl")
interp = pg.GInterpModal(psid,2,"mt")
grid, psi = interp.interpolate()
for d in range(len(grid)):
    grid[d] = 0.5*(grid[d][:-1] + grid[d][1:])
psisep = 1.5093065418975686
#ax1.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles = "dashed")
#ax2.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles = "dashed")

#ax1.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep-0.1, psisep+0.05], colors="k")
#ax2.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep-0.1, psisep+0.05], colors="k")




div_handle = mlines.Line2D([],[], color = 'r', label = "Plate")
sep_handle = mlines.Line2D([],[], color = 'r', label = "Separatrix", linestyle = "dashed")
pump_handle = mlines.Line2D([],[], color = 'b', label = "Pump")
handles= [sep_handle, div_handle, pump_handle]
ax1.legend(handles = handles, loc = "upper center")
ax2.legend(handles = handles, loc = "upper center")




#fig1.savefig('/global/homes/a/akshukla/thesisplots/overlaymolD.png', dpi=300)
#fig2.savefig('/global/homes/a/akshukla/thesisplots/overlayatomD.png', dpi=300)
