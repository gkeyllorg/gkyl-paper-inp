import filecmp
import importlib
import numpy as np
import matplotlib  as mpl
import matplotlib.pyplot  as plt
from matplotlib.collections import LineCollection
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
frame = int(sys.argv[2])

psid = pg.GData(sim_dir+"step_psi.gkyl")
interp = pg.GInterpModal(psid,2,"mt")
grid, psi = interp.interpolate()

for d in range(len(grid)):
    grid[d] = 0.5*(grid[d][:-1] + grid[d][1:])

psisep = 1.5093065418975686
win=0.05
wout=0.1
wcore=0.1
wpf=0.05
psiin = psisep -win
psiout = psisep - wout
psicore = psisep + wcore
psipf = psisep + wpf

fig, ax = plt.subplots(figsize = (4,9))


#Plot Separatrix
ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psisep], colors="r", linestyles='dashed')
#ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psiin], colors="b")
#ax.contour(grid[0], grid[1], psi[:,:,0].transpose(), levels=np.r_[psiout], colors="g")


colors = ["tab:orange","tab:blue","tab:green", "tab:brown", "tab:purple"]
baseName = sim_dir+'hstep27'
bmin = 0
bmax = 8
simNames = ['%s_b%d'%(baseName,i) for i in range(bmin,bmax)]
Rlist = []
Zlist = []
for i, simName in enumerate(simNames):
    data = pg.GData(simName+"-nodes.gkyl")
    vals = data.get_values()
    R = vals[:,:,0]
    Z = vals[:,:,1]
    phi = vals[:,:,2]

    #Construct lines
    segs1 = np.stack((R,Z), axis=2)
    segs2 = segs1.transpose(1,0,2)

    #Plot all nodes and cell boundaries
    #ax.scatter(R,Z, marker=".")
    ax.add_collection(LineCollection(segs1, linewidth=0.4, color = colors[i%len(colors)]))
    ax.add_collection(LineCollection(segs2, linewidth=0.4, color = colors[i%len(colors)]))
    ax.plot(R,Z,marker=".", color="k", linestyle="none", markersize=2.0)


    Rlist.append(R)
    Zlist.append(Z)

#Plot the upper half:
for i, simName in enumerate(simNames):
    data = pg.GData(simName+"-nodes.gkyl")
    vals = data.get_values()
    R = vals[:,:,0]
    Z = -vals[:,:,1]
    phi = vals[:,:,2]

    #Construct lines
    segs1 = np.stack((R,Z), axis=2)
    segs2 = segs1.transpose(1,0,2)

    #Plot all nodes and cell boundaries
    #ax.scatter(R,Z, marker=".")
    ax.add_collection(LineCollection(segs1, linewidth=0.4, color = colors[i%len(colors)]))
    ax.add_collection(LineCollection(segs2, linewidth=0.4, color = colors[i%len(colors)]))
    ax.plot(R,Z,marker=".", color="k", linestyle="none", markersize=2.0)


    Rlist.append(R)
    Zlist.append(Z)




filepath = sim_dir + "eirene_history/" + str(frame) + '/'
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
    ax.plot([R1[i], R2[i]], [Z1[i], Z2[i]], 'tab:green', lw=1.5)

#Pump
for i in [16]:
    ax.plot([R1[i], R2[i]], [Z1[i], Z2[i]], 'b', lw=1.5)

# Formatting for Tokamak geometry
#ax.grid(True, alpha=0.3)

#Draw divertor plates
for bidx in [4,5]:
    ax.plot(Rlist[bidx][:,-1], Zlist[bidx][:,-1], color='r', linewidth=3.0)
    ax.plot(Rlist[bidx][:,-1], -Zlist[bidx][:,-1], color='r', linewidth=3.0)

for bidx in [0,1]:
    ax.plot(Rlist[bidx][:,0], Zlist[bidx][:,0], color='r', linewidth=3.0)
    ax.plot(Rlist[bidx][:,0], -Zlist[bidx][:,0], color='r', linewidth=3.0)




div_handle = mlines.Line2D([],[], color = 'r', label = "Divertor", linewidth=3.0)
pump_handle = mlines.Line2D([],[], color = 'b', label = "Pump", linewidth=1.5)
sep_handle = mlines.Line2D([],[], color = 'r', label = "Separatrix", linestyle="dashed")
wall_handle = mlines.Line2D([],[], color = 'tab:green', label = "Wall")
handles= [sep_handle, div_handle, pump_handle, wall_handle]
ax.grid()
ax.set_xlabel("R [m]", fontsize=20)
ax.set_ylabel("Z [m]", fontsize=20)
#ax.axis("tight")
#ax.axis("image")
#ax.set_aspect('equal')
ax.legend(handles = handles, loc = "upper center", fontsize=20)
ax.set_ylim(-8.7,0)
#ax.set_xlim(0.4,1.9)
#ax.set_ylim(-8.6,8.6)
fig.tight_layout()
plt.show()


