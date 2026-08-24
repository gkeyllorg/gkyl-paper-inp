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


sim_dir = '../' + str(sys.argv[1]) + '/'
frame = str(sys.argv[2])
#filepath = sim_dir+"eirene_history/%s"%frame
filepath = sim_dir+frame
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

eZcut = eZ[eZ<-8.5]
eRcut = eR[eZ<-8.5]
linidx = np.argsort(eRcut)

plt.figure()
norm=mpl.colors.LogNorm(vmin=1e10, vmax=molecules.max())
plt.scatter(eR,eZ,c=molecules,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("Molecular D2")
#lineout
#moleculescut = molecules[eZ<-8.5]
#plt.figure()
#plt.plot(eRcut[linidx], moleculescut[linidx])
#plt.title("D2")

plt.figure()
norm=mpl.colors.LogNorm(vmin=1e10, vmax=atoms.max())
plt.scatter(eR,eZ,c=atoms,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("Atoms D")
#lineout
#atomscut = atoms[eZ<-8.5]
#plt.figure()
#plt.plot(eRcut[linidx], atomscut[linidx])
#plt.title("D")

# Plot test ion quantities
# plt.figure()
# norm=mpl.colors.LogNorm(vmin=1e1, vmax=ions.max())
# plt.scatter(eR,eZ,c=ions,cmap='inferno', norm=norm)
# plt.colorbar()
# plt.title("Ions D2")
# 
# plt.figure()
# plt.scatter(eR,eZ,c=ion_energy,cmap='inferno', norm=norm)
# plt.colorbar()
# plt.title("Energy Ions D2")
# 
# plt.figure()
# plt.scatter(eR,eZ,c=ion_momentum,cmap='inferno')
# plt.colorbar()
# plt.title("Momentum Ions D2")


edat.load_extra_forts(filepath)
eV=1.602e-19
source = edat.sources["particle"]['D2+']["SUM"]/eV*1e6
# Plot sources

plt.figure()
#norm=mpl.colors.SymLogNorm(vmin=source.min(), vmax=source.max(), linthresh=1.0)
norm=mpl.colors.SymLogNorm(vmin=1e10, vmax=source.max(), linthresh=1.0)
plt.scatter(eR,eZ,c=source,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("Particle Source D2+")

source = edat.sources["particle"]['D+']["SUM"]/eV*1e6
# Plot sources

plt.figure()
#norm=mpl.colors.SymLogNorm(vmin=source.min(), vmax=source.max(), linthresh=1.0)
norm=mpl.colors.SymLogNorm(vmin=1e10, vmax=source.max(), linthresh=1.0)
plt.scatter(eR,eZ,c=source,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("Particle Source D+")



# D2 sink
plt.figure()
source = edat.sources["particle"]['D2']["SUM"]/eV*1e6
norm=mpl.colors.SymLogNorm(vmin=-1e22, vmax=-1e18, linthresh=1.0)
plt.scatter(eR,eZ,c=source,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("Particle Sink D2")
# Try some lineouts
#sourcecut = source[eZ<-8.5]
#plt.figure()
#plt.plot(eRcut[linidx], sourcecut[linidx])
#plt.title("Particle Sink D2")

# D sink
plt.figure()
source = edat.sources["particle"]['D']["SUM"]/eV*1e6
norm=mpl.colors.SymLogNorm(vmin=-1e19, vmax=1e22, linthresh=1.0)
plt.scatter(eR,eZ,c=source,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("Particle Sink D")

# 
# esource = edat.energy_source['TEST IONS']/eV*1e6
# plt.figure()
# norm=mpl.colors.Normalize(vmin=esource.min(), vmax=esource.max())
# plt.scatter(eR,eZ,c=esource,cmap='inferno', norm=norm)
# plt.colorbar()
# plt.title("ENERGY Source D2+")


#Plot wall fluxes (impinging)
aflux = edat.fort44['wld']['wldna']
mflux = edat.fort44['wld']['wldnm']
wallgeo = edat.fort44['wld']['wall_geometry']
R1 = wallgeo[0]
Z1 = wallgeo[1]

norm=mpl.colors.LogNorm(vmin=5e13, vmax=5e21)
plt.figure()
plt.scatter(R1,Z1,c=aflux[:516,:,0], cmap='inferno', norm=norm)
plt.colorbar()
plt.title('Atom flux')

plt.figure()
plt.scatter(R1,Z1,c=mflux[:516,:,0], cmap='inferno', norm=norm)
plt.colorbar()
plt.title('Mol flux')


def clean_temp(arr):
    min = np.min(arr[(~np.isnan(arr)) & (~np.isinf(arr)) & (arr!=0)])
    max = np.max(arr[(~np.isnan(arr)) & (~np.isinf(arr)) & (arr!=0)])
    arr[np.isnan(arr)] = min

# Plot some neutral temperatures
molenergy = edat.fort46['edenm']/molecules/eV
clean_temp(molenergy)

plt.figure()
norm=mpl.colors.LogNorm(vmin=molenergy.min(), vmax=200)
plt.scatter(eR,eZ,c=molenergy,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("D2 Temp")
#lineout
#moltempcut = molenergy[eZ<-8.5]
#plt.figure()
#plt.plot(eRcut[linidx], moltempcut[linidx])
#plt.title("D2 Temp")

atomenergy = edat.fort46['edena']/atoms/eV
clean_temp(atomenergy)

plt.figure()
norm=mpl.colors.LogNorm(vmin=atomenergy.min(), vmax=500)
plt.scatter(eR,eZ,c=atomenergy,cmap='inferno', norm=norm)
plt.colorbar()
plt.title("D Temp")
#lineout
#atomtempcut = atomenergy[eZ<-8.5]
#plt.figure()
#plt.plot(eRcut[linidx], atomtempcut[linidx])
#plt.title("D Temp")

