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

def sum_source(edat, source):
    atemp = (edat.fort46['volumes'].flatten() * source.flatten())
    atemp[np.isnan(atemp)] = 0.0
    return np.sum(atemp) * 1e6/eV


sim_dir = '../' + str(sys.argv[1]) + '/'
frame = str(sys.argv[2])
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
edat.load_extra_forts(filepath)
eV=1.602e-19

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


dpimp = edat.fort44['wld']['wldpp'][:,0,0].sum()


print("\nD emitted = %g"%dflux)
print("D created by reactions = %g"%dvolsource)
print("D lost reactions = %g"%dvolsink)
print("D pumped = %g"%dpump)
print("D pumped / (D emitted + D created) = %g"%(dpump/(dvolsource+dflux)))

print("\nD2 emitted = %g"%d2flux)
print("D2 created by reactions = %g"%d2volsource)
print("D2 lost reactions = %g"%d2volsink)
print("D2 pumped = %g"%d2pump)
print("D2 pumped / (D2 emitted + D2 created) = %g"%(d2pump/(d2volsource+d2flux)))

print("\nD+ Created = %g"%dpvol)
print("\nD2+ Created = %g"%d2pvol)

print("\nD+ Impinging = %g"%dpimp)

print("\nD+ Created/D+ impinging = %g %%"%(dpvol/dpimp))



