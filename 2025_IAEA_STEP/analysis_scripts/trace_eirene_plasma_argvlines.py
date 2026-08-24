import filecmp
import importlib
import numpy as np
import matplotlib  as mpl
import matplotlib.pyplot  as plt
import sys
import os


sim_dir = str(sys.argv[1])

target_dir = "../GK-Neutral_coupling/common/"
sys.path.insert(0, target_dir)
import eireneIO
import B2IO as b2

start = int(sys.argv[2])
stop = int(sys.argv[3])
step = int(sys.argv[4])

eV=1.602e-19

frames = np.arange(start, stop+1, step)
# Remove frames with missing data
remove_frames = []
for frame in frames:
    filepath = "../%s/eirene_history/%d/fort.100"%(sim_dir, frame)
    if not os.path.exists(filepath):
        remove_frames.append(frame)



frames = frames[~np.isin(frames,remove_frames)]
time_eirene = frames*1e-5


d2_pump = np.zeros(len(frames))
d2_vol = np.zeros(len(frames))
d2_volsource = np.zeros(len(frames))
d2_volsink = np.zeros(len(frames))
d2_flux = np.zeros(len(frames))
total_d2 = np.zeros(len(frames))

d_pump = np.zeros(len(frames))
d_vol = np.zeros(len(frames))
d_volsource = np.zeros(len(frames))
d_volsink = np.zeros(len(frames))
d_flux = np.zeros(len(frames))
total_d = np.zeros(len(frames))

dp_vol = np.zeros(len(frames))
d2p_vol = np.zeros(len(frames))


dp_imp = np.zeros(len(frames))


for i, frame in enumerate(frames):
    print("Frame is %d"%frame)
    filepath = "../%s/eirene_history/%d/"%(sim_dir, frame)
    
    edat = eireneIO.eirene(filepath)
    edat.triangle_mesh.calc_incenter()
    atoms = edat.fort46['pdena']
    molecules = edat.fort46['pdenm']
    ions = edat.fort46['pdeni']
    
    edat.load_extra_forts(filepath)
    
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

    d2_pump[i] = d2pump
    d2_flux[i] = d2flux
    d2_vol[i] = d2vol
    d2_volsource[i] = d2volsource
    d2_volsink[i] = d2volsink
    
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

    d_pump[i] = dpump
    d_flux[i] = dflux
    d_vol[i] = dvol
    d_volsource[i] = dvolsource
    d_volsink[i] = dvolsink


    total_d2[i] = np.sum(edat.fort46['volumes']*molecules)
    total_d[i] = np.sum(edat.fort46['volumes']*atoms)

    #Calculate D+ source rate
    atemp = (edat.fort46['volumes'].flatten() * edat.sources["particle"]['D+']["SUM"].flatten())
    atemp[np.isnan(atemp)] = 0.0
    dpvol = np.sum(atemp) * 1e6/eV
    dp_vol[i] = dpvol

    #Calculate D2+ source rate
    atemp = (edat.fort46['volumes'].flatten() * edat.sources["particle"]['D2+']["SUM"].flatten())
    atemp[np.isnan(atemp)] = 0.0
    d2pvol = np.sum(atemp) * 1e6/eV
    d2p_vol[i] = d2pvol


    # Calculate impinging D:
    dp_imp[i] = edat.fort44['wld']['wldpp'][:,0,0].sum()





fig, ax  = plt.subplots(3,3)
ax[0,0].plot(time_eirene, d_flux, label = 'D emitted' )
ax[0,0].plot(time_eirene, d_volsource, label = 'D created by reactions' )
ax[0,0].plot(time_eirene, d_volsink, label = 'D lost to reactions' )
ax[0,0].plot(time_eirene, d_pump, label = 'D pumped' )
ax[0,1].plot(time_eirene, d_pump/(d_volsource+d_flux), label = 'D pumped/ (D emitted + D created)' )
ax[0,2].plot(time_eirene, total_d, label ="Total D")
ax[0,0].legend()
ax[0,1].legend()
ax[0,2].legend()

ax[1,0].plot(time_eirene, d2_flux, label = 'D2 emitted' )
ax[1,0].plot(time_eirene, d2_volsource, label = 'D2 created by reactions' )
ax[1,0].plot(time_eirene, d2_volsink, label = 'D2 lost to reactions' )
ax[1,0].plot(time_eirene, d2_pump, label = 'D2 pumped' )
ax[1,1].plot(time_eirene, d2_pump/(d2_volsource+d2_flux), label = 'D2 pumped/ (D2 emitted + D2 created)' )
ax[1,2].plot(time_eirene, total_d2, label ="Total D2")
ax[1,0].legend()
ax[1,1].legend()
ax[1,2].legend()

ax[2,0].plot(time_eirene, dp_vol, label = 'D+ Created' )
ax[2,0].plot(time_eirene, d2p_vol, label = 'D2+ Created' )
ax[2,0].legend()

ax[2,1].plot(time_eirene, dp_imp, label = 'D+ Impinging' )
ax[2,1].legend()


frames_vline = np.r_[int(sys.argv[5])]
times_vline = frames_vline*1e-5

for i in range(3):
    for j in range(3):
        for k in range(len(frames_vline)):
            ax[i,j].axvline(times_vline[k], color='k', linestyle='dashed')

fig.suptitle(sim_dir)
