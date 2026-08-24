# coding: utf-8
mpl.rcParams['axes.labelsize'] = 'x-large'
%run load_conf_space.py hrdivertor4 650 170
cuwallfix=sim_data.copy()
%run load_conf_space.py LiWallFix 650 190
liwallfix=sim_data.copy()
%run load_conf_space.py HorizontalPump 650 165
horizontalpump=sim_data.copy()
%run load_conf_space.py LiPlateFix 650 165
liplatefix=sim_data.copy()

name_map = {}
name_map['hrdivertor4'] = 'C1'
name_map['LiWallFix'] = 'C2'
name_map['HorizontalPump'] = 'C3'
name_map['LiPlateFix'] = 'C4'


sims = [cuwallfix, liwallfix, horizontalpump, liplatefix]
fig, ax  = plt.subplots(2,2, figsize=(16,8))
for sd in sims:
    ax[0, 0].plot(sd['R_OMP'], sd['Ti_OMP'], label=name_map[sd['name']])
    ax[0, 0].set_ylabel(r'$T_i\, [eV]$')
    ax[0, 0].set_ylim(0,1e4)

    ax[0, 1].plot(sd['R_OMP'], sd['Te_OMP'], label=name_map[sd['name']])
    ax[0, 1].set_ylabel(r'$T_e\, [eV]$')
    ax[0, 1].set_ylim(0,1e4)


    ax[1, 0].plot(sd['R_OMP'], sd['ni_OMP'], label=name_map[sd['name']])
    ax[1, 0].set_ylabel(r'$n_i\, [m^{-3}]$')

    ax[1, 1].plot(sd['R_OMP'], sd['ne_OMP'], label=name_map[sd['name']])
    ax[1, 1].set_ylabel(r'$n_e\, [m^{-3}]$')


for axi in ax.flat:
    axi.set_xlabel(r'$R\, [m]$')
    axi.legend()
    axi.axvline(x=sd['R_OMP_sep'], color='grey', linestyle='dashed')
#fig.suptitle("Profiles at OMP")
fig.tight_layout()


fig, ax  = plt.subplots(3,2, figsize=(16,8))
for sd in sims:
    ax[0, 0].plot(sd['R_OP'], sd['Ti_OP'], label=name_map[sd['name']])
    ax[0, 0].set_ylabel(r'$T_i\, [eV]$')

    ax[0, 1].plot(sd['R_OP'], sd['Te_OP'], label=name_map[sd['name']])
    ax[0, 1].set_ylabel(r'$T_e\, [eV]$')


    ax[1, 0].plot(sd['R_OP'], sd['ni_OP'], label=name_map[sd['name']])
    ax[1, 0].set_ylabel(r'$n_i\, [m^{-3}]$')

    ax[1, 1].plot(sd['R_OP'], sd['ne_OP'], label=name_map[sd['name']])
    ax[1, 1].set_ylabel(r'$n_e\, [m^{-3}]$')

    ax[2, 0].plot(sd['R_OP'], sd['Qi_OP']/1e6, label=name_map[sd['name']])
    ax[2, 0].set_ylabel(r'$Q_{normal,i}\, [MW/m^2]$')

    ax[2, 1].plot(sd['R_OP'], sd['Qe_OP']/1e6, label=name_map[sd['name']])
    ax[2, 1].set_ylabel(r'$Q_{normal,e}\, [MW/m^2]$')




for axi in ax.flat:
    axi.set_xlabel(r'$R\, [m]$')
    axi.legend()
    axi.axvline(x=sd['R_OP_sep'], color='grey', linestyle='dashed')

#fig.suptitle("Profiles at Outboard Plate")
fig.tight_layout()



#Some Individual plots for the paper, etc

#Density at inboard plate
fig, ax  = plt.subplots()
ax.plot(sd['Z_IP'], sd['ne_IP'], label = 'e-')
ax.plot(sd['Z_IP'], sd['ni_IP'], label = 'D+')
ax.axvline(x=sd['Z_IP_sep'], color='grey', linestyle='dashed')
ax.set_xlabel('Z [m]')
ax.set_ylabel(r'$n\, [m^{-3}]$')
ax.legend()
fig.tight_layout()

#Temp at outboard plate
fig, ax  = plt.subplots()
ax.plot(sd['R_OP'], sd['ne_OP'], label = 'e-')
ax.plot(sd['R_OP'], sd['ni_OP'], label = 'D+')
ax.axvline(x=sd['R_OP_sep'], color='grey', linestyle='dashed')
ax.set_xlabel('R [m]')
ax.set_ylabel(r'$n\, [m^{-3}]$')
ax.legend()
fig.tight_layout()

#Temp at inboard plate
fig, ax  = plt.subplots()
ax.plot(sd['Z_IP'], sd['Te_IP'], label = 'e-')
ax.plot(sd['Z_IP'], sd['Ti_IP'], label = 'D+')
ax.axvline(x=sd['Z_IP_sep'], color='grey', linestyle='dashed')
ax.set_xlabel('Z [m]')
ax.set_ylabel(r'$T\, [eV]$')
ax.legend()
fig.tight_layout()

#Temp at outboard plate
fig, ax  = plt.subplots()
ax.plot(sd['R_OP'], sd['Te_OP'], label = 'e-')
ax.plot(sd['R_OP'], sd['Ti_OP'], label = 'D+')
ax.axvline(x=sd['R_OP_sep'], color='grey', linestyle='dashed')
ax.set_xlabel('R [m]')
ax.set_ylabel(r'$T\, [eV]$')
ax.legend()
fig.tight_layout()

#Total Heat Flux at inboard plate
fig, ax  = plt.subplots()
ax.plot(sd['Z_IP'], (sd['Qi_IP']+sd['Qe_IP'])/1e6)
ax.axvline(x=sd['Z_IP_sep'], color='grey', linestyle='dashed')
ax.set_xlabel('Z [m]')
ax.set_ylabel(r'$Q_{normal}\, [MW/m^2]$')

#Total Heat Flux at outboard plate
fig, ax  = plt.subplots()
ax.plot(sd['R_OP'], (sd['Qi_OP']+sd['Qe_OP'])/1e6)
ax.axvline(x=sd['R_OP_sep'], color='grey', linestyle='dashed')
ax.set_xlabel('R [m]')
ax.set_ylabel(r'$Q_{normal}\, [MW/m^2]$')
