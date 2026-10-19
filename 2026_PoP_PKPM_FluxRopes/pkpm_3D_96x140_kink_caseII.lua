-- I0: -1000 A
-- Resolution: 96 x 96 x 140
-- Domain: x,y = [-0.30, 0.30] m; z = [0, 10] m
-- One-rope simulation

local PKPM = G0.PKPM
local Constants = require "Lib.Constants"

-- physical parameters
pi = math.pi
local cFac = 1000 -- Permittivity scaling factor.
local eps0, eV = Constants.EPSILON0*cFac, Constants.ELEMENTARY_CHARGE
local mu0      = Constants.MU0
local m_p = Constants.PROTON_MASS


Te_Ti = 5.0 
temp_elc = 5.*eV
temp_ion = temp_elc / Te_Ti
n0 = 2.0e18

mass_ion = 4*m_p 
charge_ion = eV 
mass_elc = mass_ion / 400. 
charge_elc = -eV 

B0 =  330. / 10000. 

-- Alfven speeds
vAElc = B0/math.sqrt(mu0*n0*mass_elc)

-- Thermal speeds
vtElc = math.sqrt(2.*temp_elc / mass_elc) 
vtIon = math.sqrt(2.*temp_ion / mass_ion) 

-- plasma and cyclotron frequencies
wpe = math.sqrt(charge_ion^2*n0/(eps0*mass_elc))
wpi = math.sqrt(charge_ion^2*n0/(eps0*mass_ion))
omegaCe = charge_ion*B0/mass_elc
omegaCi = charge_ion*B0/mass_ion

-- inertial length
de = vAElc/omegaCe

-- Ion sound gyroradius
rhos = math.sqrt(temp_elc/mass_ion) / omegaCi

-- debye length
debyeElc = vtElc / wpe
debyeIon = vtIon / wpi
debye = (debyeElc^(-1) + debyeIon^(-1))^(-1)

-- Collision lengths use SI epsilon0; eps0 above is scaled by cFac for the field solver.
-- Collision frequencies
rmin_elc = 1/(4*math.pi*Constants.EPSILON0) * charge_elc^2 / temp_elc
rmin_ion = 1/(4*math.pi*Constants.EPSILON0) * charge_ion^2 / temp_ion
rmax = debye
lambda_elc = math.log(rmax/rmin_elc) 
lambda_ion = math.log(rmax/rmin_ion) 

nu_coeff = 0.1
nu_elc = nu_coeff*(4*(2*math.pi)^0.5*n0*charge_elc^4*lambda_elc)/(3*(4*math.pi*Constants.EPSILON0)^2*mass_elc^0.5*temp_elc^1.5) 
nu_ion = nu_coeff*(4*(2*math.pi)^0.5*n0*charge_ion^4*lambda_ion)/(3*(4*math.pi*Constants.EPSILON0)^2*mass_ion^0.5*temp_ion^1.5) 

-- Hyper diffusion coefficients
D_coeff = 2.0e-3
D_scale = (wpe/(2*math.pi))*de^4

-- Simulation parameters
Nx = 96
Ny = 96
Nz = 140
Nvx = 16

Ncx = 16
Ncy = 16
Ncz = 14

Lr = 0.30
Lr_plasma = 0.20
LzSt = 0.0
LzEnd = 10.0
Lz = LzEnd - LzSt

vx_max_ion = 6.0 * vtIon
vx_max_elc = 6.0 * vtElc
poly_order = 1 
basis_type = "serendipity" 
time_stepper = "rk3" 
cfl_frac = 1.0 

-- rope params (Single Rope)
nr = (3.0e18 - n0)
Tr = 17.33*eV
I0 = -1000 -- Current in the single rope.
rs = 0.04445 -- Equivalent profile scale.
-- Center the single rope in XY plane
x01 = 0.0
y01 = 0.0

lo_I0 = -750 -- Reference current for time normalization.
lo_I1 = lo_I0 -- Reference current in the single rope.
lo_Bperp = math.abs(0.64*mu0*lo_I1/(2.*math.pi*rs)) 
lo_omegaCi_perp = charge_ion*lo_Bperp/mass_ion
hi_Bperp = math.abs(0.64*mu0*I0/(2.*math.pi*rs)) 
hi_omegaCi_perp = charge_ion*hi_Bperp/mass_ion
frame_ratio = lo_omegaCi_perp/hi_omegaCi_perp

t_end = 200*frame_ratio/omegaCi 
num_frames = 400
dt_failure_tol = 1.0e-4 
num_failures_max = 20 

-- Write parameters to file
dimsX = 3
dimsV = 1 
polyOrder = poly_order
basis = 'ms'
model = 'pkpm'
suffix = '.gkyl'

fname = os.getenv("SCRATCH") .. "/" .. os.getenv("LUA_INPUT_FILE"):match("([^/\\]+)%.lua$") .. "/" .. os.getenv("LUA_INPUT_FILE"):match("([^/\\]+)%.lua$") .. "-params.txt"
file = io.open(fname, 'w')
if file then
    file:write("eps0,\tmu0,\tB0,\tdimsX,\tdimsV(=0 for fluid),\tpolyOrder,\tbasis,\tmodel(vm,5m,10m),\tfilesuffix (.bp,gkyl)\n")
    file:write(string.format("%g,\t%g,\t%g,\t%d,\t%d,\t%d,\t%s,\t%s,\t%s\n", eps0, mu0, B0, dimsX, dimsV, polyOrder, basis, model, suffix))
    file:write("species,\tmass,\tcharge,\ttemperature,\tdensity\n")
    file:write(string.format("%s,\t%g,\t%g,\t%g,\t%g\n", 'elc', mass_elc, charge_elc, temp_elc, n0))
    file:write(string.format("%s,\t%g,\t%g,\t%g,\t%g\n", 'ion', mass_ion, charge_ion, temp_ion, n0))
    file:close()
end

-- Adapted Perturbation function (mapping to X and Z for kink trigger)
local function noise_init(x,y,z,noiseAmp,kInit,kFinal,lambda)
  local Pi = math.pi
  local sin = math.sin
  local cos = math.cos
  local sqrt = math.sqrt
  local dxNoise, dyNoise = 0.0, 0.0
  local amprand, phaserand = 0.0, 0.0
  local seed = 120387
  
  local r = math.sqrt((x)^2 + (y)^2)
  local mag = math.exp(-(r^2 / lambda^2))*(1-math.exp(-z^2))
  for k = math.floor(kInit), math.floor(kFinal) do
     seed = seed + k
     math.randomseed(seed)
     amprand = 1.0
     phaserand = math.random()

     dxNoise = dxNoise + mag*amprand*cos((2.0*Pi)*k*z/Lz + 2.*Pi*phaserand)
     dyNoise = dyNoise + mag*amprand*sin((2.0*Pi)*k*z/Lz + 2.*Pi*phaserand)
  end
  
  local kdiff = math.floor(kFinal) - math.floor(kInit) + 1.0
  -- Normalize noise by number of modes and scale by noise amplitude
  dxNoise = noiseAmp * dxNoise / kdiff
  dyNoise = noiseAmp * dyNoise / kdiff

  return dxNoise, dyNoise
end

local k_init, k_final, noise_amp = 1.0, 3.0, 0.2*rs
pkpmApp = PKPM.App.new {

  tEnd = t_end,
  nFrame = num_frames,
  dtFailureTol = dt_failure_tol,
  numFailuresMax = num_failures_max,
  lower = {-Lr, -Lr, LzSt },
  upper = {Lr, Lr, LzEnd },
  cells = { Nx, Ny, Nz },
  cflFrac = cfl_frac,

  useExplicitSource = true, 

  basis = basis_type,
  polyOrder = poly_order,
  timeStepper = time_stepper,

  decompCuts = { Ncx, Ncy, Ncz }, 
  periodicDirs = { }, 

  -- Electrons.
  elc = PKPM.Species.new {
    modelID = G0.Model.Default,
    charge = charge_elc, mass = mass_elc,
    
    lower = { -vx_max_elc },
    upper = { vx_max_elc  },
    cells = { Nvx },

    initDist = function (t, xn)
      -- 3D1V: x,y,z are spatial, xn[4] is parallel velocity
      local x, y, z, vx = xn[1], xn[2], xn[3], xn[4]
      local vt = math.sqrt(temp_elc / mass_elc)

      local dx, dy = noise_init(x, y, z, noise_amp, k_init, k_final, 2.0*rs)
      
      local x1 = x - x01 + dx
      local y1 = y - y01 + dy
      local r = math.sqrt(x1^2 + y1^2)
      local r_bg = math.sqrt((x - x01)^2 + (y - y01)^2)
      local r_safe = math.max(r, 1e-10) -- Prevent div by zero at center

      local fac1 = math.exp(-(r^2 / rs^2)) 
      local vtR = math.sqrt(Tr / mass_elc)
      
      local n0B = n0*0.5*(1.0 - math.tanh((r_bg - (Lr_plasma))/rhos)) + 0.1*n0

      -- Random perturbations
      local vx_noise = 0.
      local vy_noise = 0.
      local vz_noise = 0.
      local Bx_noise = 0.
      local By_noise = 0.
      local Bz_noise = 0.

      -- Calculate Center of Mass fluid velocity (ue) for PKPM peculiar frame
      local vdrift_x1_lab = vx_noise
      local vdrift_y1_lab = vy_noise
      local vdrift_z1_lab = (I0) / (charge_elc*pi*rs^2*nr) + vz_noise
      
      local ue_x = (vdrift_x1_lab*nr*fac1) / (n0B + nr*fac1)
      local ue_y = (vdrift_y1_lab*nr*fac1) / (n0B + nr*fac1)
      local ue_z = (vdrift_z1_lab*nr*fac1) / (n0B + nr*fac1)
      
      -- Correct PKPM frame shifts: background drifts backwards, beam is adjusted
      local vdrift_x0 = -ue_x
      local vdrift_x1 = vdrift_x1_lab - ue_x
      local vdrift_y0 = -ue_y
      local vdrift_y1 = vdrift_y1_lab - ue_y
      local vdrift_z0 = -ue_z
      local vdrift_z1 = vdrift_z1_lab - ue_z

      -- Calculate Physical Fields in 3D Cylindrical projected to Cartesian
      local Bphi1 = ((1 - math.exp(-(r^2)/rs^2))*I0*mu0)/(2.*math.pi*r_safe)
      local Bx_phys = -Bphi1 * (y1 / r_safe)
      local By_phys = Bphi1 * (x1 / r_safe)
      
      local dBz = (mu0*n0)/(2*B0)*(temp_elc + temp_ion)*(1 + math.tanh((r_bg - (Lr_plasma))/rhos))
      local Bz_phys = B0 + dBz

      -- Add in perturbation 
      local Bx_gk = Bx_phys + Bx_noise
      local By_gk = By_phys + By_noise
      local Bz_gk = Bz_phys + Bz_noise
      
      local Bmag = math.sqrt(Bx_gk^2 + By_gk^2 + Bz_gk^2)

      -- Project peculiar drift onto B-field (Z dominant)
      local vdrift_par0 = vdrift_x0*(Bx_gk / Bmag) + vdrift_y0*(By_gk / Bmag) + vdrift_z0*(Bz_gk / Bmag)
      local vdrift_par1 = vdrift_x1*(Bx_gk / Bmag) + vdrift_y1*(By_gk / Bmag) + vdrift_z1*(Bz_gk / Bmag)

      local f0B = (n0B / math.sqrt(2.0*pi*vt^2)) * (math.exp(-((vx - vdrift_par0)^2) / (2.0*vt^2)))
      local f01 = (nr*fac1 / math.sqrt(2.0*pi*vtR^2)) * (math.exp(-((vx - vdrift_par1)^2) / (2.0*vtR^2)))
      
      if f0B < 0.0 then f0B = 0.0 end
      if f01 < 0.0 then f01 = 0.0 end

      local f0 = f0B + f01
      local g = (temp_elc*f0 + Tr*f01) / mass_elc

      if f0 < 0.0 then f0 = 0.0 end
      if g < 0.0 then g = 0.0 end

      return f0, g
    end,

    initFluid = function (t, xn)
      local x, y, z = xn[1], xn[2], xn[3]
      local dx, dy = noise_init(x, y, z, noise_amp, k_init, k_final, 2.0*rs)
      
      local x1 = x - x01 + dx
      local y1 = y - y01 + dy
      local r = math.sqrt(x1^2 + y1^2)
      local fac1 = math.exp(-(r^2 / rs^2)) 

      -- Lab frame momentum goes here
      local vdrift_z1 = (I0) / (charge_elc*pi*rs^2*nr) 
      local ezmom1 = vdrift_z1 * nr * fac1 * mass_elc
      
      local vxnoise = 0.
      local vynoise = 0.
      local vznoise = 0.

      local exmom1 = 0.0 + vxnoise
      local eymom1 = 0.0 + vynoise
      ezmom1 = ezmom1 + vznoise

      return exmom1, eymom1, ezmom1 
    end,

    collisions = {
      collisionID = G0.Collisions.LBO,
      selfNu = function (t, xn) return nu_elc end
    },
    diffusion = {
      D = D_scale * D_coeff,
      order = 4
    },
    bcx = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcy = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcz = { G0.SpeciesBc.bcFixedFunc, G0.SpeciesBc.bcCopy }, 
    evolve = true
  },

  -- Ions
  ion = PKPM.Species.new {
    modelID = G0.Model.Default,
    charge = charge_ion, mass = mass_ion,
    
    lower = { -vx_max_ion },
    upper = { vx_max_ion  },
    cells = { Nvx },

    initDist = function (t, xn)
      local x, y, z, vx = xn[1], xn[2], xn[3], xn[4]
      local vt = math.sqrt(temp_ion / mass_ion)
      
      local dx, dy = noise_init(x, y, z, noise_amp, k_init, k_final, 2.0*rs)
      
      local x1 = x - x01 + dx
      local y1 = y - y01 + dy
      local r = math.sqrt(x1^2 + y1^2)
      local r_bg = math.sqrt((x - x01)^2 + (y - y01)^2)
      local r_safe = math.max(r, 1e-10) -- Prevent div by zero at center

      local fac1 = math.exp(-(r^2 / rs^2)) 
      local Tr_ion = Tr/Te_Ti
      local vtR = math.sqrt(Tr_ion / mass_ion)

      local n0B = n0*0.5*(1.0 - math.tanh((r_bg - (Lr_plasma))/rhos)) + 0.1*n0
      
      local f0B = (n0B / math.sqrt(2.0*pi*vt^2)) * (math.exp(-(vx^2) / (2.0*vt^2))) 
      local f01 = (nr*fac1 / math.sqrt(2.0 * pi * vtR^2)) * (math.exp(-(vx^2) / (2.0 * vtR^2))) 
      
      if f0B < 0.0 then f0B = 0.0 end
      if f01 < 0.0 then f01 = 0.0 end

      local f0 = f0B + f01 
      local g = (temp_ion*f0 + Tr_ion*f01) / mass_ion

      if f0 < 0.0 then f0 = 0.0 end
      if g < 0.0 then g = 0.0 end
      
      return f0, g
    end,

    initFluid = function (t, xn)
      return 0.0, 0.0, 0.0
    end,

    collisions = {
      collisionID = G0.Collisions.LBO,
      selfNu = function (t, xn) return nu_ion end
    },
    diffusion = {
      D = D_scale * D_coeff,
      order = 4
    },
    bcx = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcy = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcz = { G0.SpeciesBc.bcFixedFunc, G0.SpeciesBc.bcCopy },
    evolve = true
  },

  -- Field.
  field = PKPM.Field.new {
    epsilon0 = eps0, mu0 = mu0,

    init = function (t, xn)
      local x, y, z = xn[1], xn[2], xn[3]

      local dx, dy = noise_init(x, y, z, noise_amp, k_init, k_final, 2.0*rs)
      
      local x1 = x - x01 + dx
      local y1 = y - y01 + dy
      local r = math.sqrt(x1^2 + y1^2)
      local r_bg = math.sqrt((x - x01)^2 + (y - y01)^2)
      local r_safe = math.max(r, 1e-10)
      
      local Bphi1 = ((1 - math.exp(-(r^2)/rs^2))*I0*mu0)/(2.*math.pi*r_safe)
      
      -- Cartesian projections
      local Bx_phys = -Bphi1 * (y1 / r_safe)
      local By_phys = Bphi1 * (x1 / r_safe)

      local dBz = (mu0*n0)/(2*B0)*(temp_elc + temp_ion)*(1 + math.tanh((r_bg - (Lr_plasma))/rhos))
      local Bz_phys = dBz
      
      -- Add in perturbation
      local Bx_noise = 0.
      local By_noise = 0.
      local Bz_noise = 0.

      local Bx_gk = Bx_phys + Bx_noise
      local By_gk = By_phys + By_noise
      local Bz_gk = Bz_phys + Bz_noise

      return 0.0, 0.0, 0.0, Bx_gk, By_gk, Bz_gk, 0.0, 0.0
    end,

    sigma = function (t, xn)
      local x, y, z = xn[1], xn[2], xn[3]
      local r = math.sqrt(x^2 + y^2)

      local tau = 2.5e-10 

      local column = 0.5*(1 + math.tanh((r - (Lr_plasma))/rhos)) -- kill fields outside of top hat density column
      local boundary = 0.5*(1 + math.tanh((r - rs)/rhos))*0.5*(1 - math.tanh((z - 30*de)/(10*de))) -- kill fields away from flux rope centers near z = 0

      return (1/tau)*(1 - (1 - column)*(1 - boundary)) -- continuous verison of logical OR operator
    end,

    externalFieldInit = function (t, xn)
      -- B0 is purely axial (Z)
      return 0.0, 0.0, 0.0, 0.0, 0.0, B0
    end,
    
    bcx = { G0.FieldBc.bcPECWall, G0.FieldBc.bcPECWall },
    bcy = { G0.FieldBc.bcPECWall, G0.FieldBc.bcPECWall },
    bcz = { G0.FieldBc.bcPECWall, G0.FieldBc.bcCopy },

    evolve = true, 
    elcErrorSpeedFactor = 0.0,
    mgnErrorSpeedFactor = 0.0
  }
}

pkpmApp:run()
