-- I0: -2000 A
-- Resolution: 128 x 128 x 140
-- Domain: x,y = [-0.4, 0.4] m; z = [0, 10] m
-- Two-rope simulation

local PKPM = G0.PKPM
local Constants = require "Lib.Constants"

-- physical parameters
pi = math.pi
local cFac = 1000 -- Permittivity scaling factor.
-- Universal constant parameters.
local eps0, eV = Constants.EPSILON0*cFac, Constants.ELEMENTARY_CHARGE
local mu0      = Constants.MU0
local m_p = Constants.PROTON_MASS


Te_Ti = 5.0 -- ratio of electron to ion temperature
temp_elc = 5.*eV
temp_ion = temp_elc / Te_Ti
n0 = 2.0e18 --18 -- initial number density

mass_ion = 4*m_p -- ion mass, Helium
charge_ion = eV -- ion charge, singly ionized He
mass_elc = mass_ion / 400. -- m_e -- electron mass
charge_elc = -eV -- electron charge

B0 =  330. / 10000. -- convert to Tesla

-- Alfven speeds
vAElc = B0/math.sqrt(mu0*n0*mass_elc)

-- Thermal speeds
vtElc = math.sqrt(2.*temp_elc / mass_elc) --electron thermal velocity
vtIon = math.sqrt(2.*temp_ion / mass_ion) --ion thermal velocity

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
-- current formulation from NRL Formulary pgs 34 & 37
rmin_elc = 1/(4*math.pi*Constants.EPSILON0) * charge_elc^2 / temp_elc
rmin_ion = 1/(4*math.pi*Constants.EPSILON0) * charge_ion^2 / temp_ion
rmax = debye
lambda_elc = math.log(rmax/rmin_elc) -- Coulomb logarithm
lambda_ion = math.log(rmax/rmin_ion) -- Coulomb logarithm

nu_coeff = 0.1 -- artificially scale collisions

nu_elc = nu_coeff*(4*(2*math.pi)^0.5*n0*charge_elc^4*lambda_elc)/(3*(4*math.pi*Constants.EPSILON0)^2*mass_elc^0.5*temp_elc^1.5) -- electron collision frequency
nu_ion = nu_coeff*(4*(2*math.pi)^0.5*n0*charge_ion^4*lambda_ion)/(3*(4*math.pi*Constants.EPSILON0)^2*mass_ion^0.5*temp_ion^1.5) -- ion collision frequency

-- Hyper diffusion coefficients
D_coeff = 2.0e-3
D_scale = (wpe/(2*math.pi))*de^4

-- Simulation parameters.
Nx = 128 -- Cell count (configuration space: x- and y-directions).
Ny = Nx
Nz = 140
Nvx = 16 -- Cell count (velocity space: vx-direction).

Ncx = 16 -- Decomposition in x
Ncy = Ncx -- Decomposition in y
Ncz = 14 -- Decomposition in z

Lr = 0.4 -- Domain size (configuration space: x- and y-direction).
Lr_plasma = 0.3 -- radius of plasma column
LzSt = 0.
LzEnd = 10.0
Lz = LzEnd - LzSt

vx_max_ion = 6.0 * vtIon -- Domain boundary (ion velocity space: vx-direction).
vx_max_elc = 6.0 * vtElc -- Domain boundary (electron velocity space: vx-direction).
poly_order = 1 -- Polynomial order.
basis_type = "serendipity" -- Basis function set.
time_stepper = "rk3" -- Time integrator.
cfl_frac = 1.0 -- CFL coefficient.

-- rope params
nr = (3.0e18 - n0) -- Flux rope density. Want delta n here
Tr = 17.33*eV -- Flux rope temperature increment.
I0 = -2000 -- Total current, split equally between the ropes.
I1 = 0.5*I0 -- Current in rope 1.
I2 = 0.5*I0 -- Current in rope 2.
rs = 0.04445 -- profile scale

sep_factor = 2.661 -- current channel separation in de (0.01m in real units)
sep = sep_factor*de -- current channel seperation

lo_I0 = -750 -- Reference current for time normalization.
lo_I1 = lo_I0*0.5 -- Reference current in one rope.
lo_Bperp = math.abs(0.64*mu0*lo_I1/(2.*math.pi*rs)) -- peak Bperp field in the flux ropes
lo_omegaCi_perp = charge_ion*lo_Bperp/mass_ion
hi_Bperp = math.abs(0.64*mu0*I1/(2.*math.pi*rs)) -- peak Bperp field in the flux ropes
hi_omegaCi_perp = charge_ion*hi_Bperp/mass_ion

frame_ratio = lo_omegaCi_perp/hi_omegaCi_perp

t_end = 50*frame_ratio/omegaCi -- Final simulation time. Normalize to perpendicular field 
num_frames = 100 -- Number of output frames.
dt_failure_tol = 1.0e-4 -- Minimum allowable fraction of initial time-step.
num_failures_max = 20 -- Maximum allowable number of consecutive small time-steps.

x01 = sep/2. + 2.0*rs
y01 = 0.
x02 = -sep/2. - 2.0*rs
y02 = 0.

-- Write parameters to file for post-processing:
dimsX = 3
dimsV = 1 -- 1 for pkpm
polyOrder = poly_order
basis = 'ms'
model = 'pkpm'
suffix = '.gkyl'

fname = os.getenv("SCRATCH") .. "/" .. os.getenv("LUA_INPUT_FILE"):match("([^/\\]+)%.lua$") .. "/" .. os.getenv("LUA_INPUT_FILE"):match("([^/\\]+)%.lua$") .. "-params.txt"

file = io.open(fname, 'w')
file:write("eps0,\tmu0,\tB0,\tdimsX,\tdimsV(=0 for fluid),\tpolyOrder,\tbasis,\tmodel(vm,5m,10m),\tfilesuffix (.bp,gkyl)\n")
file:write(string.format("%g,\t%g,\t%g,\t%d,\t%d,\t%d,\t%s,\t%s,\t%s\n", eps0, mu0, B0, dimsX, dimsV, polyOrder, basis, model, suffix))
file:write("species,\tmass,\tcharge,\ttemperature,\tdensity\n")

file:write(string.format("%s,\t%g,\t%g,\t%g,\t%g\n", 'elc', mass_elc, charge_elc, temp_elc, n0))
file:write(string.format("%s,\t%g,\t%g,\t%g,\t%g\n", 'ion', mass_ion, charge_ion, temp_ion, n0))


file:close()

pkpmApp = PKPM.App.new {

  tEnd = t_end,
  nFrame = num_frames,
  dtFailureTol = dt_failure_tol,
  numFailuresMax = num_failures_max,
  lower = {-Lr, -Lr, LzSt },
  upper = {Lr, Lr, LzEnd },
  cells = { Nx, Ny, Nz },
  cflFrac = cfl_frac,

  useExplicitSource = true, --use explicit EM-fluid coupling

  basis = basis_type,
  polyOrder = poly_order,
  timeStepper = time_stepper,

  -- Decomposition for configuration space.
  decompCuts = { Ncx, Ncy, Ncz }, -- Cuts in each coodinate direction (x-direction only).

  -- Boundary conditions for configuration space.
  periodicDirs = { }, -- Periodic directions (all directions, for testing only).

  -- Electrons.
  elc = PKPM.Species.new {
    modelID = G0.Model.Default,
    charge = charge_elc, mass = mass_elc,
    
    -- Velocity space grid.
    lower = { -vx_max_elc },
    upper = { vx_max_elc  },
    cells = { Nvx },

    -- Initial conditions (distribution function).
    initDist = function (t, xn)
      local x, y, z, vx = xn[1], xn[2], xn[3], xn[4]
      local vt = math.sqrt(temp_elc / mass_elc)
      
      local x1 = x-x01
      local y1 = y-y01
      local x2 = x-x02
      local y2 = y-y02
      local r = math.sqrt( x^2. + y^2. )
      local r1 = math.sqrt( x1^2 + y1^2 )
      local r2 = math.sqrt( x2^2. + y2^2. )
      local phi1 = math.atan2(y1, x1)
      local phi2 = math.atan2(y2, x2)

      local fac1 = math.exp(-(r1^2 / rs^2)) -- Flux rope 1 density profile
      local fac2 = math.exp(-(r2^2 / rs^2)) -- Flux rope 2 density profile
      local vtR = math.sqrt(Tr / mass_elc)

      -- Drift velocity such that total current through xy plane is I: in the LAB frame
      local vdrift_z1_lab = (I1) / (charge_elc*pi*rs^2*nr) 
      local vdrift_z2_lab = (I2) / (charge_elc*pi*rs^2*nr)
      -- Transform these velocities into the fluid bulk flow frame
      local ue = (vdrift_z1_lab*nr*fac1 + vdrift_z2_lab*nr*fac2) / (n0 + nr*fac1 + nr*fac2) -- bulk fluid velocity
      local vdrift_z0 = -ue
      local vdrift_z1 = vdrift_z1_lab - ue
      local vdrift_z2 = vdrift_z2_lab - ue

      -- Compute b_hat vector and find parallel velocity
      local Bphi1 = ((1 - math.exp(-(r1^2)/rs^2))*I1*mu0)/(2.*math.pi*r1)
      local Bphi2 = ((1 - math.exp(-(r2^2)/rs^2))*I2*mu0)/(2.*math.pi*r2)
      local Bx = -math.sin(phi1)*Bphi1 - math.sin(phi2)*Bphi2
      local By = math.cos(phi1)*Bphi1 + math.cos(phi2)*Bphi2
      local Bz = B0 + 0
      local Bmag = math.sqrt(Bx^2 + By^2 + Bz^2)

      local vdrift_par0 = vdrift_z0 * (Bz / Bmag) -- project drift onto b_hat
      local vdrift_par1 = vdrift_z1 * (Bz / Bmag)
      local vdrift_par2 = vdrift_z2 * (Bz / Bmag)

      local n0B = n0*0.5*(1 - math.tanh((r - (Lr_plasma))/rhos)) + 0.1*n0

      local f0B = (n0B / math.sqrt(2.0*pi*vt^2)) * (math.exp(-((vx - vdrift_par0)^2) / (2.0*vt^2))) -- Background F0
      local f01 = (nr*fac1 / math.sqrt(2.0*pi*vtR^2)) * (math.exp(-((vx - vdrift_par1)^2) / (2.0*vtR^2))) -- Rope1 F0
      local f02 = (nr*fac2 / math.sqrt(2.0*pi*vtR^2)) * (math.exp(-((vx - vdrift_par2)^2) / (2.0*vtR^2))) -- Rope2 F0

      local f0 = f0B + f01 + f02
      local g = (temp_elc*f0 + Tr*f01 + Tr*f02) / mass_elc

      return f0, g
    end,

    -- Initial conditions (fluid).
    initFluid = function (t, xn)
      local x, y, z, vx = xn[1], xn[2], xn[3], xn[4]
      local x1 = x-x01
      local y1 = y-y01
      local x2 = x-x02
      local y2 = y-y02
      local r1 = math.sqrt( x1^2 + y1^2 )
      local r2 = math.sqrt( x2^2. + y2^2. )
      local fac1 = math.exp(-(r1^2 / rs^2)) -- Flux rope 1 density profile
      local fac2 = math.exp(-(r2^2 / rs^2)) -- Flux rope 2 density profile

      local vdrift_z1 = (I1) / (charge_elc*pi*rs^2*nr) -- Drift velocity such that total current through xy plane is I1
      local vdrift_z2 = (I2) / (charge_elc*pi*rs^2*nr)

      local ezmom1 = vdrift_z1 * nr * fac1 * mass_elc -- Electron momentum density.
      local ezmom2 = vdrift_z2 * nr * fac2 * mass_elc

      local mom_x = 0.0 -- Total momentum density (x-direction).
      local mom_y = 0.0 -- Total momentum density (y-direction).
      local mom_z = ezmom1 + ezmom2 -- Total momentum density (z-direction).

      return mom_x, mom_y, mom_z
    end,

    collisions = {
      collisionID = G0.Collisions.LBO,

      selfNu = function (t, xn)
        return nu_elc -- Collision frequency.
      end
    },
    diffusion = {
      D = D_scale * D_coeff,
      order = 4
    },
    bcx = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcy = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcz = { G0.SpeciesBc.bcLineTied, G0.SpeciesBc.bcCopy },
    evolve = true -- Evolve species?
  },

  -- Ions
  ion = PKPM.Species.new {
    modelID = G0.Model.Default,
    charge = charge_ion, mass = mass_ion,
    
    -- Velocity space grid.
    lower = { -vx_max_ion },
    upper = { vx_max_ion  },
    cells = { Nvx },

    -- Initial conditions (distribution function).
    initDist = function (t, xn)
      local x, y, z, vx = xn[1], xn[2], xn[3], xn[4]
      local vt = math.sqrt(temp_ion / mass_ion)
      
      local x1 = x-x01
      local y1 = y-y01
      local x2 = x-x02
      local y2 = y-y02
      local r = math.sqrt( x^2. + y^2. )
      local r1 = math.sqrt( x1^2 + y1^2 )
      local r2 = math.sqrt( x2^2. + y2^2. )
      local fac1 = math.exp(-(r1^2 / rs^2)) -- Flux rope 1 density profile
      local fac2 = math.exp(-(r2^2 / rs^2)) -- Flux rope 2 density profile
      local Tr_ion = Tr/Te_Ti
      local vtR = math.sqrt(Tr_ion / mass_ion)

      local n0B = n0*0.5*(1 - math.tanh((r - (Lr_plasma))/rhos)) + 0.1*n0
      
      local f0B = (n0B / math.sqrt(2.0*pi*vt^2)) * (math.exp(-(vx^2) / (2.0*vt^2))) -- Background F0
      local f01 = (nr*fac1 / math.sqrt(2.0 * pi * vtR^2)) * (math.exp(-(vx^2) / (2.0 * vtR^2))) -- Rope1 F0
      local f02 = (nr*fac2 / math.sqrt(2.0 * pi * vtR^2)) * (math.exp(-(vx^2) / (2.0 * vtR^2))) -- Rope2 F0

      local f0 = f0B + f01 + f02
      local g = (temp_ion*f0 + Tr_ion*f01 + Tr_ion*f02) / mass_ion
      
      return f0, g
    end,

    -- Initial conditions (fluid).
    initFluid = function (t, xn)
      local mom_x = 0.0 -- Total momentum density (x-direction).
      local mom_y = 0.0 -- Total momentum density (y-direction).
      local mom_z = 0.0 -- Total momentum density (z-direction).

      return mom_x, mom_y, mom_z
    end,

    collisions = {
      collisionID = G0.Collisions.LBO,

      selfNu = function (t, xn)
        return nu_ion -- Collision frequency.
      end
    },
    diffusion = {
      D = D_scale * D_coeff,
      order = 4
    },
    bcx = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcy = { G0.SpeciesBc.bcCopy, G0.SpeciesBc.bcCopy },
    bcz = { G0.SpeciesBc.bcLineTied, G0.SpeciesBc.bcCopy },
    evolve = true -- Evolve species?
  },

  -- Field.
  field = PKPM.Field.new {
    epsilon0 = eps0, mu0 = mu0,

    -- Initial conditions function.
    init = function (t, xn)
      local x, y, z = xn[1], xn[2], xn[3]

        local x1 = x-x01
        local y1 = y-y01
        local x2 = x-x02
        local y2 = y-y02
        local r = math.sqrt( x^2. + y^2. )
        local r1 = math.sqrt( x1^2 + y1^2 )
        local r2 = math.sqrt( x2^2. + y2^2. )

        local phi1 = math.atan2(y1, x1)
        local phi2 = math.atan2(y2, x2)
        
        local Bphi1 = ((1 - math.exp(-(r1^2)/rs^2))*I1*mu0)/(2.*math.pi*r1)
        local Bphi2 = ((1 - math.exp(-(r2^2)/rs^2))*I2*mu0)/(2.*math.pi*r2)
        local Bx = -math.sin(phi1)*Bphi1 - math.sin(phi2)*Bphi2
        local By = math.cos(phi1)*Bphi1 + math.cos(phi2)*Bphi2

        local dBz = (mu0*n0)/(2*B0)*(temp_elc + temp_ion)*(1 + math.tanh((r - (Lr_plasma))/rhos))
        local Bz = dBz

        local nre1 = nr*math.exp(-r1^2/rs^2)
        local nre2 = nr*math.exp(-r2^2/rs^2)
        local ne = n0 + nre1 + nre2

        local vdrift_z1_lab = (I1) / (charge_elc*pi*rs^2*nr)
        local vdrift_z2_lab = (I2) / (charge_elc*pi*rs^2*nr)
        local j_z1 = vdrift_z1_lab*nre1*charge_elc
        local j_z2 = vdrift_z2_lab*nre2*charge_elc
        
        local E_jxb_x = (1/(charge_elc*ne))*(j_z1 + j_z2)*By
        local E_jxb_y = -(1/(charge_elc*ne))*(j_z1 + j_z2)*Bx

        local E_gradP1_r = -(8*r1*(nr*(math.exp((r1^2)/rs^2)*n0 +
                          nr)^2*charge_elc^2*16*rs^4*Tr -
                          (64*B0^2*I1^2*mass_elc*nr*r1^2)/(4*B0^2*math.pi^2*r1^2 +
                          (-1 + math.exp(-(r1^2)/rs^2))^2*I1^2*mu0^2) +
                          (128*B0^2*math.exp((r1^2)/rs^2)*I1^2*mass_elc*(n0 +
                          nr/math.exp((r1^2)/rs^2))*r1^2)/(4*B0^2*math.pi^2*
                          r1^2 + (-1 + math.exp(-(r1^2)/rs^2))^2*I1^2*mu0^2) -
                          (64*B0^2*math.exp((2*r1^2)/rs^2)*(-1 +
                          math.exp((r1^2)/rs^2))*I1^4*mass_elc*(math.exp((r1^2)/
                          rs^2)*n0 + nr)*((-1 + math.exp((r1^2)/rs^2))*rs^2 -
                          2*r1^2)*mu0^2)/(4*B0^2*math.exp((2*r1^2)/rs^2)*math.
                          pi^2*r1^2 + (-1 +
                          math.exp((r1^2)/rs^2))^2*I1^2*mu0^2)^2))/(math.exp((3*r1^2)/rs^2)*(n0 +
                          nr/math.exp((r1^2)/rs^2))^3*charge_elc^3*64*rs^6)

        local E_gradP2_r = -(8*r2*(nr*(math.exp((r2^2)/rs^2)*n0 +
                          nr)^2*charge_elc^2*16*rs^4*Tr -
                          (64*B0^2*I2^2*mass_elc*nr*r2^2)/(4*B0^2*math.pi^2*r2^2 +
                          (-1 + math.exp(-(r2^2)/rs^2))^2*I2^2*mu0^2) +
                          (128*B0^2*math.exp((r2^2)/rs^2)*I2^2*mass_elc*(n0 +
                          nr/math.exp((r2^2)/rs^2))*r2^2)/(4*B0^2*math.pi^2*
                          r2^2 + (-1 + math.exp(-(r2^2)/rs^2))^2*I2^2*mu0^2) -
                          (64*B0^2*math.exp((2*r2^2)/rs^2)*(-1 +
                          math.exp((r2^2)/rs^2))*I2^4*mass_elc*(math.exp((r2^2)/
                          rs^2)*n0 + nr)*((-1 + math.exp((r2^2)/rs^2))*rs^2 -
                          2*r2^2)*mu0^2)/(4*B0^2*math.exp((2*r2^2)/rs^2)*math.
                          pi^2*r2^2 + (-1 +
                          math.exp((r2^2)/rs^2))^2*I2^2*mu0^2)^2))/(math.exp((3*r2^2)/rs^2)*(n0 +
                          nr/math.exp((r2^2)/rs^2))^3*charge_elc^3*64*rs^6)

        local Ex = E_gradP1_r*math.cos(phi1) + E_gradP2_r*math.cos(phi2) + E_jxb_x -- Electric field (x-direction).
        local Ey = E_gradP1_r*math.sin(phi1) + E_gradP2_r*math.sin(phi2) + E_jxb_y -- Electric field (y-direction).
        if Ex == math.huge or Ex == -math.huge then Ex = 0.0 end
        if Ey == math.huge or Ey == -math.huge then Ey = 0.0 end
        if Ex ~= Ex then Ex = 0.0 end
        if Ey ~= Ey then Ey = 0.0 end

        local Ez = 0.0 -- Electric field (z-direction).

      return Ex, Ey, Ez, Bx, By, Bz, 0.0, 0.0
    end,

    sigma = function (t, xn)
      local x, y, z = xn[1], xn[2], xn[3]
      local x1 = x-x01
      local y1 = y-y01
      local x2 = x-x02
      local y2 = y-y02
      local r = math.sqrt( x^2. + y^2. )
      local r1 = math.sqrt( x1^2 + y1^2 )
      local r2 = math.sqrt( x2^2. + y2^2. )

      -- hyperbolic tanh profile for damping EM waves
      local tau = 2.5e-10 -- decay timescale of fields in boundary region (must be greater than dt)
      -- Define sigma shaping functions noramlized to 1 (1 -> sigma is HIGH, 0 -> sigma is LOW)
      local column = 0.5*(1 + math.tanh((r - (Lr_plasma))/rhos)) -- kill fields outside of top hat density column
      local boundary = 0.25*(1 + math.tanh((r1 - rs)/rhos))*(1 + math.tanh((r2 - rs)/rhos))*0.5*(1 - math.tanh((z - 30*de)/(10*de))) -- kill fields away from flux rope centers near z = 0

      return (1/tau)*(1 - (1 - column)*(1 - boundary)) -- continuous verison of logical OR operator
    end,

    externalFieldInit = function (t, xn)
      local x, y, z = xn[1], xn[2], xn[3]
      local r = math.sqrt( x^2. + y^2. )
      
      local Ex = 0.0 -- External electric field (x-direction).
      local Ey = 0.0 -- External electric field (y-direction).
      local Ez = 0.0 -- External electric field (z-direction).

      local Bx = 0.0 -- External magnetic field (x-direction).
      local By = 0.0 -- External magnetic field (y-direction).

      local Bz = B0 -- External magnetic field (z-direction).
         
      return Ex, Ey, Ez, Bx, By, Bz

    end,
    bcx = { G0.FieldBc.bcPECWall, G0.FieldBc.bcPECWall },
    bcy = { G0.FieldBc.bcPECWall, G0.FieldBc.bcPECWall },
    bcz = { G0.FieldBc.bcPECWall, G0.FieldBc.bcCopy },

    evolve = true, -- Evolve field?
    elcErrorSpeedFactor = 0.0,
    mgnErrorSpeedFactor = 0.0
  }
}

pkpmApp:run()
