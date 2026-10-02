#include <math.h>
#include <stdio.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_const.h>
#include <gkyl_eqn_type.h>
#include <gkyl_fem_parproj.h>
#include <gkyl_fem_poisson_bctype.h>
#include <gkyl_gyrokinetic.h>
#include <gkyl_null_comm.h>

#ifdef GKYL_HAVE_MPI
#include <mpi.h>
#include <gkyl_mpi_comm.h>
#ifdef GKYL_HAVE_NCCL
#include <gkyl_nccl_comm.h>
#endif
#endif

#include <gkyl_math.h>
#include <rt_arg_parse.h>

// Define the context of the simulation. This stores global parameters.
struct gk_app_ctx {
    char sim_name[128]; // Simulation name.
    int cdim, vdim;
    // Geometry and magnetic field parameters
    double a_shift, Z_axis, R_axis, R0, a_mid, x_inner, r0, B0, kappa, delta, q0, qaxis, qlcfs, Bref, x_LCFS;
    // Plasma parameters
    double me, qe, mi, qi, n0, Te0, Ti0;
    // Collision parameters
    double nuFrac;
    // Initial condition parameters
    double dens_init_mean[3], dens_init_sigma[3];
    double num_particle_init, energy_init;
    // Source parameters
    double num_sources;
    bool adapt_energy_srcCORE, adapt_particle_srcCORE; 
    double center_srcCORE[3], sigma_srcCORE[3];
    double energy_srcCORE, particle_srcCORE;
    double floor_srcCORE;
    bool adapt_energy_srcRECY, adapt_particle_srcRECY;
    double center_srcRECY[3], sigma_srcRECY[3];
    double energy_srcRECY, particle_srcRECY;
    double floor_srcRECY;
    // Grid parameters
    double Lx, Ly, Lz;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    int num_cell_x, num_cell_y, num_cell_z, num_cell_vpar, num_cell_mu;
    int cells[GKYL_MAX_DIM], poly_order;
    double vpar_max_elc, mu_max_elc, vpar_max_ion, mu_max_ion;
    // Simulation control parameters
    double final_time, write_phase_freq;
    int num_frames, int_diag_calc_num, num_failures_max;
    double dt_failure_tol;
    double max_run_time; // Maximum run time in seconds, 0 means no limit.
};

// Function prototypes (defined at the end of the file)
static double r_x(double x, double a_mid, double x_inner);
static double qprofile(double r, double a_mid, double qaxis, double qlcfs);
static void density_init(double t, const double *xn, double *fout, void *ctx);
static void temp_elc(double t, const double *xn, double *fout, void *ctx);
static void temp_ion(double t, const double *xn, double *fout, void *ctx);
static void zero_func(double t, const double *xn, double *fout, void *ctx);
static void mapc2p(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void mapc2p_vel_elc(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void mapc2p_vel_ion(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx);
static void bfield_func(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx);
static void bc_shift_func_lo(double t, const double *xn, double *fout, void *ctx);
static void bc_shift_func_up(double t, const double *xn, double *fout, void *ctx);
static void write_data(struct gkyl_tm_trigger *iot_conf, struct gkyl_tm_trigger *iot_phase,
                       gkyl_gyrokinetic_app *app, double t_curr, bool force_write);
static void calc_integrated_diagnostics(struct gkyl_tm_trigger *iot, gkyl_gyrokinetic_app *app, double t_curr, bool force_write);
static void write_ctx_to_json(struct gk_app_ctx *ctx, const char *filename);

struct gk_app_ctx create_ctx(void)
{
  int cdim = 3, vdim = 2; // Dimensionality.
  // Universal constant parameters.
  double eps0 = GKYL_EPSILON0, eV = GKYL_ELEMENTARY_CHARGE;
  double mp = GKYL_PROTON_MASS, me = GKYL_ELECTRON_MASS;
  double qi = eV; // ion charge
  double qe = -eV; // electron charge

  // Geometry and magnetic field.
  double a_shift   = 0.2;               // Parameter in Shafranov shift.
  double Z_axis    = 0.1414361745;       // Magnetic axis height [m].
  double R_axis    = 0.8727315068;       // Magnetic axis major radius [m].
  double B_axis    = 1.4;                // Magnetic field at the magnetic axis [T].
  double R_LCFSmid = 1.0968432365089495; // Major radius of the LCFS at the outboard midplane [m].
  double x_inner   = 0.04;               // Radial extent inside LCFS    
  double x_outer   = 0.08;               // Radial extent outside LCFS
  double Rmid_min  = R_LCFSmid - x_inner;      // Minimum midplane major radius of simulation box [m].
  double Rmid_max  = R_LCFSmid + x_outer;      // Maximum midplane major radius of simulation box [m].
  double R0        = 0.5*(Rmid_min+Rmid_max);  // Major radius of the simulation box [m].
  double a_mid     = R_LCFSmid-R_axis;   // Minor radius at outboard midplane [m].
  // Redefine a_mid with Shafranov shift, to ensure LCFS radial location.
  a_mid = R_axis/a_shift - sqrt(R_axis*(R_axis - 2*a_shift*R_LCFSmid + 2*a_shift*R_axis))/a_shift;
  double r0        = R0-R_axis;           // Minor radius of the simulation box [m].
  double B0        = B_axis*(R_axis/R0);  // Magnetic field magnitude in the simulation box [T].
  double kappa     = 1.45;                // Elongation (=1 for no elongation).
  double delta     = 0.35;                // Triangularity (=0 for no triangularity).
  double qaxis     = 1.2;                 // Safety factor at r=0.
  double qlcfs     = 2.6;                // Safety factor at the LCFS.
  // Plasma parameters. Chosen based on the value of a cubic sline
  // between the last TS data inside the LCFS and the probe data in
  // in the far SOL, near R=0.475 m.
  double AMU = 2.01410177811;
  double mi  = mp*AMU; // Deuterium ions.
  double Te0 = 100*eV;
  double Ti0 = 100*eV;
  double n0  = 2.0e19; // [1/m^3]
  double Bref = 1.129; // Reference magnetic field [T].
  double nuFrac = 0.5; // Collision factor.

  double vte = sqrt(Te0/me), vti = sqrt(Ti0/mi); // Thermal speeds.
  double c_s = sqrt(Te0/mi);
  double omega_ci = fabs(qi*B0/mi);
  double rho_s = c_s/omega_ci;
  double q0 = qprofile(r0, a_mid, qaxis, qlcfs);

  // Configuration domain parameters 
  double Lx        = Rmid_max-Rmid_min;   // Domain size along x.
  double x_min     = 0.;
  double x_max     = Lx;
  double x_LCFS    = R_LCFSmid - Rmid_min; // Radial location of the last closed flux surface.
  double Ly        = 150*rho_s;           // Domain size along y.
  // Adjust the domain size along y to have integer toroidal mode number.
  // We need: 2*pi*Cy/Ly = integer (Cy = r0/q0)
  Ly = 2.*M_PI*r0/q0/round(2.*M_PI*r0/q0/Ly); 
  double y_min     = -Ly/2.;
  double y_max     =  Ly/2.;  
  double Lz        = 2.*M_PI-1e-10;       // Domain size along magnetic field.
  double z_min     = -Lz/2.;
  double z_max     =  Lz/2.;
  double vol_frac = 1.0/(2.*M_PI*r0/q0/Ly);

  // Initial conditions
  double num_particle_init = 1e19; // Initial number of particles.
  double energy_init = 3.0/2.0 * num_particle_init * 300 * eV; // Initial total kinetic energy.
  double dens_init_mean[3] = {0.0, 0.0, 0.0};
  double dens_init_sigma[3] = {x_inner, 0.0, 0.0};

  // Source parameters
  double num_sources = 2; // We do not activate the recycling source here.
  // Core source parameters
  bool adapt_energy_srcCORE = true;
  bool adapt_particle_srcCORE = true;
  double energy_srcCORE = 0.25e6 * vol_frac; // [W]
  double particle_srcCORE = 0.0; // [1/s]
  double center_srcCORE[3] = {x_min, 0.0, -Lz/4}; // This is the position of the ion source,
  double sigma_srcCORE[3] = {0.03*Lx, 0.0, Lz/6}; //  the electron source will be at +Lz/2.
  double floor_srcCORE = 1e-10;
  // Recycling source parameters (we do not use it here)
  bool adapt_energy_srcRECY = false;
  bool adapt_particle_srcRECY = true;
  double energy_srcRECY = 0.0; // [W]
  double particle_srcRECY = 0.0; // [1/s]
  double center_srcRECY[3] = {0.5*x_LCFS, 0.0, M_PI};
  double sigma_srcRECY[3] = {0.25*x_LCFS, 0.0, 0.05*Lz};
  double floor_srcRECY = 1e-10;
  // Grid parameters
  int num_cell_x = 36;
  int num_cell_y = 24;
  int num_cell_z = 16;
  int num_cell_vpar = 12;
  int num_cell_mu = 8;
  int poly_order = 1;
  // Velocity box dimensions
  double vpar_max_elc = 6.*vte;
  double mu_max_elc   = 1.5*me*pow(4*vte,2)/(2*B0);
  double vpar_max_ion = 6.*vti;
  double mu_max_ion   = 1.5*mi*pow(4*vti,2)/(2*B0);
  double final_time = 3.0e-3;
  int num_frames = 1500;
  double write_phase_freq = 0.2;
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-3; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

  struct gk_app_ctx ctx = {
    .sim_name = "tcv_miller_scan",
    .cdim = cdim,
    .vdim = vdim,
    .a_shift = a_shift,
    .R_axis = R_axis,
    .R0     = R0    ,
    .a_mid  = a_mid ,
    .x_inner = x_inner,
    .r0     = r0    ,
    .B0     = B0    ,
    .kappa  = kappa ,
    .delta  = delta ,
    .q0     = q0    ,
    .qaxis  = qaxis ,
    .qlcfs  = qlcfs ,
    .Lx     = Lx    ,
    .Ly     = Ly    ,
    .Lz     = Lz    ,
    .x_min = x_min,  .x_max = x_max,
    .y_min = y_min,  .y_max = y_max,
    .z_min = z_min,  .z_max = z_max,
    .Bref = Bref,
    .x_LCFS = x_LCFS,
    .me = me,  .qe = qe,
    .mi = mi,  .qi = qi,
    .n0 = n0,  .Te0 = Te0,  .Ti0 = Ti0,
    .nuFrac = nuFrac,
    .num_particle_init = num_particle_init,
    .energy_init = energy_init,
    .dens_init_mean = {dens_init_mean[0], dens_init_mean[1], dens_init_mean[2]},
    .dens_init_sigma = {dens_init_sigma[0], dens_init_sigma[1], dens_init_sigma[2]},
    .num_sources = num_sources,
    .adapt_energy_srcCORE = adapt_energy_srcCORE,
    .adapt_particle_srcCORE = adapt_particle_srcCORE,
    .center_srcCORE = {center_srcCORE[0], center_srcCORE[1], center_srcCORE[2]},
    .sigma_srcCORE = {sigma_srcCORE[0], sigma_srcCORE[1], sigma_srcCORE[2]},
    .energy_srcCORE = energy_srcCORE,  .particle_srcCORE = particle_srcCORE,
    .floor_srcCORE = floor_srcCORE,
    .adapt_energy_srcRECY = adapt_energy_srcRECY,
    .adapt_particle_srcRECY = adapt_particle_srcRECY,
    .center_srcRECY = {center_srcRECY[0], center_srcRECY[1], center_srcRECY[2]},
    .sigma_srcRECY = {sigma_srcRECY[0], sigma_srcRECY[1], sigma_srcRECY[2]},
    .energy_srcRECY = energy_srcRECY,  .particle_srcRECY = particle_srcRECY,
    .floor_srcRECY = floor_srcRECY,
    .num_cell_x     = num_cell_x,
    .num_cell_y     = num_cell_y,
    .num_cell_z     = num_cell_z,
    .num_cell_vpar  = num_cell_vpar,
    .num_cell_mu    = num_cell_mu,
    .cells = {num_cell_x, num_cell_y, num_cell_z, num_cell_vpar, num_cell_mu},
    .poly_order   = poly_order,
    .vpar_max_elc = vpar_max_elc,  .mu_max_elc = mu_max_elc,
    .vpar_max_ion = vpar_max_ion,  .mu_max_ion = mu_max_ion,
    .write_phase_freq = write_phase_freq,
    .final_time = final_time,  .num_frames = num_frames,
    .int_diag_calc_num = int_diag_calc_num,
    .dt_failure_tol = dt_failure_tol,
    .num_failures_max = num_failures_max,
  };
  return ctx;
}

int 
main(int argc, char **argv)
{
  struct timespec timer_global = gkyl_wall_clock();
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    MPI_Init(&argc, &argv);
#endif
  
  if (app_args.trace_mem) {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct gk_app_ctx ctx = create_ctx(); // context for init functions

  // Extract variables from command line arguments.
  sscanf(app_args.opt_args, "kappa=%lf,delta=%lf,energy_srcCORE=%lf,max_run_time=%lf,sim_name=%s", 
    &ctx.kappa, &ctx.delta, &ctx.energy_srcCORE, &ctx.max_run_time, &ctx.sim_name);
  
  // Check if --write-ctx flag was provided
  bool write_ctx_flag = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--write-ctx") == 0) {
      write_ctx_flag = true;
      break;
    }
  }
  
  if (write_ctx_flag) {
    char json_filename[256];
    snprintf(json_filename, sizeof(json_filename), "%s_ctx.json", ctx.sim_name);
    write_ctx_to_json(&ctx, json_filename);
    printf("Context written to %s\n", json_filename);
    
#ifdef GKYL_HAVE_MPI
    if (app_args.use_mpi)
      MPI_Finalize();
#endif
    return 0;
  }
  
  printf("Command line arguments:\n");
  printf("  sim_name = %s\n", ctx.sim_name);
  printf("  kappa    = %3.2f\n", ctx.kappa);
  printf("  delta    = %3.2f\n", ctx.delta);
  printf("  energy_srcCORE = %3.2f\n", ctx.energy_srcCORE);
  printf("  max_run_time = %f\n", ctx.max_run_time);

  int cells_x[ctx.cdim], cells_v[ctx.vdim];
  for (int d=0; d<ctx.cdim; d++)
    cells_x[d] = APP_ARGS_CHOOSE(app_args.xcells[d], ctx.cells[d]);
  for (int d=0; d<ctx.vdim; d++)
    cells_v[d] = APP_ARGS_CHOOSE(app_args.vcells[d], ctx.cells[ctx.cdim+d]);
  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_gyrokinetic_comms_new(app_args.use_mpi, app_args.use_gpu, stderr);
  int my_rank = 0;
#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    gkyl_comm_get_rank(comm, &my_rank);
#endif


  struct gkyl_gyrokinetic_projection proj_srcCORE_e = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcCORE[0], ctx.center_srcCORE[1], -ctx.center_srcCORE[2]},
    .gaussian_std_dev = {ctx.sigma_srcCORE[0], ctx.sigma_srcCORE[1], ctx.sigma_srcCORE[2]},
    .total_num_particles = ctx.particle_srcCORE,
    .total_kin_energy = ctx.energy_srcCORE,
    .temp_max = 50.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcCORE,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcCORE_e ={
    .adapt_to_species = "elc",
    .adapt_particle = ctx.adapt_particle_srcCORE,
    .adapt_energy = ctx.adapt_energy_srcCORE,
    .num_boundaries = 1,
    .dir = {0, 0, 2, 2},
    .edge = {GKYL_LOWER_EDGE, GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  struct gkyl_gyrokinetic_projection proj_srcRECY_e = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcRECY[0], ctx.center_srcRECY[1], ctx.center_srcRECY[2]},
    .gaussian_std_dev = {ctx.sigma_srcRECY[0], ctx.sigma_srcRECY[1], ctx.sigma_srcRECY[2]},
    .total_num_particles = ctx.particle_srcRECY,
    .total_kin_energy = ctx.energy_srcRECY,
    .temp_max = 50.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcRECY,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcRECY_e = {
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcRECY,
    .adapt_energy = ctx.adapt_energy_srcRECY,
    .num_boundaries = 3,
    .dir = {0, 2, 2},
    .edge = {GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  // electrons
  struct gkyl_gyrokinetic_species elc = {
    .name = "elc",
    .charge = ctx.qe, .mass = ctx.me,
    .vdim = ctx.vdim,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0},
    .cells = { cells_v[0], cells_v[1] },
    .polarization_density = ctx.n0,

/*
    .init_from_file = {
       .type = GKYL_IC_IMPORT_F,
       .file_name = "restart-elc.gkyl"
    },
//*/  
    .mapc2p = {
      .mapping = mapc2p_vel_elc,
      .ctx = &ctx,
    },

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temp = &ctx,
      .density = density_init,
      .upar = zero_func,
      .temp = temp_elc,
    },

    .collisionless = {
      .type = GKYL_GK_COLLISIONLESS_ES,
    },

    .collisions =  {
      .collision_id = GKYL_LBO_COLLISIONS,
      .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .temp_ref = ctx.Te0, // Temperature used to calculate coulomb logarithm
      .num_cross_collisions = 1,
      .collide_with = { "ion"},
      .nu_frac = ctx.nuFrac,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .num_adapt_sources = ctx.num_sources,
      .projection[0] = proj_srcCORE_e,
      .adapt[0] = adapt_srcCORE_e,
      .projection[1] = proj_srcRECY_e,
      .adapt[1] = adapt_srcRECY_e,
      .diagnostics = {
        .num_diag_moments = 1,
        .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      }
    },

    .bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 2, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_IWL, .aux_profile = bc_shift_func_lo, .aux_ctx = &ctx, },
      { .dir = 2, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_IWL, .aux_profile = bc_shift_func_up, .aux_ctx = &ctx, },
    },
    .num_diag_moments = 9,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_BIMAXWELLIAN, 
      GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, 
      GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .boundary_flux_diagnostics = {
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },
  };

  // ions

  struct gkyl_gyrokinetic_projection proj_srcCORE_i = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcCORE[0], ctx.center_srcCORE[1], ctx.center_srcCORE[2]},
    .gaussian_std_dev = {ctx.sigma_srcCORE[0], ctx.sigma_srcCORE[1], ctx.sigma_srcCORE[2]},
    .total_num_particles = ctx.particle_srcCORE,
    .total_kin_energy = ctx.energy_srcCORE,
    .temp_max = 50.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcCORE,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcCORE_i ={
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcCORE,
    .adapt_energy = ctx.adapt_energy_srcCORE,
    .num_boundaries = 1,
    .dir = {0, 0, 2, 2},
    .edge = {GKYL_LOWER_EDGE, GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  struct gkyl_gyrokinetic_projection proj_srcRECY_i = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcRECY[0], ctx.center_srcRECY[1], ctx.center_srcRECY[2]},
    .gaussian_std_dev = {ctx.sigma_srcRECY[0], ctx.sigma_srcRECY[1], ctx.sigma_srcRECY[2]},
    .total_num_particles = ctx.particle_srcRECY,
    .total_kin_energy = ctx.energy_srcRECY,
    .temp_max = 50.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcRECY,
  };

  struct gkyl_gyrokinetic_adapt_source adapt_srcRECY_i = {
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcRECY,
    .adapt_energy = ctx.adapt_energy_srcRECY,
    .num_boundaries = 3,
    .dir = {0, 2, 2},
    .edge = {GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  struct gkyl_gyrokinetic_species ion = {
    .name = "ion",
    .charge = ctx.qi, .mass = ctx.mi,
    .vdim = ctx.vdim,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0},
    .cells = { cells_v[0], cells_v[1] },
    .polarization_density = ctx.n0,

/*
    .init_from_file = {
       .type = GKYL_IC_IMPORT_F,
       .file_name = "restart-ion.gkyl"
    },
//*/
    .mapc2p = {
      .mapping = mapc2p_vel_ion,
      .ctx = &ctx,
    },

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temp = &ctx,
      .density = density_init,
      .upar = zero_func,
      .temp = temp_ion,
    },

    .collisionless = {
      .type = GKYL_GK_COLLISIONLESS_ES,
    },

    .collisions =  {
      .collision_id = GKYL_LBO_COLLISIONS,
      .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .temp_ref = ctx.Te0, // Temperature used to calculate coulomb logarithm
      .num_cross_collisions = 1,
      .collide_with = { "elc"},
      .nu_frac = ctx.nuFrac,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .num_adapt_sources = ctx.num_sources,
      .projection[0] = proj_srcCORE_i,
      .adapt[0] = adapt_srcCORE_i,
      .projection[1] = proj_srcRECY_i,
      .adapt[1] = adapt_srcRECY_i,
      .diagnostics = {
        .num_diag_moments = 1,
        .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      }
    },
    .bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 2, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_IWL, .aux_profile = bc_shift_func_lo, .aux_ctx = &ctx, },
      { .dir = 2, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_IWL, .aux_profile = bc_shift_func_up, .aux_ctx = &ctx, },
    },
    .num_diag_moments = 9,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_BIMAXWELLIAN, 
      GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, 
      GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .boundary_flux_diagnostics = {
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },
  };

  struct gkyl_poisson_bias_plane target_corner_bc = {
    .dir = 0, // Direction perpendicular to the plane.
    .loc = ctx.x_LCFS, // Location of the plane in the 'dir' dimension.
    .val = 0.0, // Biasing value.
  };
  
  struct gkyl_poisson_bias_plane_list bias_plane_list = {
    .num_bias_plane = 1,
    .bp = &target_corner_bc,
  };

  // field
  struct gkyl_gyrokinetic_field field = {
    .gkfield_id = GKYL_GK_FIELD_ES_IWL,
    .polarization_bmag = ctx.Bref,
    .poisson_bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0} },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0} },
    },
    .bias_plane_list = &bias_plane_list,
    .time_rate_diagnostics = true,
  };

  // Geometry
  struct gkyl_gyrokinetic_geometry geometry = {
    .geometry_id = GKYL_MAPC2P,
    .world = {0.},
    .mapc2p = mapc2p, // mapping of computational to physical space
    .c2p_ctx = &ctx,
    .bfield_func = bfield_func, // magnetic field magnitude
    .bfield_ctx = &ctx,
    .has_LCFS = true,
    .x_LCFS = ctx.x_LCFS,
  };

  // Parallelism
  struct gkyl_app_parallelism_inp parallelism = {
    .comm = comm,
    .cuts = {app_args.cuts[0], app_args.cuts[1], app_args.cuts[2]},
    .use_gpu = app_args.use_gpu,
  };

  // GK app
  struct gkyl_gk *gk = gkyl_malloc(sizeof *gk);
  memset(gk, 0, sizeof(*gk));
  
  gk->cfl_frac_omegaH = 1.0e9;
  gk->cfl_frac = 1.0;
  gk->cdim = ctx.cdim;
  gk->lower[0] = ctx.x_min;
  gk->lower[1] = ctx.y_min;
  gk->lower[2] = ctx.z_min;
  gk->upper[0] = ctx.x_max;
  gk->upper[1] = ctx.y_max;
  gk->upper[2] = ctx.z_max;
  gk->cells[0] = cells_x[0];
  gk->cells[1] = cells_x[1];
  gk->cells[2] = cells_x[2];
  gk->poly_order = ctx.poly_order;
  gk->basis_type = app_args.basis_type;
  gk->geometry = geometry;
  gk->num_periodic_dir = 1;
  gk->periodic_dirs[0] = 1;
  gk->num_species = 2;
  gk->species[0] = elc;
  gk->species[1] = ion;
  gk->field = field;
  gk->parallelism = parallelism;

  memcpy(gk->name, ctx.sim_name, sizeof(char[128]));

  // create app object
  if (my_rank == 0) printf("Creating app object ...\n");
  gkyl_gyrokinetic_app *app = gkyl_gyrokinetic_app_new(gk);
  
  // Initial and final simulation times.
  int frame_curr = 0;
  double t_curr = 0.0, t_end = ctx.final_time;
  // Initialize simulation.
  if (app_args.is_restart) {
    struct gkyl_app_restart_status status = gkyl_gyrokinetic_app_read_from_frame(app, app_args.restart_frame);

    if (status.io_status != GKYL_ARRAY_RIO_SUCCESS) {
      gkyl_gyrokinetic_app_cout(app, stderr, "*** Failed to read restart file! (%s)\n",
        gkyl_array_rio_status_msg(status.io_status));
      goto freeresources;
    }

    frame_curr = status.frame;
    t_curr = status.stime;

    gkyl_gyrokinetic_app_cout(app, stdout, "Restarting from frame %d", frame_curr);
    gkyl_gyrokinetic_app_cout(app, stdout, " at time = %g\n", t_curr);
  }
  else {
    gkyl_gyrokinetic_app_apply_ic(app, t_curr);
  }  

  // Create triggers for IO.
  int num_frames = ctx.num_frames, num_int_diag_calc = ctx.int_diag_calc_num;
  struct gkyl_tm_trigger trig_write_conf = { .dt = t_end/num_frames, .tcurr = t_curr, .curr = frame_curr };
  struct gkyl_tm_trigger trig_write_phase = { .dt = t_end/(ctx.write_phase_freq*num_frames), .tcurr = t_curr, .curr = frame_curr};
  struct gkyl_tm_trigger trig_calc_intdiag = { .dt = t_end/GKYL_MAX2(num_frames, num_int_diag_calc),
    .tcurr = t_curr, .curr = frame_curr };

  // Write out ICs (if restart, it overwrites the restart frame).
  if (!app_args.is_restart) {
    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, false);
    write_data(&trig_write_conf, &trig_write_phase, app, t_curr, false);
  }

  // Initial time-step.
  double dt = t_end-t_curr;

  // Initialize small time-step check.
  double dt_init = -1.0, dt_failure_tol = ctx.dt_failure_tol;
  int num_failures = 0, num_failures_max = ctx.num_failures_max;

  // MAIN TIME LOOP
  if (my_rank == 0) printf("Starting main loop ...\n");
  long step = 1, num_steps = app_args.num_steps;
  while ((t_curr < t_end) && (step <= num_steps)) {
    if (step == 1 || step % 100 == 0)
      gkyl_gyrokinetic_app_cout(app, stdout, "Taking time-step at t = %g mus ...", t_curr*1.0e6);
    struct gkyl_update_status status = gkyl_gyrokinetic_update(app, dt);
    if (step == 1 || step % 100 == 0)
      gkyl_gyrokinetic_app_cout(app, stdout, " dt = %g mus\n", status.dt_actual*1.0e6);

    if (!status.success) {
      if (my_rank == 0) gkyl_gyrokinetic_app_cout(app, stdout, "** Update method failed! Aborting simulation ....\n");
      break;
    }
    t_curr += status.dt_actual;
    dt = status.dt_suggested;

    calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, t_curr > t_end);
    write_data(&trig_write_conf, &trig_write_phase, app, t_curr, t_curr > t_end);

    if (dt_init < 0.0) {
      dt_init = status.dt_actual;
    }
    else if (status.dt_actual < dt_failure_tol * dt_init) {
      num_failures += 1;

      gkyl_gyrokinetic_app_cout(app, stdout, "WARNING: Time-step dt = %g", status.dt_actual);
      gkyl_gyrokinetic_app_cout(app, stdout, " is below %g*dt_init ...", dt_failure_tol);
      gkyl_gyrokinetic_app_cout(app, stdout, " num_failures = %d\n", num_failures);
      if (num_failures >= num_failures_max) {
        gkyl_gyrokinetic_app_cout(app, stdout, "ERROR: Time-step was below %g*dt_init ", dt_failure_tol);
        gkyl_gyrokinetic_app_cout(app, stdout, "%d consecutive times. Aborting simulation ....\n", num_failures_max);
        calc_integrated_diagnostics(&trig_calc_intdiag, app, t_curr, true);
        write_data(&trig_write_conf, &trig_write_phase, app, t_curr, true);
        break;
      }
    }
    else {
      num_failures = 0;
    }

    step += 1;

    if (ctx.max_run_time > 0.0 && 
        gkyl_time_diff_now_sec(timer_global) > ctx.max_run_time) {
      if (my_rank == 0) gkyl_gyrokinetic_app_cout(app, stdout, "Reached max run time, exiting ...\n");
      break;
    }

  }
  if (my_rank == 0) printf(" ... finished\n");
  gkyl_gyrokinetic_app_stat_write(app);
  
  // fetch simulation statistics
  struct gkyl_gyrokinetic_stat stat = gkyl_gyrokinetic_app_stat(app);

  gkyl_gyrokinetic_app_cout(app, stdout, "\n");
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of update calls %ld\n", stat.nup);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of forward-Euler calls %ld\n", stat.nfeuler);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-2 failures %ld\n", stat.nstage_2_fail);
  if (stat.nstage_2_fail > 0) {
    gkyl_gyrokinetic_app_cout(app, stdout, "Max rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[1]);
    gkyl_gyrokinetic_app_cout(app, stdout, "Min rel dt diff for RK stage-2 failures %g\n", stat.stage_2_dt_diff[0]);
  }  
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of RK stage-3 failures %ld\n", stat.nstage_3_fail);
  gkyl_gyrokinetic_app_cout(app, stdout, "Number of write calls %ld\n", stat.n_io);
  gkyl_gyrokinetic_app_print_timings(app, stdout);

  freeresources:
  // simulation cCORElete, free app
  gkyl_gyrokinetic_app_release(app);
  gkyl_gyrokinetic_comms_release(comm);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) {
    MPI_Finalize();
  }
#endif
  
  return 0;
}

// End of main function
// ---------------------------------------------------------------

// Function definitions
// ---------------------------------------------------------------

// Write context structure to JSON file
static void write_ctx_to_json(struct gk_app_ctx *ctx, const char *filename)
{
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Failed to open file %s for writing\n", filename);
    return;
  }
  
  fprintf(fp, "{\n");
  fprintf(fp, "  \"sim_name\": \"%s\",\n", ctx->sim_name);
  fprintf(fp, "  \"cdim\": %d,\n", ctx->cdim);
  fprintf(fp, "  \"vdim\": %d,\n", ctx->vdim);
  fprintf(fp, "  \"geometry\": {\n");
  fprintf(fp, "    \"a_shift\": %.12e,\n", ctx->a_shift);
  fprintf(fp, "    \"Z_axis\": %.12e,\n", ctx->Z_axis);
  fprintf(fp, "    \"R_axis\": %.12e,\n", ctx->R_axis);
  fprintf(fp, "    \"R0\": %.12e,\n", ctx->R0);
  fprintf(fp, "    \"a_mid\": %.12e,\n", ctx->a_mid);
  fprintf(fp, "    \"x_inner\": %.12e,\n", ctx->x_inner);
  fprintf(fp, "    \"r0\": %.12e,\n", ctx->r0);
  fprintf(fp, "    \"B0\": %.12e,\n", ctx->B0);
  fprintf(fp, "    \"kappa\": %.12e,\n", ctx->kappa);
  fprintf(fp, "    \"delta\": %.12e,\n", ctx->delta);
  fprintf(fp, "    \"q0\": %.12e,\n", ctx->q0);
  fprintf(fp, "    \"qaxis\": %.12e,\n", ctx->qaxis);
  fprintf(fp, "    \"qlcfs\": %.12e,\n", ctx->qlcfs);
  fprintf(fp, "    \"Bref\": %.12e,\n", ctx->Bref);
  fprintf(fp, "    \"x_LCFS\": %.12e\n", ctx->x_LCFS);
  fprintf(fp, "  },\n");
  fprintf(fp, "  \"plasma\": {\n");
  fprintf(fp, "    \"me\": %.12e,\n", ctx->me);
  fprintf(fp, "    \"qe\": %.12e,\n", ctx->qe);
  fprintf(fp, "    \"mi\": %.12e,\n", ctx->mi);
  fprintf(fp, "    \"qi\": %.12e,\n", ctx->qi);
  fprintf(fp, "    \"n0\": %.12e,\n", ctx->n0);
  fprintf(fp, "    \"Te0\": %.12e,\n", ctx->Te0);
  fprintf(fp, "    \"Ti0\": %.12e,\n", ctx->Ti0);
  fprintf(fp, "    \"nuFrac\": %.12e\n", ctx->nuFrac);
  fprintf(fp, "  },\n");
  fprintf(fp, "  \"initial_conditions\": {\n");
  fprintf(fp, "    \"dens_init_mean\": [%.12e, %.12e, %.12e],\n", 
          ctx->dens_init_mean[0], ctx->dens_init_mean[1], ctx->dens_init_mean[2]);
  fprintf(fp, "    \"dens_init_sigma\": [%.12e, %.12e, %.12e],\n",
          ctx->dens_init_sigma[0], ctx->dens_init_sigma[1], ctx->dens_init_sigma[2]);
  fprintf(fp, "    \"num_particle_init\": %.12e,\n", ctx->num_particle_init);
  fprintf(fp, "    \"energy_init\": %.12e\n", ctx->energy_init);
  fprintf(fp, "  },\n");
  fprintf(fp, "  \"sources\": {\n");
  fprintf(fp, "    \"num_sources\": %.0f,\n", ctx->num_sources);
  fprintf(fp, "    \"core\": {\n");
  fprintf(fp, "      \"adapt_energy\": %s,\n", ctx->adapt_energy_srcCORE ? "true" : "false");
  fprintf(fp, "      \"adapt_particle\": %s,\n", ctx->adapt_particle_srcCORE ? "true" : "false");
  fprintf(fp, "      \"center\": [%.12e, %.12e, %.12e],\n",
          ctx->center_srcCORE[0], ctx->center_srcCORE[1], ctx->center_srcCORE[2]);
  fprintf(fp, "      \"sigma\": [%.12e, %.12e, %.12e],\n",
          ctx->sigma_srcCORE[0], ctx->sigma_srcCORE[1], ctx->sigma_srcCORE[2]);
  fprintf(fp, "      \"energy\": %.12e,\n", ctx->energy_srcCORE);
  fprintf(fp, "      \"particle\": %.12e,\n", ctx->particle_srcCORE);
  fprintf(fp, "      \"floor\": %.12e\n", ctx->floor_srcCORE);
  fprintf(fp, "    },\n");
  fprintf(fp, "    \"recycling\": {\n");
  fprintf(fp, "      \"adapt_energy\": %s,\n", ctx->adapt_energy_srcRECY ? "true" : "false");
  fprintf(fp, "      \"adapt_particle\": %s,\n", ctx->adapt_particle_srcRECY ? "true" : "false");
  fprintf(fp, "      \"center\": [%.12e, %.12e, %.12e],\n",
          ctx->center_srcRECY[0], ctx->center_srcRECY[1], ctx->center_srcRECY[2]);
  fprintf(fp, "      \"sigma\": [%.12e, %.12e, %.12e],\n",
          ctx->sigma_srcRECY[0], ctx->sigma_srcRECY[1], ctx->sigma_srcRECY[2]);
  fprintf(fp, "      \"energy\": %.12e,\n", ctx->energy_srcRECY);
  fprintf(fp, "      \"particle\": %.12e,\n", ctx->particle_srcRECY);
  fprintf(fp, "      \"floor\": %.12e\n", ctx->floor_srcRECY);
  fprintf(fp, "    }\n");
  fprintf(fp, "  },\n");
  fprintf(fp, "  \"grid\": {\n");
  fprintf(fp, "    \"Lx\": %.12e,\n", ctx->Lx);
  fprintf(fp, "    \"Ly\": %.12e,\n", ctx->Ly);
  fprintf(fp, "    \"Lz\": %.12e,\n", ctx->Lz);
  fprintf(fp, "    \"x_min\": %.12e,\n", ctx->x_min);
  fprintf(fp, "    \"x_max\": %.12e,\n", ctx->x_max);
  fprintf(fp, "    \"y_min\": %.12e,\n", ctx->y_min);
  fprintf(fp, "    \"y_max\": %.12e,\n", ctx->y_max);
  fprintf(fp, "    \"z_min\": %.12e,\n", ctx->z_min);
  fprintf(fp, "    \"z_max\": %.12e,\n", ctx->z_max);
  fprintf(fp, "    \"num_cell_x\": %d,\n", ctx->num_cell_x);
  fprintf(fp, "    \"num_cell_y\": %d,\n", ctx->num_cell_y);
  fprintf(fp, "    \"num_cell_z\": %d,\n", ctx->num_cell_z);
  fprintf(fp, "    \"num_cell_vpar\": %d,\n", ctx->num_cell_vpar);
  fprintf(fp, "    \"num_cell_mu\": %d,\n", ctx->num_cell_mu);
  fprintf(fp, "    \"poly_order\": %d,\n", ctx->poly_order);
  fprintf(fp, "    \"vpar_max_elc\": %.12e,\n", ctx->vpar_max_elc);
  fprintf(fp, "    \"mu_max_elc\": %.12e,\n", ctx->mu_max_elc);
  fprintf(fp, "    \"vpar_max_ion\": %.12e,\n", ctx->vpar_max_ion);
  fprintf(fp, "    \"mu_max_ion\": %.12e\n", ctx->mu_max_ion);
  fprintf(fp, "  },\n");
  fprintf(fp, "  \"simulation\": {\n");
  fprintf(fp, "    \"final_time\": %.12e,\n", ctx->final_time);
  fprintf(fp, "    \"num_frames\": %d,\n", ctx->num_frames);
  fprintf(fp, "    \"write_phase_freq\": %.12e,\n", ctx->write_phase_freq);
  fprintf(fp, "    \"int_diag_calc_num\": %d,\n", ctx->int_diag_calc_num);
  fprintf(fp, "    \"dt_failure_tol\": %.12e,\n", ctx->dt_failure_tol);
  fprintf(fp, "    \"num_failures_max\": %d,\n", ctx->num_failures_max);
  fprintf(fp, "    \"max_run_time\": %.12e\n", ctx->max_run_time);
  fprintf(fp, "  }\n");
  fprintf(fp, "}\n");
  
  fclose(fp);
}

// Geometry related functions 
double r_x(double x, double a_mid, double x_inner)
{
  return x+a_mid-x_inner;
} 
// quadratic q profile
double qprofile(double r, double a_mid, double qaxis, double qlcfs)
{
  // This profile ensures:
  // - dq/dr = 0 at r=0
  // - q = qaxis at r=0
  // - q = qlcfs at r=a_mid
  return qaxis + (qlcfs-qaxis)*(r*r)/(a_mid*a_mid);
}
// // cubic q profile
// double qprofile(double r, double a_mid, double qaxis, double qlcfs, double slcfs)
// {
//   // This profile ensures:
//   // - dq/dr = 0 at r=0
//   // - q = qaxis at r=0
//   // - q = qlcfs at r=a_mid
//   // - s = slcfs at r=a_mid
//   double rho = r/a_mid;
//   double dq = qlcfs - qaxis;
//   double alpha = slcfs * qlcfs;
//   double c3 = alpha - 2*dq;
//   double c2 = 3*dq - alpha;
//   double c1 = 0.0;
//   double c0 = qaxis;
//   return c3*rho*rho*rho + c2*rho*rho + c1*rho + c0;
// }
double R_rtheta(double r, double theta, void *ctx)
{
  // Major radius as a function of minor radius r and poloidal angle theta.
  struct gk_app_ctx *app = ctx;
  double a_shift = app->a_shift;
  double R_axis = app->R_axis;
  double delta = app->delta;
  return R_axis - a_shift*r*r/(2.*R_axis) + r*cos(theta + asin(delta)*sin(theta));
}

double Z_rtheta(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double Z_axis = app->Z_axis;
  double kappa = app->kappa;
  return Z_axis + kappa*r*sin(theta);
}

double dRdr(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double a_shift = app->a_shift;
  double R_axis = app->R_axis;
  double delta = app->delta;
  return - a_shift*r/(R_axis) + cos(theta + asin(delta)*sin(theta));
}

double dRdtheta(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double delta = app->delta;
  return -r*sin(theta + asin(delta)*sin(theta))*(1.+asin(delta)*cos(theta));
}

double dZdr(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double kappa = app->kappa;
  return kappa*sin(theta);
}

double dZdtheta(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double kappa = app->kappa;
  return kappa*r*cos(theta);
}

double Jr(double r, double theta, void *ctx)
{
  return R_rtheta(r,theta,ctx)*( dRdr(r,theta,ctx) * dZdtheta(r,theta,ctx)
		                -dRdtheta(r,theta,ctx) * dZdr(r,theta,ctx) );
}

struct integrand_ctx {
  struct gk_app_ctx *app_ctx;
  double r;
};

double integrand(double t, void *int_ctx)
{
  struct integrand_ctx *inctx = int_ctx;
  double r = inctx->r;
  struct gk_app_ctx *app = inctx->app_ctx;
  return Jr(r,t,app) / pow(R_rtheta(r,t,app),2);
}

double dPsidr(double r, double theta, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  struct integrand_ctx tmp_ctx = {.app_ctx = app, .r = r};
  struct gkyl_qr_res integral;
  integral = gkyl_dbl_exp(integrand, &tmp_ctx, 0., 2.*M_PI, 7, 1e-10);

  double B0 = app->B0;
  double a_mid = app->a_mid;
  double R_axis = app->R_axis;
  return ( B0*R_axis/(2.*M_PI*qprofile(r,app->a_mid,app->qaxis,app->qlcfs)))*integral.res;
}

double alpha(double r, double theta, double phi, void *ctx)
{
  double twrap = theta;
  while (twrap < -M_PI) twrap = twrap+2.*M_PI;
  while (M_PI < twrap) twrap = twrap-2.*M_PI;

  struct gk_app_ctx *app = ctx;
  struct integrand_ctx tmp_ctx = {.app_ctx = app, .r = r};
  struct gkyl_qr_res integral;
  if (0. < twrap) {
    integral = gkyl_dbl_exp(integrand, &tmp_ctx, 0., twrap, 7, 1e-10);
  } else {
    integral = gkyl_dbl_exp(integrand, &tmp_ctx, twrap, 0., 7, 1e-10);
    integral.res = -integral.res;
  }

  double B0 = app->B0;
  double R_axis = app->R_axis;

  return phi - B0*R_axis*integral.res/dPsidr(r,theta,ctx);
}

double Bphi(double R, void *ctx)
{
  // Toroidal magnetic field.
  struct gk_app_ctx *app = ctx;
  double B0 = app->B0;
  double R0 = app->R0;
  return B0*R0/R;
}

double gradr(double r, double theta, void *ctx)
{
  return (R_rtheta(r,theta,ctx)/Jr(r,theta,ctx))*sqrt(pow(dRdtheta(r,theta,ctx),2) + pow(dZdtheta(r,theta,ctx),2));
}

void zero_func(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
}

// Density initial condition (like TCV exp profile)
void density_init(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double n0 = 5e19;
  double x0 = -0.03;
  double c1 = 0.5;
  double c2 = 8.0;
  double c3 = 0.005;
  fout[0] = n0*(c1*(1.+tanh(c2*(-10*(x+x0))))+c3);
}

// Electron temperature initial conditions
void temp_elc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double T0 = 200 * GKYL_ELEMENTARY_CHARGE;
  double x0 = -0.03; // position of the transition region
  double c0 = 1.3; // multiplicative factor
  double c1 = 0.5; // control the temperature at the _core
  double c2 = 8.0; // control the width of the transition region
  double c3 = 0.1; // control the temperature at the SOL
  fout[0] = c0*T0*(c1*(1.+tanh(c2*(-10*(x+x0))))+c3);
}

// Ion temperature initial conditions
void temp_ion(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double T0 = 200 * GKYL_ELEMENTARY_CHARGE;
  double x0 = -0.04; // position of the transition region
  double c0 = 1.0; // multiplicative factor
  double c1 = 0.5; // control the temperature at the _core
  double c2 = 3.0; // control the width of the transition region
  double c3 = 0.2; // control the temperature at the SOL
  fout[0] = c0*T0*(c1*(1.+tanh(c2*(-10*(x+x0))))+c3);
}

// Geometry evaluation functions for the gk app
void mapc2p(double t, const double *xc, double* GKYL_RESTRICT xp, void *ctx)
{
  double x = xc[0], y = xc[1], z = xc[2];
  struct gk_app_ctx *app = ctx;
  double r0 = app->r0;
  double q0 = app->q0;
  double a_mid = app->a_mid;
  double x_inner = app->x_inner;
  double r = r_x(x,a_mid,x_inner);
  // Map to cylindrical (R, Z, phi) coordinates.
  double R   = R_rtheta(r, z, ctx);
  double Z   = Z_rtheta(r, z, ctx);
  double phi = -q0/r0*y - alpha(r, z, 0, ctx);
  // Map to Cartesian (X, Y, Z) coordinates.
  double X = R*cos(phi);
  double Y = R*sin(phi);
  xp[0] = X; xp[1] = Y; xp[2] = Z;
}

// Taken from rt gk d3d 3x2c, is this the non uniform v grid mapping?
void mapc2p_vel_elc(double t, const double *vc, double* GKYL_RESTRICT vp, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double vpar_max_elc = app->vpar_max_elc;
  double mu_max_elc = app->mu_max_elc;
  double cvpar = vc[0], cmu = vc[1];
  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vp[0] = vpar_max_elc*cvpar;
  else if (cvpar < -0.5)
    vp[0] = -vpar_max_elc*2.0*pow(cvpar,2);
  else
    vp[0] =  vpar_max_elc*2.0*pow(cvpar,2);
  // Quadratic map in mu.
  vp[1] = mu_max_elc*pow(cmu,2);
}

void mapc2p_vel_ion(double t, const double *vc, double* GKYL_RESTRICT vp, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double vpar_max_ion = app->vpar_max_ion;
  double mu_max_ion = app->mu_max_ion;
  double cvpar = vc[0], cmu = vc[1];
  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vp[0] = vpar_max_ion*cvpar;
  else if (cvpar < -0.5)
    vp[0] = -vpar_max_ion*2.0*pow(cvpar,2);
  else
    vp[0] =  vpar_max_ion*2.0*pow(cvpar,2);
  // Quadratic map in mu.
  vp[1] = mu_max_ion*pow(cmu,2);
}

void bfield_func(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xc[0], y = xc[1], z = xc[2];
  struct gk_app_ctx *app = ctx;
  double a_mid = app->a_mid;
  double r0 = app->r0;
  double q0 = app->q0;
  double x_inner = app->x_inner;
  double r = r_x(x,a_mid,x_inner);
  double Bt = Bphi(R_rtheta(r,z,ctx),ctx);
  double Bp = dPsidr(r,z,ctx)/R_rtheta(r,z,ctx)*gradr(r,z,ctx);

  double drdtheta = dRdtheta(r,z,ctx);
  double dzdtheta = dZdtheta(r,z,ctx);
  double den = sqrt(pow(drdtheta,2) + pow(dzdtheta,2));
  double B_r = Bp*drdtheta/den;
  double B_z = Bp*dzdtheta/den;
  double phi = -q0/r0*y - alpha(r, z, 0, ctx);
  double R = R_rtheta(r, z, ctx);

  // xc are computational coords. 
  // Set Cartesian components of magnetic field.
  fout[0] = -(B_r * cos(phi) - Bt * sin(phi));
  fout[1] = -(B_r * sin(phi) + Bt * cos(phi));
  fout[2] = -B_z;
}

void bc_shift_func_lo(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xc[0];
  struct gk_app_ctx *app = ctx;
  double r = r_x(x, app->a_mid, app->x_inner);

  fout[0] = -app->r0/app->q0*alpha(r, -app->Lz/2.0, 0.0, ctx);
}

void bc_shift_func_up(double t, const double *xc, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xc[0];
  struct gk_app_ctx *app = ctx;
  double r = r_x(x, app->a_mid, app->x_inner);

  fout[0] = -app->r0/app->q0*alpha(r, app->Lz/2.0, 0.0, ctx);
}

void
calc_integrated_diagnostics(struct gkyl_tm_trigger* iot, gkyl_gyrokinetic_app* app, double t_curr, bool force_calc)
{
  if (gkyl_tm_trigger_check_and_bump(iot, t_curr) || force_calc) {
    gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
  }
}

void
write_data(struct gkyl_tm_trigger* iot_conf, struct gkyl_tm_trigger* iot_phase,
  gkyl_gyrokinetic_app* app, double t_curr, bool force_write)
{
  bool trig_now_conf = gkyl_tm_trigger_check_and_bump(iot_conf, t_curr);
  if (trig_now_conf || force_write) {
    int frame = (!trig_now_conf) && force_write? iot_conf->curr : iot_conf->curr-1;

    gkyl_gyrokinetic_app_write_conf(app, t_curr, frame);

    gkyl_gyrokinetic_app_calc_field_energy(app, t_curr);
    gkyl_gyrokinetic_app_write_field_energy(app);

    gkyl_gyrokinetic_app_calc_integrated_mom(app, t_curr);
    gkyl_gyrokinetic_app_write_integrated_mom(app);
  }

  bool trig_now_phase = gkyl_tm_trigger_check_and_bump(iot_phase, t_curr);
  if (trig_now_phase || force_write) {
    int frame = (!trig_now_conf) && force_write? iot_conf->curr : iot_conf->curr-1;

    gkyl_gyrokinetic_app_write_phase(app, t_curr, frame);
  }
}
