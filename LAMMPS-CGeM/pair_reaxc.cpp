/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)
   Per-atom energy/virial added by Ray Shan (Sandia)
   Fix reax/c/bonds and fix reax/c/species for pair_style reax/c added by
        Ray Shan (Sandia)
   Hybrid and hybrid/overlay compatibility added by Ray Shan (Sandia)
------------------------------------------------------------------------- */

#include "pair_reaxc.h"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include "atom.h"
#include "update.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "modify.h"
#include "fix.h"
#include "fix_reaxc.h"
#include "citeme.h"
#include "memory.h"
#include "error.h"
#include "domain.h"

#include "reaxc_defs.h"
#include "reaxc_types.h"
#include "reaxc_allocate.h"
#include "reaxc_control.h"
#include "reaxc_ffield.h"
#include "reaxc_forces.h"
#include "reaxc_init_md.h"
#include "reaxc_io_tools.h"
#include "reaxc_list.h"
#include "reaxc_lookup.h"
#include "reaxc_reset_tools.h"
#include "reaxc_vector.h"

using namespace LAMMPS_NS;

static const char cite_pair_reax_c[] =
  "pair reax/c command:\n\n"
  "@Article{Aktulga12,\n"
  " author = {H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama},\n"
  " title = {Parallel reactive molecular dynamics: Numerical methods and algorithmic techniques},\n"
  " journal = {Parallel Computing},\n"
  " year =    2012,\n"
  " volume =  38,\n"
  " pages =   {245--259}\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

PairReaxC::PairReaxC(LAMMPS *lmp) : Pair(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_pair_reax_c);

  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  ghostneigh = 1;

  fix_id = new char[24];
  snprintf(fix_id,24,"REAXC_%d",instance_me);

  system = (reax_system *)
    memory->smalloc(sizeof(reax_system),"reax:system");
  memset(system,0,sizeof(reax_system));
  control = (control_params *)
    memory->smalloc(sizeof(control_params),"reax:control");
  memset(control,0,sizeof(control_params));
  data = (simulation_data *)
    memory->smalloc(sizeof(simulation_data),"reax:data");
  workspace = (storage *)
    memory->smalloc(sizeof(storage),"reax:storage");
  lists = (reax_list *)
    memory->smalloc(LIST_N * sizeof(reax_list),"reax:lists");
  memset(lists,0,LIST_N * sizeof(reax_list));
  out_control = (output_controls *)
    memory->smalloc(sizeof(output_controls),"reax:out_control");
  memset(out_control,0,sizeof(output_controls));
  mpi_data = (mpi_datatypes *)
    memory->smalloc(sizeof(mpi_datatypes),"reax:mpi");

  control->me = system->my_rank = comm->me;

  system->my_coords[0] = 0;
  system->my_coords[1] = 0;
  system->my_coords[2] = 0;
  system->num_nbrs = 0;
  system->n = 0; // my atoms
  system->N = 0; // mine + ghosts
  system->bigN = 0;  // all atoms in the system
  system->local_cap = 0;
  system->total_cap = 0;
  system->gcell_cap = 0;
  system->bndry_cuts.ghost_nonb = 0;
  system->bndry_cuts.ghost_hbond = 0;
  system->bndry_cuts.ghost_bond = 0;
  system->bndry_cuts.ghost_cutoff = 0;
  system->my_atoms = NULL;
  system->pair_ptr = this;
  system->error_ptr = error;
  control->error_ptr = error;

  system->omp_active = 0;
  
  fix_reax = NULL;
  tmpid = NULL;
  tmpbo = NULL;

  nextra = 14;
  pvector = new double[nextra];

  setup_flag = 0;
  fixspecies_flag = 0;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

PairReaxC::~PairReaxC()
{
  if (copymode) return;

  if (fix_reax) modify->delete_fix(fix_id);
  delete[] fix_id;

  if (setup_flag) {
    Close_Output_Files( system, control, out_control, mpi_data );

    // deallocate reax data-structures

    if (control->tabulate ) Deallocate_Lookup_Tables( system);

    if (control->hbond_cut > 0 )  Delete_List( lists+HBONDS );
    Delete_List( lists+BONDS );
    Delete_List( lists+THREE_BODIES );
    Delete_List( lists+FAR_NBRS );

    DeAllocate_Workspace( control, workspace );
    DeAllocate_System( system );
  }

  memory->destroy( system );
  memory->destroy( control );
  memory->destroy( data );
  memory->destroy( workspace );
  memory->destroy( lists );
  memory->destroy( out_control );
  memory->destroy( mpi_data );

  // deallocate interface storage
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
    delete [] map;

    delete [] chi;
    delete [] eta;
    delete [] gamma;
  }

  memory->destroy(tmpid);
  memory->destroy(tmpbo);

  delete [] pvector;

}

/* ---------------------------------------------------------------------- */

void PairReaxC::allocate( )
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cutghost,n+1,n+1,"pair:cutghost");
  map = new int[n+1];

  chi = new double[n+1];
  eta = new double[n+1];
  gamma = new double[n+1];
}

/* ---------------------------------------------------------------------- */

void PairReaxC::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal pair_style command");

  // read name of control file or use default controls

  if (strcmp(arg[0],"NULL") == 0) {
    strcpy( control->sim_name, "simulate" );
    control->ensemble = 0;
    out_control->energy_update_freq = 0;
    control->tabulate = 0;

    control->reneighbor = 1;
    control->vlist_cut = control->nonb_cut;
    control->bond_cut = 5.;
    control->hbond_cut = 7.50;
    control->thb_cut = 0.001;
    control->thb_cutsq = 0.00001;
    control->bg_cut = 0.3;

    // Initialize for when omp style included
    control->nthreads = 1;

    out_control->write_steps = 0;
    out_control->traj_method = 0;
    strcpy( out_control->traj_title, "default_title" );
    out_control->atom_info = 0;
    out_control->bond_info = 0;
    out_control->angle_info = 0;
  } else Read_Control_File(arg[0], control, out_control);

  // default values

  qeqflag = 1;
  control->lgflag = 0;
  control->enobondsflag = 1;
  system->mincap = MIN_CAP;
  system->safezone = SAFE_ZONE;
  system->saferzone = SAFER_ZONE;

  // process optional keywords

  int iarg = 1;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"checkqeq") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
      if (strcmp(arg[iarg+1],"yes") == 0) qeqflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) qeqflag = 0;
      else error->all(FLERR,"Illegal pair_style reax/c command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"enobonds") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
      if (strcmp(arg[iarg+1],"yes") == 0) control->enobondsflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) control->enobondsflag = 0;
      else error->all(FLERR,"Illegal pair_style reax/c command");
      iarg += 2;
  } else if (strcmp(arg[iarg],"lgvdw") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
      if (strcmp(arg[iarg+1],"yes") == 0) control->lgflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) control->lgflag = 0;
      else error->all(FLERR,"Illegal pair_style reax/c command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"safezone") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
      system->safezone = force->numeric(FLERR,arg[iarg+1]);
      if (system->safezone < 0.0)
        error->all(FLERR,"Illegal pair_style reax/c safezone command");
      system->saferzone = system->safezone*1.2 + 0.2;
      iarg += 2;
    } else if (strcmp(arg[iarg],"mincap") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
      system->mincap = force->inumeric(FLERR,arg[iarg+1]);
      if (system->mincap < 0)
        error->all(FLERR,"Illegal pair_style reax/c mincap command");
      iarg += 2;
    } else error->all(FLERR,"Illegal pair_style reax/c command");
  }

  // LAMMPS is responsible for generating nbrs

  control->reneighbor = 1;
}

/* ---------------------------------------------------------------------- */

void PairReaxC::coeff( int nargs, char **args )
{
  if (!allocated) allocate();

  if (nargs != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(args[0],"*") != 0 || strcmp(args[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read ffield file

  char *file = args[2];
  FILE *fp;
  fp = force->open_potential(file);
  if (fp != NULL)
    Read_Force_Field(fp, &(system->reax_param), control);
  else {
      char str[128];
      snprintf(str,128,"Cannot open ReaxFF potential file %s",file);
      error->all(FLERR,str);
  }

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL

  int itmp = 0;
  int nreax_types = system->reax_param.num_atom_types;
  for (int i = 3; i < nargs; i++) {
    if (strcmp(args[i],"NULL") == 0) {
      map[i-2] = -1;
      itmp ++;
      continue;
    }
  }

  int n = atom->ntypes;

  // pair_coeff element map
  for (int i = 3; i < nargs; i++)
    for (int j = 0; j < nreax_types; j++)
      if (strcasecmp(args[i],system->reax_param.sbp[j].name) == 0) {
        map[i-2] = j;
        itmp ++;
      }

  // error check
  if (itmp != n)
    error->all(FLERR,"Non-existent ReaxFF type");

  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

}

/* ---------------------------------------------------------------------- */

void PairReaxC::init_style( )
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style reax/c requires atom attribute q");

  // firstwarn = 1;

  int iqeq;
  for (iqeq = 0; iqeq < modify->nfix; iqeq++)
    if (strstr(modify->fix[iqeq]->style,"qeq/reax")
       || strstr(modify->fix[iqeq]->style,"qeq/shielded")) break;
  if (iqeq == modify->nfix && qeqflag == 1)
    error->all(FLERR,"Pair reax/c requires use of fix qeq/reax");

  system->n = atom->nlocal; // my atoms
  system->N = atom->nlocal + atom->nghost; // mine + ghosts
  system->bigN = static_cast<int> (atom->natoms);  // all atoms in the system
  system->wsize = comm->nprocs;

  system->big_box.V = 0;
  system->big_box.box_norms[0] = 0;
  system->big_box.box_norms[1] = 0;
  system->big_box.box_norms[2] = 0;

  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style reax/c requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style reax/c requires newton pair on");
  if ((atom->map_tag_max > 99999999) && (comm->me == 0))
    error->warning(FLERR,"Some Atom-IDs are too large. Pair style reax/c "
                   "native output files may get misformatted or corrupted");

  // because system->bigN is an int, we cannot have more atoms than MAXSMALLINT

  if (atom->natoms > MAXSMALLINT)
    error->all(FLERR,"Too many atoms for pair style reax/c");

  // need a half neighbor list w/ Newton off and ghost neighbors
  // built whenever re-neighboring occurs

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->newton = 2;
  neighbor->requests[irequest]->ghost = 1;

  cutmax = MAX3(control->nonb_cut, control->hbond_cut, control->bond_cut);
  if ((cutmax < 2.0*control->bond_cut) && (comm->me == 0))
    error->warning(FLERR,"Total cutoff < 2*bond cutoff. May need to use an "
                   "increased neighbor list skin.");

  for( int i = 0; i < LIST_N; ++i )
    if (lists[i].allocated != 1)
      lists[i].allocated = 0;

  if (fix_reax == NULL) {
    char **fixarg = new char*[3];
    fixarg[0] = (char *) fix_id;
    fixarg[1] = (char *) "all";
    fixarg[2] = (char *) "REAXC";
    modify->add_fix(3,fixarg);
    delete [] fixarg;
    fix_reax = (FixReaxC *) modify->fix[modify->nfix-1];
  }
}

/* ---------------------------------------------------------------------- */

void PairReaxC::setup( )
{
  int oldN;
  int mincap = system->mincap;
  double safezone = system->safezone;

  system->n = atom->nlocal; // my atoms
  system->N = atom->nlocal + atom->nghost; // mine + ghosts
  oldN = system->N;
  system->bigN = static_cast<int> (atom->natoms);  // all atoms in the system

  system->my_box.xlo=domain->boxlo[0]; //CGeM
  system->my_box.ylo=domain->boxlo[1];
  system->my_box.zlo=domain->boxlo[2];
  system->my_box.xhi=domain->boxhi[0];
  system->my_box.yhi=domain->boxhi[1];
  system->my_box.zhi=domain->boxhi[2];
    
  if (setup_flag == 0) {

    setup_flag = 1;

    int *num_bonds = fix_reax->num_bonds;
    int *num_hbonds = fix_reax->num_hbonds;

    control->vlist_cut = neighbor->cutneighmax;

    // determine the local and total capacity

    system->local_cap = MAX( (int)(system->n * safezone), mincap );
    system->total_cap = MAX( (int)(system->N * safezone), mincap );

    // initialize my data structures

    PreAllocate_Space( system, control, workspace );
    write_reax_atoms();

    int num_nbrs = estimate_reax_lists();
    if(!Make_List(system->total_cap, num_nbrs, TYP_FAR_NEIGHBOR,
                  lists+FAR_NBRS))
      error->one(FLERR,"Pair reax/c problem in far neighbor list");
    (lists+FAR_NBRS)->error_ptr=error;

    write_reax_lists();
    Initialize( system, control, data, workspace, &lists, out_control,
                mpi_data, world );
    for( int k = 0; k < system->N; ++k ) {
      num_bonds[k] = system->my_atoms[k].num_bonds;
      num_hbonds[k] = system->my_atoms[k].num_hbonds;
    }

  } else {

    // fill in reax datastructures

    write_reax_atoms();

    // reset the bond list info for new atoms

    for(int k = oldN; k < system->N; ++k)
      Set_End_Index( k, Start_Index( k, lists+BONDS ), lists+BONDS );

    // check if I need to shrink/extend my data-structs

    ReAllocate( system, control, data, workspace, &lists );
  }

  bigint local_ngroup = list->inum;
  MPI_Allreduce( &local_ngroup, &ngroup, 1, MPI_LMP_BIGINT, MPI_SUM, world );
}

/* ---------------------------------------------------------------------- */

double PairReaxC::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  cutghost[i][j] = cutghost[j][i] = cutmax;
  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairReaxC::compute(int eflag, int vflag)
{
  double evdwl,ecoul;
  double t_start, t_end;

  // communicate num_bonds once every reneighboring
  // 2 num arrays stored by fix, grab ptr to them

  if (neighbor->ago == 0) comm->forward_comm_fix(fix_reax);
  int *num_bonds = fix_reax->num_bonds;
  int *num_hbonds = fix_reax->num_hbonds;

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  if (vflag_global) control->virial = 1;
  else control->virial = 0;

  system->n = atom->nlocal; // my atoms
  system->N = atom->nlocal + atom->nghost; // mine + ghosts
  system->bigN = static_cast<int> (atom->natoms);  // all atoms in the system

  system->big_box.V = 0;
  system->big_box.box_norms[0] = 0;
  system->big_box.box_norms[1] = 0;
  system->big_box.box_norms[2] = 0;
  if (comm->me == 0 ) t_start = MPI_Wtime();

  // setup data structures

  setup();

  Reset( system, control, data, workspace, &lists );
  workspace->realloc.num_far = write_reax_lists();
  // timing for filling in the reax lists
  if (comm->me == 0) {
    t_end = MPI_Wtime();
    data->timing.nbrs = t_end - t_start;
  }

  // forces
  //double Energy;
  //Energy = CGeM_Energy(system,control,data,workspace);
  //printf("Energy=%.8f\n",Energy);
  //exit(0);
  double Energy,e_ele;
  
  //if(data->step==0){
  //  CGeM_Minimization(system,control,data,workspace);
  // }
  Energy = CGeM_Forces(system,control,data,workspace);
  //Energy = CGeM_Energy(system,control,data,workspace);
  //Energy=0;
  e_ele=data->my_en.e_ele=Energy;
  
  //CGeM_Forces(system,control,data,workspace);
    
  //***
  /*
  double dev=0.00001,Eng_p,Eng_m,deriv;
  
  for(int i = 0; i < system->N; i++ ) {
    for(int j = 0; j<3;j++){
      atom->x[i][j]+=dev;
      Eng_p = 23.06*CGeM_Energy(system,control,data,workspace);
      atom->x[i][j]-=2.0*dev;
      Eng_m = 23.06*CGeM_Energy(system,control,data,workspace);
      deriv=(Eng_p-Eng_m)/(2.0*dev);
      printf("i=%i type=%i dir=%i deriv=%.16f\n",i,atom->type[i],j,deriv);
      atom->x[i][j]+=dev;
      //atom->f[i][1]*=23.06;
      //atom->f[i][2]*=23.06;
    }
  }
  exit(0);
  */
  //***
  
  Compute_Forces(system,control,data,workspace,&lists,out_control,mpi_data);
  system->pair_ptr->ev_tally(1,1,system->n,1,0,e_ele,0,0,0,0);      

  //for( i = 0; i < system->N; i++ ) {
  //  type_i = map[atom->type[i]];
  //  printf("i= %i orig_id=%i type_i=%i my_atomsf[i][0]=%.8f \n",i,atom->tag[i],type_i,atom->f[i][0]);
  //} 
  read_reax_forces(vflag);
  //for( i = 0; i < system->N; i++ ) {
  //  type_i = map[atom->type[i]];
  //  printf("i= %i orig_id=%i type_i=%i my_atomsf[i][0]=%.8f \n",i,atom->tag[i],type_i,atom->f[i][0]);
  //}
  //exit(0);
  //comm->forward_comm();
  for(int k = 0; k < system->N; ++k) {
    num_bonds[k] = system->my_atoms[k].num_bonds;
    num_hbonds[k] = system->my_atoms[k].num_hbonds;
  }
  
  // energies and pressure
  
  if (eflag_global) {
    evdwl += data->my_en.e_bond;
    evdwl += data->my_en.e_ov;
    evdwl += data->my_en.e_un;
    evdwl += data->my_en.e_lp;
    evdwl += data->my_en.e_ang;
    evdwl += data->my_en.e_pen;
    evdwl += data->my_en.e_coa;
    evdwl += data->my_en.e_hb;
    evdwl += data->my_en.e_tor;
    evdwl += data->my_en.e_con;
    evdwl += data->my_en.e_vdW;

    ecoul += data->my_en.e_ele;
    ecoul += data->my_en.e_pol;
    
    //printf("e_bond=%.8f= ov=%.8f un=%.8f lp=%.8f ang=%.8f pen=%.8f coa=%.8f hb=%.8f tor=%.8f con=%.8f vdW=%.8f ele=%.8f pol=%.8f \n",data->my_en.e_bond,data->my_en.e_ov,data->my_en.e_un,data->my_en.e_lp,data->my_en.e_ang,data->my_en.e_pen,data->my_en.e_coa,data->my_en.e_hb,data->my_en.e_tor,data->my_en.e_con,data->my_en.e_vdW,data->my_en.e_ele,data->my_en.e_pol);
    //eng_vdwl += evdwl;
    //eng_coul += ecoul;

    // Store the different parts of the energy
    // in a list for output by compute pair command

    pvector[0] = data->my_en.e_bond;
    pvector[1] = data->my_en.e_ov + data->my_en.e_un;
    pvector[2] = data->my_en.e_lp;
    pvector[3] = 0.0;
    pvector[4] = data->my_en.e_ang;
    pvector[5] = data->my_en.e_pen;
    pvector[6] = data->my_en.e_coa;
    pvector[7] = data->my_en.e_hb;
    pvector[8] = data->my_en.e_tor;
    pvector[9] = data->my_en.e_con;
    pvector[10] = data->my_en.e_vdW;
    pvector[11] = data->my_en.e_ele;
    pvector[12] = 0.0;
    pvector[13] = data->my_en.e_pol;
  }

  //printf("ecoul=%.8f vdwl=8.%f\n",ecoul,evdwl);
  if (vflag_fdotr) virial_fdotr_compute();

// Set internal timestep counter to that of LAMMPS

  data->step = update->ntimestep;

  Output_Results( system, control, data, &lists, out_control, mpi_data );

  // populate tmpid and tmpbo arrays for fix reax/c/species
  int i, j;

  if(fixspecies_flag) {
    if (system->N > nmax) {
      memory->destroy(tmpid);
      memory->destroy(tmpbo);
      nmax = system->N;
      memory->create(tmpid,nmax,MAXSPECBOND,"pair:tmpid");
      memory->create(tmpbo,nmax,MAXSPECBOND,"pair:tmpbo");
    }

    for (i = 0; i < system->N; i ++)
      for (j = 0; j < MAXSPECBOND; j ++) {
        tmpbo[i][j] = 0.0;
        tmpid[i][j] = 0;
      }
    FindBond();
  }

}

/* ---------------------------------------------------------------------- */

void PairReaxC::write_reax_atoms()
{
  int *num_bonds = fix_reax->num_bonds;
  int *num_hbonds = fix_reax->num_hbonds;

  if (system->N > system->total_cap)
    error->all(FLERR,"Too many ghost atoms");

  for( int i = 0; i < system->N; ++i ){
    system->my_atoms[i].orig_id = atom->tag[i];
    system->my_atoms[i].type = map[atom->type[i]];
    system->my_atoms[i].x[0] = atom->x[i][0];
    system->my_atoms[i].x[1] = atom->x[i][1];
    system->my_atoms[i].x[2] = atom->x[i][2];
    system->my_atoms[i].q = atom->q[i];
    system->my_atoms[i].num_bonds = num_bonds[i];
    system->my_atoms[i].num_hbonds = num_hbonds[i];
  }
}

/* ---------------------------------------------------------------------- */

void PairReaxC::get_distance( rvec xj, rvec xi, double *d_sqr, rvec *dvec )
{
  (*dvec)[0] = xj[0] - xi[0];
  (*dvec)[1] = xj[1] - xi[1];
  (*dvec)[2] = xj[2] - xi[2];
  *d_sqr = SQR((*dvec)[0]) + SQR((*dvec)[1]) + SQR((*dvec)[2]);
}

/* ---------------------------------------------------------------------- */

void PairReaxC::set_far_nbr( far_neighbor_data *fdest,
                              int j, double d, rvec dvec )
{
  fdest->nbr = j;
  fdest->d = d;
  rvec_Copy( fdest->dvec, dvec );
  ivec_MakeZero( fdest->rel_box );
}

/* ---------------------------------------------------------------------- */

int PairReaxC::estimate_reax_lists()
{
  int itr_i, itr_j, i, j;
  int num_nbrs, num_marked;
  int *ilist, *jlist, *numneigh, **firstneigh, *marked;
  double d_sqr;
  rvec dvec;
  double **x;

  int mincap = system->mincap;
  double safezone = system->safezone;

  x = atom->x;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  num_nbrs = 0;
  num_marked = 0;
  marked = (int*) calloc( system->N, sizeof(int) );

  int numall = list->inum + list->gnum;

  for( itr_i = 0; itr_i < numall; ++itr_i ){
    i = ilist[itr_i];
    marked[i] = 1;
    ++num_marked;
    jlist = firstneigh[i];

    for( itr_j = 0; itr_j < numneigh[i]; ++itr_j ){
      j = jlist[itr_j];
      j &= NEIGHMASK;
      get_distance( x[j], x[i], &d_sqr, &dvec );

      if (d_sqr <= SQR(control->nonb_cut))
        ++num_nbrs;
    }
  }

  free( marked );

  return static_cast<int> (MAX( num_nbrs*safezone, mincap*MIN_NBRS ));
}

/* ---------------------------------------------------------------------- */

int PairReaxC::write_reax_lists()
{
  int itr_i, itr_j, i, j;
  int num_nbrs;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double d_sqr, cutoff_sqr;
  rvec dvec;
  double *dist, **x;
  reax_list *far_nbrs;
  far_neighbor_data *far_list;

  x = atom->x;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  far_nbrs = lists + FAR_NBRS;
  far_list = far_nbrs->select.far_nbr_list;

  num_nbrs = 0;
  int inum = list->inum;
  dist = (double*) calloc( system->N, sizeof(double) );

  int numall = list->inum + list->gnum;

  for( itr_i = 0; itr_i < numall; ++itr_i ){
    i = ilist[itr_i];
    jlist = firstneigh[i];
    Set_Start_Index( i, num_nbrs, far_nbrs );

    if (i < inum)
      cutoff_sqr = control->nonb_cut*control->nonb_cut;
    else
      cutoff_sqr = control->nonb_cut*control->nonb_cut;
      //cutoff_sqr = control->bond_cut*control->bond_cut;

    for( itr_j = 0; itr_j < numneigh[i]; ++itr_j ){
      j = jlist[itr_j];
      j &= NEIGHMASK;
      get_distance( x[j], x[i], &d_sqr, &dvec );

      if (d_sqr <= (cutoff_sqr)) {
        dist[j] = sqrt( d_sqr );
        set_far_nbr( &far_list[num_nbrs], j, dist[j], dvec );
        ++num_nbrs;
      }
    }
    Set_End_Index( i, num_nbrs, far_nbrs );
  }

  free( dist );

  return num_nbrs;
}

/* ---------------------------------------------------------------------- */

void PairReaxC::read_reax_forces(int /*vflag*/)
{
  for( int i = 0; i < system->N; ++i ) {
    
    //if(system->my_atoms[i].type!=12){
    int type_i=system->my_atoms[i].type;
    //if(strcmp(system->reax_param.sbp[type_i].name,"EL")!=0){  
    system->my_atoms[i].f[0] = workspace->f[i][0];
    system->my_atoms[i].f[1] = workspace->f[i][1];
    system->my_atoms[i].f[2] = workspace->f[i][2];
    
    atom->f[i][0] += -workspace->f[i][0];
    atom->f[i][1] += -workspace->f[i][1];
    atom->f[i][2] += -workspace->f[i][2];
    //}
    /*
      else {
      system->my_atoms[i].f[0] = 0.0;
      system->my_atoms[i].f[1] = 0.0;
      system->my_atoms[i].f[2] = 0.0;
      atom->f[i][0] = 0.0; 
      atom->f[i][1] = 0.0; 
      atom->f[i][2] = 0.0;
      atom->v[i][0] = 0.0; 
      atom->v[i][1] = 0.0; 
      atom->v[i][2] = 0.0;
      //atom->x[i][0] = system->my_atoms[i].x[0];
      //atom->x[i][1] = system->my_atoms[i].x[1];
      //atom->x[i][2] = system->my_atoms[i].x[2];  
      }
    */
  }
  
}

/* ---------------------------------------------------------------------- */

void *PairReaxC::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str,"chi") == 0 && chi) {
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) chi[i] = system->reax_param.sbp[map[i]].chi;
      else chi[i] = 0.0;
    return (void *) chi;
  }
  if (strcmp(str,"eta") == 0 && eta) {
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) eta[i] = system->reax_param.sbp[map[i]].eta;
      else eta[i] = 0.0;
    return (void *) eta;
  }
  if (strcmp(str,"gamma") == 0 && gamma) {
    for (int i = 1; i <= atom->ntypes; i++)
      if (map[i] >= 0) gamma[i] = system->reax_param.sbp[map[i]].gamma;
      else gamma[i] = 0.0;
    return (void *) gamma;
  }
  return NULL;
}

/* ---------------------------------------------------------------------- */

double PairReaxC::memory_usage()
{
  double bytes = 0.0;

  // From pair_reax_c
  bytes += 1.0 * system->N * sizeof(int);
  bytes += 1.0 * system->N * sizeof(double);

  // From reaxc_allocate: BO
  bytes += 1.0 * system->total_cap * sizeof(reax_atom);
  bytes += 19.0 * system->total_cap * sizeof(double);
  bytes += 3.0 * system->total_cap * sizeof(int);

  // From reaxc_lists
  bytes += 2.0 * lists->n * sizeof(int);
  bytes += lists->num_intrs * sizeof(three_body_interaction_data);
  bytes += lists->num_intrs * sizeof(bond_data);
  bytes += lists->num_intrs * sizeof(dbond_data);
  bytes += lists->num_intrs * sizeof(dDelta_data);
  bytes += lists->num_intrs * sizeof(far_neighbor_data);
  bytes += lists->num_intrs * sizeof(hbond_data);

  if(fixspecies_flag)
    bytes += 2 * nmax * MAXSPECBOND * sizeof(double);

  return bytes;
}

/* ---------------------------------------------------------------------- */

void PairReaxC::FindBond()
{
  int i, j, pj, nj;
  double bo_tmp, bo_cut;

  bond_data *bo_ij;
  bo_cut = 0.10;

  for (i = 0; i < system->n; i++) {
    nj = 0;
    for( pj = Start_Index(i, lists); pj < End_Index(i, lists); ++pj ) {
      bo_ij = &( lists->select.bond_list[pj] );
      j = bo_ij->nbr;
      if (j < i) continue;

      bo_tmp = bo_ij->bo_data.BO;

      if (bo_tmp >= bo_cut ) {
        tmpid[i][nj] = j;
        tmpbo[i][nj] = bo_tmp;
        nj ++;
        if (nj > MAXSPECBOND) error->all(FLERR,"Increase MAXSPECBOND in reaxc_defs.h");
      }
    }
  }
}

void PairReaxC::CGeM_Minimization(reax_system *system,control_params *control,
		       simulation_data *data, storage *workspace)
{

  double Coulomb_const =14.4;
  int i=0;

  double E_elec =0.0;
  double E_Gauss=0.0;
  double Energy;
  double dt=0.01;
  double dt_original=dt;
  int DELAYSTEP = 5;
  double DT_GROW = 1.1;
  double DT_SHRINK= 0.5;
  double ALPHA0 =0.1;
  double ALPHA_SHRINK= 0.99;
  double TMAX =10.0;
  double alpha_fire=0.1;
  double alpha_final = 0.0;
  int flag=1;
  //maxiter=1000000
  double maxiter=150;
  int niter=0;
  int ntimestep=0;
  double dtmax=dt*TMAX;
  double E_Deviation =0.1;
  double Tolerence = 1e-7;
  int print_E_flag=1;
  int print_traj_flag=0;
  int print_Final_traj_flag=1;
  int    count=0;
  double  vdotvall = 0.0;
  int    iter=0;
  double E_old;
  int type_i;
  double fdotfall,scale1,scale2,vdotfall;
  double EPS_ENERGY = 1.0e-8;
  double etol = 1.0e-4;
  //comm->forward_comm();
  Energy = CGeM_Energy(system,control,data,workspace);
  //printf("%i %.8f %.8f\n",iter,Energy,E_Deviation);
  E_old = Energy+10;
  comm->forward_comm();
  int flagall = 1;
  CGeM_Forces(system,control,data,workspace);

  comm->reverse_comm();
  //exit(0);

  if(data->step==0)maxiter=1000;
  else maxiter=100;

  double box_size_x=system->my_box.xhi-system->my_box.xlo;
  double box_size_y=system->my_box.yhi-system->my_box.ylo;
  double box_size_z=system->my_box.zhi-system->my_box.zlo;

  //for (iter=0;(iter < maxiter && E_Deviation > Tolerence) || iter < 2;iter++){
  for (iter=0;iter < maxiter;iter++ || flagall==1){
    
    vdotfall = 0.0;
    for( i = 0; i < system->n; i++ ) {
      type_i = map[atom->type[i]];
      if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0)
	vdotfall += atom->v[i][0]*atom->f[i][0] + atom->v[i][1]*atom->f[i][1] + atom->v[i][2]*atom->f[i][2];
    }

    if (vdotfall > 0.0){
      vdotvall = 0.0;

      for( i = 0; i < system->n; i++ ) {
	type_i = map[atom->type[i]];
	if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0)
	  vdotvall += atom->v[i][0] *atom->v[i][0] + atom->v[i][1]*atom->v[i][1] + atom->v[i][2]*atom->v[i][2];
      }

      fdotfall = 0.0;

      for( i = 0; i < system->n; i++ ) {
	type_i = map[atom->type[i]];
	if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0)
	  fdotfall += atom->f[i][0]*atom->f[i][0] + atom->f[i][1]*atom->f[i][1] + atom->f[i][2]*atom->f[i][2];
      }

      scale1 = 1.0 - alpha_fire;
      if (fdotfall == 0.0) scale2 = 0.0;
      else scale2 = alpha_fire * sqrt(vdotvall/fdotfall);


      for( i = 0; i < system->n; i++ ) {
	type_i = map[atom->type[i]];
	if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0){
	  atom->v[i][0] = scale1*atom->v[i][0] + scale2*atom->f[i][0];
	  atom->v[i][1] = scale1*atom->v[i][1] + scale2*atom->f[i][1];
	  atom->v[i][2] = scale1*atom->v[i][2] + scale2*atom->f[i][2];
	}
      }
      //printf("iter=%i scale1=%8f scale2=%.8f  alpha_fire=%.8f vdotvall=%.8f fdotvall=%.8f\n",iter,scale1,scale2,alpha_fire,vdotvall,fdotfall);
      if(count > DELAYSTEP){
	dt = MIN(dt*DT_GROW,dtmax);
	alpha_fire *= ALPHA_SHRINK;
      }
    }
    else{
      dt *= DT_SHRINK;
      alpha_fire = ALPHA0;

      for( i = 0; i < system->n; i++ ) {
	type_i = map[atom->type[i]];
	if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0){
	  atom->v[i][0]=atom->v[i][1]=atom->v[i][2]=0.0;
	}
      }
    }
    for( i = 0; i < system->n; i++ ) {
      type_i = map[atom->type[i]];
      if(strcmp(system->reax_param.sbp[type_i].name,"EL" )==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0){
	
	atom->v[i][0] += 0.5*dt * atom->f[i][0];
	atom->v[i][1] += 0.5*dt * atom->f[i][1];
	atom->v[i][2] += 0.5*dt * atom->f[i][2];
      }
    }
    //exit(0);
    for( i = 0; i < system->n; i++ ) {
      type_i = map[atom->type[i]];
      if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0){
	
	atom->x[i][0] += dt * atom->v[i][0];
	atom->x[i][1] += dt * atom->v[i][1];
	atom->x[i][2] += dt * atom->v[i][2];
	
      }
    }
    comm->forward_comm();
    E_old= Energy;
    Energy = CGeM_Energy(system,control,data,workspace); //here!!
    
    //****New exit****
    
    if(E_old < Energy){
      for( i = 0; i < system->n; i++ ) {
	type_i = map[atom->type[i]];
	if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0){
	  atom->x[i][0] -= dt * atom->v[i][0];
	  atom->x[i][1] -= dt * atom->v[i][1];
	  atom->x[i][2] -= dt * atom->v[i][2];
	}
      }
      dt=dt_original*0.1;
      dt_original=dt;
      
      for( i = 0; i < system->n; i++ ) {
	type_i = map[atom->type[i]];
	if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0){
	  atom->v[i][0]=atom->v[i][1]=atom->v[i][2]=0.0;
	}
      }
      CGeM_Forces(system,control,data,workspace); //here!!
      
      //comm->forward_comm();
      comm->reverse_comm();
      
      }
    else{
      CGeM_Forces(system,control,data,workspace); //here!!
      
      //comm->forward_comm();
      comm->reverse_comm();
      
      for( i = 0; i < system->n; i++ ) {
	type_i = map[atom->type[i]];
	if(strcmp(system->reax_param.sbp[type_i].name,"EL")==0 || strcmp(system->reax_param.sbp[type_i].name,"DU")==0){
	atom->v[i][0] += 0.5*dt*atom->f[i][0];
	atom->v[i][1] += 0.5*dt*atom->f[i][1];
	atom->v[i][2] += 0.5*dt*atom->f[i][2];
	//printf("iter=%i i=%i dt=%.8f v[i][0]=%.8f v[i][1]=%.8f v[i][2]=%.8f \n",iter,i,dt,atom->v[i][0],atom->v[i][1],atom->v[i][2]);
	//printf("iter=%i i=%i dt=%.8f f[i][0]=%.8f f[i][1]=%.8f f[i][2]=%.8f \n",iter,i,dt,atom->f[i][0],atom->f[i][1],atom->f[i][2]);
	}
      }
    }
    
    if(Energy > E_old)E_Deviation=1.0;
    else E_Deviation = E_old-Energy;
    
    //printf("%i %.8f %.8f\n",iter,Energy,E_Deviation);
    
    count+=1;
    
    
  }
}

double PairReaxC::CGeM_dRij( reax_system *system,int i,int j,int dir,double R)
{
  //if(R != 0)return (atom->[i].x[dir]-atom->[j].x[dir])/R;
  double box_size;
  if(dir==0)box_size=system->my_box.xhi-system->my_box.xlo;
  else if(dir==1)box_size=system->my_box.yhi-system->my_box.ylo;
  else if(dir==2)box_size=system->my_box.zhi-system->my_box.zlo;

  if(R != 0)return ((atom->x[i][dir]-atom->x[j][dir] - rint( (atom->x[j][dir]-atom->x[i][dir])/(box_size) )*(box_size))/R);
  else return 0.0;
  //if(R != 0)return ((atom->x[i][dir]-atom->x[j][dir])/R);
  
}

double PairReaxC::d_erf_di( reax_system *system,double erf_alpha,double R)
{
  return 2.0/sqrt(constPI)*exp(-pow((erf_alpha*R),2.0))*erf_alpha;
}
void PairReaxC::distance(reax_system *system, rvec xj, rvec xi, double *d_sqr, rvec *dvec )
{
  double x_box_size,y_box_size,z_box_size;
  double box_low_x=system->my_box.xlo;
  double box_low_y=system->my_box.ylo;
  double box_low_z=system->my_box.zlo;
  double box_hi_x=system->my_box.xhi;
  double box_hi_y=system->my_box.yhi;
  double box_hi_z=system->my_box.zhi;

  x_box_size=system->my_box.xhi-system->my_box.xlo;
  y_box_size=system->my_box.yhi-system->my_box.ylo;
  z_box_size=system->my_box.zhi-system->my_box.zlo;
  //double x_floor;
  //x_floor=int( (xj[0]-xi[0])/(0.5*x_box_size) )*(x_box_size);

  (*dvec)[0] = xj[0] - xi[0] - rint( (xj[0]-xi[0])/(x_box_size) )*(x_box_size);
  (*dvec)[1] = xj[1] - xi[1] - rint( (xj[1]-xi[1])/(y_box_size) )*(y_box_size);
  (*dvec)[2] = xj[2] - xi[2] - rint( (xj[2]-xi[2])/(z_box_size) )*(z_box_size);
  //(*dvec)[0] = xj[0] - xi[0];
  //(*dvec)[1] = xj[1] - xi[1];
  //(*dvec)[2] = xj[2] - xi[2];

  //printf("box_x_floor=%.8f xj-xi=%.8f x_box_size=%.8f\n",x_floor,xj[0]-xi[0],x_box_size);

  *d_sqr = pow(SQR((*dvec)[0]) + SQR((*dvec)[1]) + SQR((*dvec)[2]),0.5);
}

double PairReaxC::CGeM_Energy(reax_system *system,control_params *control,
		   simulation_data *data, storage *workspace)
{

  double E_elec;
  double E_Gauss;
  double E_tot;
  int i,j,type_i,type_j;
  two_body_parameters *twbp;
  far_neighbor_data *nbr_pj;
  reax_list *far_nbrs;
  int pj, natoms;
  int start_i, end_i, flag;
  rc_tagint orig_i, orig_j;
  double r_ij,SMALL = 0.0001;
  int type_core;
  double Tap;
  type_core=0;
  E_elec =0.0;
  E_Gauss=0.0;
  E_tot=0;
  
  natoms = system->n;
  //far_nbrs = (*lists) + FAR_NBRS;
  far_nbrs = lists + FAR_NBRS;

  for( i = 0; i < natoms; ++i ) {
    type_i  = map[atom->type[i]];
    type_core=type_i;
    if (type_i < 0) continue;

    //std::cout<<atom->type[i]<<std::endl;
    start_i = Start_Index(i,far_nbrs);
    end_i   = End_Index(i,far_nbrs);
    orig_i  = atom->tag[i];
    for( pj = start_i; pj < end_i; ++pj ) {
      nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
      j = nbr_pj->nbr;
      type_j = map[atom->type[j]];
      if (type_j < 0) continue;
      orig_j  = atom->tag[j];

      distance(system,atom->x[j],atom->x[i],&nbr_pj->d,&nbr_pj->dvec);
      flag = 0;
      if(nbr_pj->d <= control->nonb_cut) {
	if (j < natoms) flag = 1;
	else if (orig_i < orig_j) flag = 1;
	else if (orig_i == orig_j) {
	  if (nbr_pj->dvec[2] > SMALL) flag = 1;
	  else if (fabs(nbr_pj->dvec[2]) < SMALL) {
	    if (nbr_pj->dvec[1] > SMALL) flag = 1;
	    else if (fabs(nbr_pj->dvec[1]) < SMALL && nbr_pj->dvec[0] > SMALL)
	      flag = 1;
	  }
	}
      }

      if (flag) {
	
	if(strcmp(system->reax_param.sbp[type_i].name,"EL") !=0)type_core=type_i;
	else if(strcmp(system->reax_param.sbp[type_j].name,"EL") !=0)type_core=type_j;
	else type_core=type_i;
	//else type_core=12;
	//if(type_i != 12)type_core=type_i;
	//else if(type_j != 12)type_core=type_j;
	//	else type_core=12;
	r_ij   = nbr_pj->d;
	
	
	Tap = workspace->Tap[7] * r_ij + workspace->Tap[6];
	Tap = Tap * r_ij + workspace->Tap[5];
	Tap = Tap * r_ij + workspace->Tap[4];
	Tap = Tap * r_ij + workspace->Tap[3];
	Tap = Tap * r_ij + workspace->Tap[2];
	Tap = Tap * r_ij + workspace->Tap[1];
	Tap = Tap * r_ij + workspace->Tap[0];
	//Tap=1.0; // tmp!!
	twbp = &(system->reax_param.tbp[ map[atom->type[i]] ]
		 [ map[atom->type[j]] ]);
	
	if(r_ij==0)r_ij=Tiny;
	
	E_elec+=Tap*(EV_to_KCALpMOL*atom->q[i]*atom->q[j]/(r_ij)*erf(twbp->alpha_CGeM*r_ij));
	E_Gauss-=Tap*(twbp->beta_CGeM*exp(-system->reax_param.sbp[type_core].gamma_CGeM*r_ij*r_ij)) ;
	
      }
    }
  }
  //exit(0);
  E_tot =E_elec+E_Gauss;

  return E_tot;
}


double PairReaxC::CGeM_Forces(reax_system *system,control_params *control,
		 simulation_data *data, storage *workspace)
{


  double E_elec;
  double E_Gauss;
  double E_tot;
  double deriv,exp_term,R_1,d_erf;
  int i,j,type_i,type_j;
  two_body_parameters *twbp;
  far_neighbor_data *nbr_pj;
  reax_list *far_nbrs;
  int pj, natoms;
  int start_i, end_i, flag;
  rc_tagint orig_i, orig_j;
  double r_ij,SMALL = 0.0001;
  double Tap,dTap;
  int type_core;
  int i_6, j_6;
  //** electrostatic field
  double deriv_elec;

  double *fx_efield;
  double *fy_efield;
  double *fz_efield;
  get_names("fx_efield",fx_efield); 
  get_names("fy_efield",fy_efield);
  get_names("fz_efield",fz_efield);
  double *E_pot;
  get_names("E_pot",E_pot); 
  
  //**
  type_core=0;
  E_elec =0.0;
  E_Gauss=0.0;
  E_tot=0;
  i=0;

  natoms = system->n;
  
  for( i = 0; i < system->N; ++i ) E_pot[i] = fx_efield[i]=fy_efield[i]=fz_efield[i]=atom->f[i][0]=atom->f[i][1]=atom->f[i][2]=0.0;
  //far_nbrs = (*lists) + FAR_NBRS;
  far_nbrs = lists + FAR_NBRS;
  for( i = 0; i < natoms; ++i ) {
    type_i  = map[atom->type[i]];
    if (type_i < 0) continue;
    start_i = Start_Index(i,far_nbrs);
    end_i   = End_Index(i,far_nbrs);
    orig_i  = atom->tag[i];

    for( pj = start_i; pj < end_i; ++pj ) {
      nbr_pj = &(far_nbrs->select.far_nbr_list[pj]);
      j = nbr_pj->nbr;
      type_j = map[atom->type[j]];
      if (type_j < 0) continue;
      if(strcmp(system->reax_param.sbp[type_i].name,"DU") ==0 && strcmp(system->reax_param.sbp[type_j].name,"DU") ==0) continue;
      orig_j  = atom->tag[j];
      distance(system,atom->x[j],atom->x[i],&nbr_pj->d,&nbr_pj->dvec);
      flag = 0;
      //if(nbr_pj->d <= control->nonb_cut)flag = 1;
      
      if(nbr_pj->d <= control->nonb_cut) {
	if (j < natoms) flag = 1;
	else if (orig_i < orig_j) flag = 1;
	else if (orig_i == orig_j) {
	  if (nbr_pj->dvec[2] > SMALL) flag = 1;
	  else if (fabs(nbr_pj->dvec[2]) < SMALL) {
	    if (nbr_pj->dvec[1] > SMALL) flag = 1;
	    else if (fabs(nbr_pj->dvec[1]) < SMALL && nbr_pj->dvec[0] > SMALL)
	      flag = 1;
	  }
	}
      }
      
      if (flag) {
	if(strcmp(system->reax_param.sbp[type_i].name,"EL") !=0)type_core=type_i;
	else if(strcmp(system->reax_param.sbp[type_j].name,"EL") !=0)type_core=type_j;
	else type_core=type_i;
	
	r_ij   = nbr_pj->d;
	twbp = &(system->reax_param.tbp[ map[atom->type[i]] ]
		 [ map[atom->type[j]] ]);
	
	Tap = workspace->Tap[7] * r_ij + workspace->Tap[6];
	Tap = Tap * r_ij + workspace->Tap[5];
	Tap = Tap * r_ij + workspace->Tap[4];
	Tap = Tap * r_ij + workspace->Tap[3];
	Tap = Tap * r_ij + workspace->Tap[2];
	Tap = Tap * r_ij + workspace->Tap[1];
	Tap = Tap * r_ij + workspace->Tap[0];
	//Tap=1.0;// tmp!!
	dTap = 7*workspace->Tap[7] * r_ij + 6*workspace->Tap[6];
	dTap = dTap * r_ij + 5*workspace->Tap[5];
	dTap = dTap * r_ij + 4*workspace->Tap[4];
	dTap = dTap * r_ij + 3*workspace->Tap[3];
	dTap = dTap * r_ij + 2*workspace->Tap[2];
	dTap = dTap*r_ij + workspace->Tap[1];

	//printf("i=%i type_i=%i j=%i type_j=%i r_ij=%.4f\n",orig_i,type_i,orig_j,type_j,r_ij);
	//printf("alpha=%.4f gamma=%.4f beta=%.4f \n",twbp->alpha_CGeM,system->reax_param.sbp[type_core].gamma_CGeM,twbp->beta_CGeM);
	E_elec=(EV_to_KCALpMOL*atom->q[i]*atom->q[j]/(r_ij)*erf(twbp->alpha_CGeM*r_ij));
	E_Gauss=-(twbp->beta_CGeM*exp(-system->reax_param.sbp[type_core].gamma_CGeM*r_ij*r_ij)) ;
	d_erf= 2.0/sqrt(constPI)*exp(-SQR(twbp->alpha_CGeM*r_ij))*twbp->alpha_CGeM;
	exp_term=exp(-system->reax_param.sbp[type_core].gamma_CGeM*r_ij*r_ij);
	R_1=EV_to_KCALpMOL*atom->q[i]*atom->q[j]/r_ij;
	deriv=-dTap*(E_elec+E_Gauss)+Tap*(EV_to_KCALpMOL*atom->q[i]*atom->q[j]/(r_ij*r_ij)*erf(twbp->alpha_CGeM*r_ij) - R_1*d_erf - (twbp->beta_CGeM)*exp_term*(2.0*system->reax_param.sbp[type_core].gamma_CGeM*r_ij)) ;
	deriv_elec=-dTap*(E_elec)+Tap*(EV_to_KCALpMOL*atom->q[i]*atom->q[j]/(r_ij*r_ij)*erf(twbp->alpha_CGeM*r_ij) - R_1*d_erf) ;
	
	if(strcmp(system->reax_param.sbp[type_i].name,"DU") !=0 && strcmp(system->reax_param.sbp[type_j].name,"DU") !=0)
	  {
	    E_tot += Tap*KCALpMOL_to_EV*(E_elec+E_Gauss);
	    atom->f[i][0]+=KCALpMOL_to_EV*deriv*CGeM_dRij(system,i,j,0,r_ij);
	    atom->f[i][1]+=KCALpMOL_to_EV*deriv*CGeM_dRij(system,i,j,1,r_ij);
	    atom->f[i][2]+=KCALpMOL_to_EV*deriv*CGeM_dRij(system,i,j,2,r_ij);
	    
	    atom->f[j][0]-=KCALpMOL_to_EV*deriv*CGeM_dRij(system,i,j,0,r_ij);
	    atom->f[j][1]-=KCALpMOL_to_EV*deriv*CGeM_dRij(system,i,j,1,r_ij);
	    atom->f[j][2]-=KCALpMOL_to_EV*deriv*CGeM_dRij(system,i,j,2,r_ij);
	    
	    fx_efield[i]+=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,0,r_ij);
	    fy_efield[i]+=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,1,r_ij);
	    fz_efield[i]+=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,2,r_ij);
	    
	    fx_efield[j]-=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,0,r_ij);
	    fy_efield[j]-=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,1,r_ij);
	    fz_efield[j]-=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,2,r_ij);
	    E_pot[i] += Tap*KCALpMOL_to_EV*(E_elec);
	    E_pot[j] += Tap*KCALpMOL_to_EV*(E_elec);
	  }
	else if(strcmp(system->reax_param.sbp[type_i].name,"DU") ==0){
	  fx_efield[i]+=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,0,r_ij);
	  fy_efield[i]+=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,1,r_ij);
	  fz_efield[i]+=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,2,r_ij);
	  E_pot[i] += Tap*KCALpMOL_to_EV*(E_elec);
	}
	else{
	  fx_efield[j]-=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,0,r_ij);
	  fy_efield[j]-=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,1,r_ij);
	  fz_efield[j]-=KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,2,r_ij);
	  E_pot[j] += Tap*KCALpMOL_to_EV*(E_elec);
	}
	//printf("i=%i type_i=%i j=%i type_j=%i r_ij=%.4f\n",orig_i,type_i,orig_j,type_j,r_ij);
	//printf("E_elec=%.4f E_gauss=%.4f deriv=%.4f Tap=%.4f dTap=%.4f\n",E_elec,E_Gauss,deriv,Tap,dTap);
	//std::cout<<********<<std::endl;
	//std::cout<<orig_i<<std::endl;
	//std::cout<<type_i<<std::endl;
	//std::cout<<orig_j<<std::endl;
	//std::cout<<type_j<<std::endl;
	//std::cout<<r_ij<<std::endl;	
	//std::cout<<********<<std::endl;
	//printf("i=%i type_i=%i j=%i type_j=%i r_ij=%.4f\n",orig_i,type_i,orig_j,type_j,r_ij);
	//printf("f_x=%.8f\n",KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,0,r_ij));
	//printf("f_y=%.8f\n",KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,1,r_ij));
	//printf("f_z=%.8f\n",KCALpMOL_to_EV*deriv_elec*CGeM_dRij(system,i,j,2,r_ij));
	//printf("erf=%.4f r_ij=%.4f alpha=%.4f value=%.4f\n",erf(twbp->alpha_CGeM*r_ij),r_ij,twbp->alpha_CGeM,atom->q[i]*atom->q[j]/(r_ij)*erf(twbp->alpha_CGeM*r_ij));
	//printf("Coulomb=%.4f Gauss=%.4f\n",Tap*KCALpMOL_to_EV*E_elec,Tap*KCALpMOL_to_EV*E_Gauss);
	//printf("Tap=%.4f Cal_to_EV=%.4f EV_to_Cal=%.4f\n",Tap,KCALpMOL_to_EV,EV_to_KCALpMOL);
	//printf("type_i=%s\n",system->reax_param.sbp[type_i].name);
	//printf("r_ij=%.8f\n",r_ij);
	
	
      }
    }
  }
  //i=0;
  //while (i < system->N){
  //  printf("i=%i  type=%i fx=%.4f fy=%.4f fz=%.4f\n",i,map[atom->type[i]],atom->f[i][0],atom->f[i][1],atom->f[i][2]);
  //  i++;  
  //}

return(E_tot);
}

void PairReaxC::get_names(char *c,double *&ptr)
{
  int index,flag;
  index = atom->find_custom(c,flag);
  
  if(index!=-1) ptr = atom->dvector[index];
  else error->all(FLERR,"fix iEL-Scf requires fix property/atom ?? command");
}
