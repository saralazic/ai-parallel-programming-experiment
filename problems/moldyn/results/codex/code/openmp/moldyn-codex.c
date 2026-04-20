#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

/*
 *  Function declarations
 */

void dfill(int, double, double[], int);

void domove(int, double[], double[], double[], double);

void dscal(int, double, double[], int);

void fcc(double[], int, int, double);

void forces(int, double[], double[], double, double);

double
mkekin(int, double[], double[], double, double);

void mxwell(double[], int, double, double);

void prnout(int, double, double, double, double, double, double, int, double);

double
velavg(int, double[], double, double);

double
secnds(void);

/*
 *  Variable declarations
 */

double epot;
double vir;
double count;

/*
 *  function dfill : intialises double precision array to scalar value
 */
  void
  dfill(int n, double val, double a[], int ia){
    int i;

    for (i=0; i<(n-1)*ia+1; i+=ia)
      a[i] = val;
  }

/*
 *  Move particles
 */
  void
  domove(int n3, double x[], double vh[], double f[], double side){
    int i;

    #pragma omp parallel for default(none) shared(n3, x, vh, f, side) private(i)
    for (i=0; i<n3; i++) {
      x[i] += vh[i]+f[i];
  /*
   *  Periodic boundary conditions
   */
      if (x[i] < 0.0)  x[i] += side;
      if (x[i] > side) x[i] -= side;
  /*
   *  Partial velocity updates
   */
      vh[i] += f[i];
  /*
   *  Initialise forces for the next iteration
   */
      f[i] = 0.0;
    }
  }

/*
 *  Scales an array
 */
  void
  dscal(int n,double sa,double sx[], int incx){
    int i,j;

    if (incx == 1) {
      #pragma omp parallel for default(none) shared(n, sx, sa) private(i)
      for (i=0; i<n; i++)
        sx[i] *= sa;
    } else {
      #pragma omp parallel for default(none) shared(n, sx, sa, incx) private(i, j)
      for (i=0; i<n; i++) {
        j = i * incx;
        sx[j] *= sa;
      }
    }
  }

/*
 *   Generate fcc lattice for atoms inside the box
 */
  void
  fcc(double x[], int npart, int mm, double a){
    int ijk=0;
    int i,j,k,lg;

    for (lg=0; lg<2; lg++)
      for (i=0; i<mm; i++)
        for (j=0; j<mm; j++)
          for (k=0; k<mm; k++) {
            x[ijk]   = i*a+lg*a*0.5;
            x[ijk+1] = j*a+lg*a*0.5;
            x[ijk+2] = k*a;
            ijk += 3;
          }

    for (lg=1; lg<3; lg++)
      for (i=0; i<mm; i++)
        for (j=0; j<mm; j++)
          for (k=0; k<mm; k++) {
            x[ijk]   = i*a+(2-lg)*a*0.5;
            x[ijk+1] = j*a+(lg-1)*a*0.5;
            x[ijk+2] = k*a+a*0.5;
            ijk += 3;
          }

  }

/*
 *  Compute forces and accumulate the virial and the potential
 */
extern double epot, vir;

void forces(int npart, double x[], double f[], double side, double rcoff)
{
  int i, j;
  int n3;
  double sideh, rcoffs;
  double xi, yi, zi, fxi, fyi, fzi, xx, yy, zz;
  double rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
  double forcex, forcey, forcez;
  double epot_sum, vir_sum;

  vir = 0.0;
  epot = 0.0;
  n3 = npart * 3;
  sideh = 0.5 * side;
  rcoffs = rcoff * rcoff;
  epot_sum = 0.0;
  vir_sum = 0.0;
  dfill(n3, 0.0, f, 1);

  #pragma omp parallel default(none) \
    shared(n3, x, f, side, sideh, rcoffs, stderr) \
    private(i, j, xi, yi, zi, fxi, fyi, fzi, xx, yy, zz, rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148, forcex, forcey, forcez) \
    reduction(+:epot_sum, vir_sum)
  {
    double *f_priv = (double *)calloc((size_t)n3, sizeof(double));
    int k;
    if (f_priv == NULL) {
      fprintf(stderr, "Unable to allocate thread-local force buffer\n");
      exit(EXIT_FAILURE);
    }

    #pragma omp for schedule(dynamic)
    for (i = 0; i < n3; i += 3)
    {
      xi = x[i];
      yi = x[i + 1];
      zi = x[i + 2];
      fxi = 0.0;
      fyi = 0.0;
      fzi = 0.0;

      for (j = i + 3; j < n3; j += 3)
      {
        xx = xi - x[j];
        yy = yi - x[j + 1];
        zz = zi - x[j + 2];
        if (xx < -sideh)
          xx += side;
        if (xx > sideh)
          xx -= side;
        if (yy < -sideh)
          yy += side;
        if (yy > sideh)
          yy -= side;
        if (zz < -sideh)
          zz += side;
        if (zz > sideh)
          zz -= side;
        rd = xx * xx + yy * yy + zz * zz;

        if (rd <= rcoffs)
        {
          rrd = 1.0 / rd;
          rrd2 = rrd * rrd;
          rrd3 = rrd2 * rrd;
          rrd4 = rrd2 * rrd2;
          rrd6 = rrd2 * rrd4;
          rrd7 = rrd6 * rrd;
          epot_sum += (rrd6 - rrd3);
          r148 = rrd7 - 0.5 * rrd4;
          vir_sum -= rd * r148;
          forcex = xx * r148;
          fxi += forcex;
          f_priv[j] -= forcex;
          forcey = yy * r148;
          fyi += forcey;
          f_priv[j + 1] -= forcey;
          forcez = zz * r148;
          fzi += forcez;
          f_priv[j + 2] -= forcez;
        }
      }
      f_priv[i] += fxi;
      f_priv[i + 1] += fyi;
      f_priv[i + 2] += fzi;
    }

    #pragma omp critical
    {
      for (k = 0; k < n3; k++)
        f[k] += f_priv[k];
    }

    free(f_priv);
  }

  epot = epot_sum;
  vir = vir_sum;
}

/*
 *  Main program : Molecular Dynamics simulation.
 */
int main(int arc, char **argv)
{
  int mm = atoi(argv[1]); //15;
  int npart = 4 * mm * mm * mm;
  int move;
  double x[npart * 3], vh[npart * 3], f[npart * 3];
  double ekin;
  double vel;
  double sc;
  double start, time;

  /*
   *  Parameter definitions
   */

  double den = 0.83134;
  double side = pow((double)npart / den, 0.3333333);
  double tref = 0.722;
  double rcoff = (double)mm / 4.0;
  double h = 0.064;
  int irep = 10;
  int istop = 20;
  int iprint = 5;
  int movemx = 20;

  double a = side / (double)mm;
  double hsq = h * h;
  double hsq2 = hsq * 0.5;
  double tscale = 16.0 / ((double)npart - 1.0);
  double vaver = 1.13 * sqrt(tref / 24.0);

  /*
   *  Initial output
   */

  printf(" Molecular Dynamics Simulation example program\n");
  printf(" ---------------------------------------------\n");
  printf(" number of particles is ............ %6d\n", npart);
  printf(" side length of the box is ......... %13.6f\n", side);
  printf(" cut off is ........................ %13.6f\n", rcoff);
  printf(" reduced temperature is ............ %13.6f\n", tref);
  printf(" basic timestep is ................. %13.6f\n", h);
  printf(" temperature scale interval ........ %6d\n", irep);
  printf(" stop scaling at move .............. %6d\n", istop);
  printf(" print interval .................... %6d\n", iprint);
  printf(" total no. of steps ................ %6d\n", movemx);

  /*
   *  Generate fcc lattice for atoms inside box
   */
  fcc(x, npart, mm, a);
  /*
   *  Initialise velocities and forces (which are zero in fcc positions)
   */
  mxwell(vh, 3 * npart, h, tref);
  dfill(3 * npart, 0.0, f, 1);
  /*
   *  Start of md
   */
  printf("\n    i       ke         pe            e         temp   "
         "   pres      vel      rp\n  -----  ----------  ----------"
         "  ----------  --------  --------  --------  ----\n");

  start = secnds();

  for (move = 1; move <= movemx; move++)
  {

    /*
     *  Move the particles and partially update velocities
     */
    domove(3 * npart, x, vh, f, side);

    /*
     *  Compute forces in the new positions and accumulate the virial
     *  and potential energy.
     */
    forces(npart, x, f, side, rcoff);

    /*
     *  Scale forces, complete update of velocities and compute k.e.
     */
    ekin = mkekin(npart, f, vh, hsq2, hsq);

    /*
     *  Average the velocity and temperature scale if desired
     */
    vel = velavg(npart, vh, vaver, h);
    if (move < istop && fmod(move, irep) == 0)
    {
      sc = sqrt(tref / (tscale * ekin));
      dscal(3 * npart, sc, vh, 1);
      ekin = tref / tscale;
    }

    /*
     *  Sum to get full potential energy and virial
     */
    if (fmod(move, iprint) == 0)
      prnout(move, ekin, epot, tscale, vir, vel, count, npart, den);
  }

  time = secnds() - start;

  printf("Elapsed time =  %f\n", (float)time);
}

time_t starttime = 0;

double secnds()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#include <stdio.h>
/*
 *  Scale forces, update velocities and compute K.E.
 */
  double
  mkekin(int npart, double f[], double vh[], double hsq2, double hsq){
    int i;
    double sum=0.0, ekin;

    #pragma omp parallel for default(none) shared(npart, f, vh, hsq2) private(i) reduction(+:sum)
    for (i=0; i<3*npart; i++) {
      f[i]*=hsq2;
      vh[i]+=f[i];
      sum+=vh[i]*vh[i];
    }
    ekin=sum/hsq;

    return(ekin);
  }
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

  void srand48(long);
  double drand48(void);
/*
 *  Sample Maxwell distribution at temperature tref
 */
  void
  mxwell(double vh[], int n3, double h, double tref){
    int i;
    int npart=n3/3;
    double r, tscale, v1, v2, s, ekin=0.0, sp=0.0, sc;
    
    srand48(4711);
    tscale=16.0/((double)npart-1.0);

    for (i=0; i<n3; i+=2) {
      s=2.0;
      while (s>=1.0) {
        v1=2.0*drand48()-1.0;
        v2=2.0*drand48()-1.0;
        s=v1*v1+v2*v2;
      }
      r=sqrt(-2.0*log(s)/s);
      vh[i]=v1*r;
      vh[i+1]=v2*r;
    }

    for (i=0; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;
    for(i=0; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sp=0.0;
    for (i=1; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;
    for(i=1; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sp=0.0;
    for (i=2; i<n3; i+=3) sp+=vh[i];
    sp/=(double)npart;
    for(i=2; i<n3; i+=3) {
      vh[i]-=sp;
      ekin+=vh[i]*vh[i];
    }

    sc=h*sqrt(tref/(tscale*ekin));
    for (i=0; i<n3; i++) vh[i]*=sc;
  }

/*
 *   Print out interesting information at current timestep
 */
  void
  prnout(int move, double ekin, double epot, double tscale, double vir,
         double vel, double count, int npart, double den){
    double ek, etot, temp, pres, rp;

    ek=24.0*ekin;
    epot*=4.0;
    etot=ek+epot;
    temp=tscale*ekin;
    pres=den*16.0*(ekin-vir)/(double)npart;
    vel/=(double)npart;
    rp=(count/(double)npart)*100.0;
    printf(" %6d%12.4f%12.4f%12.4f%10.4f%10.4f%10.4f%6.1f\n",
           move,ek,epot,etot,temp,pres,vel,rp);

  }

/*
 *  Compute average velocity
 */
  double
  velavg(int npart, double vh[], double vaver, double h){
    int i;
    double vaverh=vaver*h;
    double vel=0.0;
    double sq;
    extern double count;

    count=0.0;
    #pragma omp parallel for default(none) shared(npart, vh, vaverh) private(i, sq) reduction(+:count, vel)
    for (i=0; i<npart*3; i+=3){
      sq=sqrt(vh[i]*vh[i]+vh[i+1]*vh[i+1]+vh[i+2]*vh[i+2]);
      if (sq>vaverh) count++;
      vel+=sq;
    }
    vel/=h;

    return(vel);
  }