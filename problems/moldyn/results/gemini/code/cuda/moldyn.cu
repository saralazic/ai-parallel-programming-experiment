#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// --- CUDA Block Reduction Helper ---
__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ double blockReduceSum(double val) {
    __shared__ double shared[32]; // Maximum 1024 threads -> 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;

    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val);

    return val;
}

/*
 *  Kernel 1: Move particles
 */
__global__ void domove_kernel(int n3, double* x, double* vh, double* f, double side) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n3) {
        x[i] += vh[i] + f[i];
        if (x[i] < 0.0)  x[i] += side;
        if (x[i] > side) x[i] -= side;
        vh[i] += f[i];
        f[i] = 0.0; // Reset forces for next step
    }
}

/*
 *  Kernel 2: Compute forces and accumulate potential energy and virial
 */
__global__ void forces_kernel(int npart, double* x, double* f, double side, double rcoff, double* epot_out, double* vir_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double epot_local = 0.0;
    double vir_local = 0.0;

    if (i < npart) {
        double sideh = 0.5 * side;
        double rcoffs = rcoff * rcoff;
        
        int idx = i * 3;
        double xi = x[idx];
        double yi = x[idx + 1];
        double zi = x[idx + 2];
        double fxi = 0.0;
        double fyi = 0.0;
        double fzi = 0.0;

        for (int j = 0; j < npart; j++) {
            if (i == j) continue;
            int jdx = j * 3;
            double xx = xi - x[jdx];
            double yy = yi - x[jdx + 1];
            double zz = zi - x[jdx + 2];

            if (xx < -sideh) xx += side;
            if (xx > sideh) xx -= side;
            if (yy < -sideh) yy += side;
            if (yy > sideh) yy -= side;
            if (zz < -sideh) zz += side;
            if (zz > sideh) zz -= side;

            double rd = xx * xx + yy * yy + zz * zz;

            if (rd <= rcoffs) {
                double rrd = 1.0 / rd;
                double rrd2 = rrd * rrd;
                double rrd3 = rrd2 * rrd;
                double rrd4 = rrd2 * rrd2;
                double rrd6 = rrd2 * rrd4;
                double rrd7 = rrd6 * rrd;
                epot_local += (rrd6 - rrd3);
                double r148 = rrd7 - 0.5 * rrd4;
                vir_local -= rd * r148;
                fxi += xx * r148;
                fyi += yy * r148;
                fzi += zz * r148;
            }
        }
        f[idx] += fxi;
        f[idx + 1] += fyi;
        f[idx + 2] += fzi;
    }

    double block_epot = blockReduceSum(epot_local);
    double block_vir  = blockReduceSum(vir_local);

    if (threadIdx.x == 0) {
        atomicAdd(epot_out, block_epot);
        atomicAdd(vir_out, block_vir);
    }
}

/*
 *  Kernel 3: Scale forces, update velocities and compute kinetic energy
 */
__global__ void mkekin_kernel(int npart, double* f, double* vh, double hsq2, double* ekin_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double sum_local = 0.0;

    if (i < 3 * npart) {
        f[i] *= hsq2;
        vh[i] += f[i];
        sum_local = vh[i] * vh[i];
    }

    double block_sum = blockReduceSum(sum_local);
    
    if (threadIdx.x == 0) {
        atomicAdd(ekin_out, block_sum);
    }
}

/*
 *  Kernel 4: Compute average velocity
 */
__global__ void velavg_kernel(int npart, double* vh, double vaverh, double* vel_out, double* count_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double vel_local = 0.0;
    double count_local = 0.0;

    if (i < npart) {
        int idx = i * 3;
        double sq = sqrt(vh[idx]*vh[idx] + vh[idx+1]*vh[idx+1] + vh[idx+2]*vh[idx+2]);
        if (sq > vaverh) count_local = 1.0;
        vel_local = sq;
    }

    double block_vel = blockReduceSum(vel_local);
    double block_count = blockReduceSum(count_local);

    if (threadIdx.x == 0) {
        atomicAdd(vel_out, block_vel);
        atomicAdd(count_out, block_count);
    }
}

/*
 *  Kernel 5: Scale an array
 */
__global__ void dscal_kernel(int n, double sa, double* sx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sx[i] *= sa;
    }
}

// ==========================================
// CPU Initialization and Utility Functions
// ==========================================

void dfill(int n, double val, double a[], int ia){
    int i;
    for (i=0; i<(n-1)*ia+1; i+=ia)
      a[i] = val;
}

void fcc(double x[], int npart, int mm, double a){
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

void mxwell(double vh[], int n3, double h, double tref){
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

void prnout(int move, double ekin, double epot, double tscale, double vir,
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

double secnds(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ==========================================
// Main Function
// ==========================================

int main(int arc, char **argv)
{
    if (arc != 2) {
        printf("Usage: %s <mm>\n", argv[0]);
        return 1;
    }
    
    int mm = atoi(argv[1]);
    int npart = 4 * mm * mm * mm;
    int move;
    
    // Allocate host memory (Standard C++ compatible instead of VLA)
    double* x = (double*)malloc(npart * 3 * sizeof(double));
    double* vh = (double*)malloc(npart * 3 * sizeof(double));
    double* f = (double*)malloc(npart * 3 * sizeof(double));
    
    double ekin;
    double vel;
    double epot;
    double vir;
    double count;
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
    double vaverh = vaver * h;

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

    // ==========================================
    // CUDA Memory Management
    // ==========================================
    double *d_x, *d_vh, *d_f;
    cudaMalloc(&d_x, 3 * npart * sizeof(double));
    cudaMalloc(&d_vh, 3 * npart * sizeof(double));
    cudaMalloc(&d_f, 3 * npart * sizeof(double));

    // Copy initial data from host to device
    cudaMemcpy(d_x, x, 3 * npart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vh, vh, 3 * npart * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, 3 * npart * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate device memory for global reduction variables
    double *d_epot, *d_vir, *d_ekin, *d_vel, *d_count;
    cudaMalloc(&d_epot, sizeof(double));
    cudaMalloc(&d_vir, sizeof(double));
    cudaMalloc(&d_ekin, sizeof(double));
    cudaMalloc(&d_vel, sizeof(double));
    cudaMalloc(&d_count, sizeof(double));

    // Configure block and grid dimensions
    int threads = 256;
    int blocks_n = (npart + threads - 1) / threads;
    int blocks_3n = (3 * npart + threads - 1) / threads;

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
        domove_kernel<<<blocks_3n, threads>>>(3 * npart, d_x, d_vh, d_f, side);

        /*
         *  Compute forces in the new positions and accumulate the virial
         *  and potential energy.
         */
        cudaMemset(d_epot, 0, sizeof(double));
        cudaMemset(d_vir, 0, sizeof(double));
        forces_kernel<<<blocks_n, threads>>>(npart, d_x, d_f, side, rcoff, d_epot, d_vir);

        /*
         *  Scale forces, complete update of velocities and compute k.e.
         */
        cudaMemset(d_ekin, 0, sizeof(double));
        mkekin_kernel<<<blocks_3n, threads>>>(npart, d_f, d_vh, hsq2, d_ekin);

        // Fetch kinetic energy back to CPU to compute potential temperature scaling
        double h_ekin;
        cudaMemcpy(&h_ekin, d_ekin, sizeof(double), cudaMemcpyDeviceToHost);
        ekin = h_ekin / hsq;

        /*
         *  Average the velocity and temperature scale if desired
         */
        cudaMemset(d_vel, 0, sizeof(double));
        cudaMemset(d_count, 0, sizeof(double));
        velavg_kernel<<<blocks_n, threads>>>(npart, d_vh, vaverh, d_vel, d_count);

        if (move < istop && fmod(move, irep) == 0)
        {
            sc = sqrt(tref / (tscale * ekin));
            dscal_kernel<<<blocks_3n, threads>>>(3 * npart, sc, d_vh);
            ekin = tref / tscale;
        }

        /*
         *  Sum to get full potential energy and virial (and print)
         */
        if (fmod(move, iprint) == 0)
        {
            double h_epot, h_vir, h_vel, h_count;
            cudaMemcpy(&h_epot, d_epot, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_vir, d_vir, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_vel, d_vel, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_count, d_count, sizeof(double), cudaMemcpyDeviceToHost);
            
            // Adjust scaling factors as per original code
            epot = h_epot * 0.5;
            vir = h_vir * 0.5;
            vel = h_vel / h;
            count = h_count;

            prnout(move, ekin, epot, tscale, vir, vel, count, npart, den);
        }
    }

    // Wait for all GPU operations to finish
    cudaDeviceSynchronize();

    time = secnds() - start;

    printf("Elapsed time =  %f\n", (float)time);

    // Free GPU Memory
    cudaFree(d_x);
    cudaFree(d_vh);
    cudaFree(d_f);
    cudaFree(d_epot);
    cudaFree(d_vir);
    cudaFree(d_ekin);
    cudaFree(d_vel);
    cudaFree(d_count);
    
    // Free Host Memory
    free(x);
    free(vh);
    free(f);

    return 0;
}
