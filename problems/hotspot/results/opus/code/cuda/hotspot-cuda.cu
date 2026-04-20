#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE 256

#define MAX_PD (3.0e6)
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP 0.5

typedef float FLOAT;

const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;
const FLOAT amb_temp = 80.0;

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

/* ---------------------------------------------------------------------------
 * CUDA kernel: one thread per grid cell, shared memory tile with 1-cell halo.
 *
 * Boundary conditions use index clamping: a missing neighbor is replaced by
 * the cell's own value, which gives the same Neumann (zero-gradient) BC as
 * the sequential code's corner/edge branches.
 * ---------------------------------------------------------------------------*/
__global__ void hotspot_kernel(FLOAT *result, FLOAT *temp, FLOAT *power,
                               int row, int col,
                               FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1,
                               FLOAT Rz_1, FLOAT amb_temp_val) {

    __shared__ FLOAT temp_s[BLOCK_SIZE_R + 2][BLOCK_SIZE_C + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int c  = blockIdx.x * BLOCK_SIZE_C + tx;
    int r  = blockIdx.y * BLOCK_SIZE_R + ty;

    /* Shared-memory indices (offset by 1 for the halo ring) */
    int sx = tx + 1;
    int sy = ty + 1;

    /*
     * Clamp global coordinates so that out-of-bounds threads (when the grid
     * dimensions aren't a multiple of BLOCK_SIZE) still load valid data.
     * This avoids divergent early-returns before __syncthreads().
     */
    int cr = min(r, row - 1);
    int cc = min(c, col - 1);

    /* --- Load the tile center ------------------------------------------- */
    temp_s[sy][sx] = temp[cr * col + cc];

    /* --- Load the 1-cell halo ring from global memory ------------------- */
    if (ty == 0)
        temp_s[0][sx] = temp[max(cr - 1, 0) * col + cc];

    if (ty == BLOCK_SIZE_R - 1)
        temp_s[BLOCK_SIZE_R + 1][sx] = temp[min(cr + 1, row - 1) * col + cc];

    if (tx == 0)
        temp_s[sy][0] = temp[cr * col + max(cc - 1, 0)];

    if (tx == BLOCK_SIZE_C - 1)
        temp_s[sy][BLOCK_SIZE_C + 1] = temp[cr * col + min(cc + 1, col - 1)];

    __syncthreads();

    /* --- Compute the 5-point stencil from shared memory ----------------- */
    if (r < row && c < col) {
        FLOAT center = temp_s[sy][sx];
        result[r * col + c] =
            center +
            Cap_1 * (power[r * col + c] +
                     (temp_s[sy + 1][sx] + temp_s[sy - 1][sx] -
                      2.f * center) * Ry_1 +
                     (temp_s[sy][sx + 1] + temp_s[sy][sx - 1] -
                      2.f * center) * Rx_1 +
                     (amb_temp_val - center) * Rz_1);
    }
}

/* ---------------------------------------------------------------------------
 * Host driver: computes physics constants, transfers data to GPU, runs the
 * iterative kernel loop, and copies results back.
 * ---------------------------------------------------------------------------*/
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp,
                       FLOAT *power, int row, int col) {

    FLOAT grid_height = chip_height / row;
    FLOAT grid_width  = chip_width  / col;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx  = grid_width  / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry  = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz  = t_chip      / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step      = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1  = 1.f / Rx;
    FLOAT Ry_1  = 1.f / Ry;
    FLOAT Rz_1  = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;

    /* ---- Device memory allocation -------------------------------------- */
    size_t size = (size_t)row * col * sizeof(FLOAT);
    FLOAT *d_temp, *d_result, *d_power;

    CUDA_CHECK(cudaMalloc(&d_temp,   size));
    CUDA_CHECK(cudaMalloc(&d_result, size));
    CUDA_CHECK(cudaMalloc(&d_power,  size));

    /* ---- Host → Device transfers --------------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_temp,  temp,  size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_power, power, size, cudaMemcpyHostToDevice));

    /* ---- Kernel launch configuration ----------------------------------- */
    dim3 blockDim(BLOCK_SIZE_C, BLOCK_SIZE_R);
    dim3 gridDim((col + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C,
                 (row + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R);

    /* ---- Iterative solver: one kernel launch per time step ------------- */
    FLOAT *d_src = d_temp;
    FLOAT *d_dst = d_result;

    for (int i = 0; i < num_iterations; i++) {
        hotspot_kernel<<<gridDim, blockDim>>>(
            d_dst, d_src, d_power,
            row, col, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);

        FLOAT *tmp = d_src;
        d_src = d_dst;
        d_dst = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- Device → Host transfer ---------------------------------------- */
    /* After the loop d_src points to the buffer that received the last write */
    CUDA_CHECK(cudaMemcpy(result, d_src, size, cudaMemcpyDeviceToHost));
    memcpy(temp, result, size);

    /* ---- Free device memory -------------------------------------------- */
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_power));
}

/* ---------------------------------------------------------------------------
 * Utility / I/O — identical to the sequential version
 * ---------------------------------------------------------------------------*/
void fatal(const char *s) {
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

void writeoutput(FLOAT *vect, int grid_rows, int grid_cols, char *file) {
    int i, j, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }

    fclose(fp);
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file) {
    int i;
    FILE *fp;
    char str[STR_SIZE];
    FLOAT val;

    fp = fopen(file, "r");
    if (!fp)
        fatal("file could not be opened for reading");

    for (i = 0; i < grid_rows * grid_cols; i++) {
        fgets(str, STR_SIZE, fp);
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1))
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv) {
    fprintf(stderr,
            "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads>"
            " <temp_file> <power_file> <output_file>\n",
            argv[0]);
    fprintf(stderr,
            "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(stderr,
            "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<no. of threads>   - number of threads (ignored)\n");
    fprintf(stderr,
            "\t<temp_file>  - name of the file containing the initial "
            "temperature values of each cell\n");
    fprintf(stderr,
            "\t<power_file> - name of the file containing the dissipated "
            "power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char **argv) {
    int grid_rows, grid_cols, sim_time, i;
    FLOAT *temp, *power, *result;
    char *tfile, *pfile, *ofile;

    if (argc != 8)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 ||
        (sim_time = atoi(argv[3])) <= 0 || atoi(argv[4]) <= 0)
        usage(argc, argv);

    temp   = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    power  = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    result = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    if (!temp || !power || !result)
        fatal("unable to allocate memory");

    tfile = argv[5];
    pfile = argv[6];
    ofile = argv[7];

    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    printf("Start computing the transient temperature\n");
    long long start_time = get_time();

    compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);

    long long end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %g seconds\n", (end_time - start_time) / 1e6);

    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "w+");
        for (i = 0; i < grid_rows * grid_cols; i++)
            fprintf(file, "%d\t%g\n", i, temp[i]);
        fclose(file);
    }

    free(temp);
    free(power);
    free(result);

    return 0;
}
