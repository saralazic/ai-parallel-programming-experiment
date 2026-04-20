#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

#define BLOCK_SIZE 16
#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD (3.0e6)
/* required precision in degrees    */
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor   */
#define FACTOR_CHIP 0.5

typedef float FLOAT;

/* chip parameters  */
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;
const FLOAT amb_temp = 80.0;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void calculate_temp(FLOAT *power,
                               FLOAT *temp_src,
                               FLOAT *temp_dst,
                               int grid_cols,
                               int grid_rows,
                               FLOAT Cap_1,
                               FLOAT Rx_1,
                               FLOAT Ry_1,
                               FLOAT Rz_1,
                               FLOAT amb_temp) {
    // Allocate shared memory with a halo around the block for neighborhood values
    __shared__ FLOAT temp_on_cuda[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ FLOAT power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];

    FLOAT temp_t, temp_b, temp_l, temp_r, temp_c, power_c;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and col
    int c = bx * BLOCK_SIZE + tx;
    int r = by * BLOCK_SIZE + ty;

    int index = r * grid_cols + c;

    // 1. Load data from global memory to shared memory
    // Each thread loads its own target cell
    if (r < grid_rows && c < grid_cols) {
        temp_on_cuda[ty + 1][tx + 1] = temp_src[index];
        power_on_cuda[ty][tx] = power[index];
    }

    // 2. Load the halo cells (the 1-element border around the block)
    if (ty == 0) { // top halo
        if (r > 0 && c < grid_cols) temp_on_cuda[0][tx + 1] = temp_src[(r - 1) * grid_cols + c];
    }
    if (ty == BLOCK_SIZE - 1) { // bottom halo
        if (r < grid_rows - 1 && c < grid_cols) temp_on_cuda[BLOCK_SIZE + 1][tx + 1] = temp_src[(r + 1) * grid_cols + c];
    }
    if (tx == 0) { // left halo
        if (c > 0 && r < grid_rows) temp_on_cuda[ty + 1][0] = temp_src[r * grid_cols + (c - 1)];
    }
    if (tx == BLOCK_SIZE - 1) { // right halo
        if (c < grid_cols - 1 && r < grid_rows) temp_on_cuda[ty + 1][BLOCK_SIZE + 1] = temp_src[r * grid_cols + (c + 1)];
    }

    // Synchronize to ensure all shared memory is populated
    __syncthreads();

    // Threads outside the active grid limits don't compute results
    if (r >= grid_rows || c >= grid_cols) return;

    // Read values from shared memory
    temp_c = temp_on_cuda[ty + 1][tx + 1];
    power_c = power_on_cuda[ty][tx];

    // Determine neighbor temperatures with boundary conditions
    // If the neighbor is out of bounds, use the center cell's temperature (which neutralizes the gradient)
    temp_t = (r == 0) ? temp_c : temp_on_cuda[ty][tx + 1];
    temp_b = (r == grid_rows - 1) ? temp_c : temp_on_cuda[ty + 2][tx + 1];
    temp_l = (c == 0) ? temp_c : temp_on_cuda[ty + 1][tx];
    temp_r = (c == grid_cols - 1) ? temp_c : temp_on_cuda[ty + 1][tx + 2];

    // Main finite difference computation
    FLOAT delta = Cap_1 * (power_c +
        (temp_b + temp_t - 2.0f * temp_c) * Ry_1 +
        (temp_r + temp_l - 2.0f * temp_c) * Rx_1 +
        (amb_temp - temp_c) * Rz_1);

    temp_dst[index] = temp_c + delta;
}

void compute_tran_temp_cuda(FLOAT *result, int num_iterations, FLOAT *temp,
                            FLOAT *power, int row, int col) {
    FLOAT grid_height = chip_height / row;
    FLOAT grid_width = chip_width / col;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1 = 1.f / Rx;
    FLOAT Ry_1 = 1.f / Ry;
    FLOAT Rz_1 = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;

    FLOAT *d_temp_src, *d_temp_dst, *d_power;
    size_t size = row * col * sizeof(FLOAT);
    
    // Allocate device memory (global memory)
    CUDA_CHECK(cudaMalloc((void**)&d_temp_src, size));
    CUDA_CHECK(cudaMalloc((void**)&d_temp_dst, size));
    CUDA_CHECK(cudaMalloc((void**)&d_power, size));

    // Transfer initial data to device
    CUDA_CHECK(cudaMemcpy(d_temp_src, temp, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp_dst, result, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_power, power, size, cudaMemcpyHostToDevice));

    // Configure block and grid dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((col + block.x - 1) / block.x, (row + block.y - 1) / block.y);

    for (int i = 0; i < num_iterations; i++) {
        // Launch kernel
        calculate_temp<<<grid, block>>>(d_power, d_temp_src, d_temp_dst,
                                        col, row,
                                        Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
        
        // Ping-pong pointers to avoid memory copies between time steps
        FLOAT *tmp = d_temp_src;
        d_temp_src = d_temp_dst;
        d_temp_dst = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Replicate sequential code pointer behavior exactly to match expected outputs
    if (num_iterations % 2 == 1) {
        // odd num_iterations: latest data was written to result array initially
        CUDA_CHECK(cudaMemcpy(result, d_temp_src, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(temp, d_temp_dst, size, cudaMemcpyDeviceToHost));
    } else {
        // even num_iterations: latest data was written to temp array initially
        CUDA_CHECK(cudaMemcpy(temp, d_temp_src, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result, d_temp_dst, size, cudaMemcpyDeviceToHost));
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_temp_src));
    CUDA_CHECK(cudaFree(d_temp_dst));
    CUDA_CHECK(cudaFree(d_power));
}

void fatal(const char *s) {
    fprintf(stderr, "error: %s\n", s);
    exit(1);
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
        if(fgets(str, STR_SIZE, fp) == NULL)
            fatal("not enough lines in file");
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1))
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv) {
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads> <temp_file> <power_file> <output_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<no. of threads>   - number of threads (ignored in CUDA version)\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char **argv) {
    int grid_rows, grid_cols, sim_time, i;
    FLOAT *temp, *power, *result;
    char *tfile, *pfile;

    /* check validity of inputs */
    if (argc != 8)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0 ||
        (sim_time = atoi(argv[3])) <= 0 || atoi(argv[4]) <= 0)
        usage(argc, argv);

    /* allocate memory for the temperature and power arrays */
    temp = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    power = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    result = (FLOAT *)calloc(grid_rows * grid_cols, sizeof(FLOAT));
    if (!temp || !power || !result)
        fatal("unable to allocate memory");

    /* read initial temperatures and input power    */
    tfile = argv[5];
    pfile = argv[6];

    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    printf("Start computing the transient temperature (CUDA)\n");
    long long start_time = get_time();
    
    compute_tran_temp_cuda(result, sim_time, temp, power, grid_rows, grid_cols);

    long long end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %g seconds\n", (end_time - start_time) / 1e6);

    /* output results   */
    if (getenv("OUTPUT")) {
        FILE *file = fopen("output.txt", "w+");
        if (file) {
            for (i = 0; i < grid_rows * grid_cols; i++)
                fprintf(file, "%d\t%g\n", i, temp[i]);
            fclose(file);
        }
    }

    /* cleanup  */
    free(temp);
    free(power);
    free(result);

    return 0;
}