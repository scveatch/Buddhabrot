#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define GAIN 5
#define CUTOFF 1

// Declare Complex Structure:
struct Complex {
    double real;
    double imag;
};

// Device function to add two complex numbers
__device__ struct Complex addComplex(struct Complex a, struct Complex b) {
    struct Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

// Create base on the GPU
__global__ void create_base(unsigned char *d_base, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (i * size + j) * 3;

    if (i < size && j < size) {
        d_base[idx] = 0; // Red
        d_base[idx + 1] = 0; // Green
        d_base[idx + 2] = 0; // Blue
    }
}

// Device function to compute z_n^2 + c
__device__ struct Complex m_seq(struct Complex z_n, struct Complex c){
    struct Complex result;
    // Corrected the calculation to compute z_n^2
    result.real = (z_n.real * z_n.real) - (z_n.imag * z_n.imag);
    result.imag = (2 * (z_n.real * z_n.imag));
    struct Complex x = addComplex(result, c);
    // printf("%f + %fi\n", x.real, x.imag); // Confirm matches m_seq in brot.c
    return x;
}

// Complex to base; min and max come predefined in cuda. 
__device__ void c2b(struct Complex c, int size, int *x, int *y){
    *x = (int) ((c.real + 2) * size) / 4; 
    *y = (int) ((c.imag + 2) * size) / 4;
    *x = min(*x, size - 1);
    *y = min(*y, size - 1);
    *x = max(*x, 0);
    *y = max(*y, 0);
    return;
}

// Base to complex -- create complex "out", assign real and
// imaginary components and return complex out.
__device__ struct Complex b2c(int size, int x, int y){
    struct Complex out;
    out.real =  x * 4.0 / size - 2.0;
	out.imag =  y * 4.0 / size - 2.0;
    return out;
}

// Determine if a complex value c escapes within iters
// iterations.
__device__ int escapes(struct Complex c, int iters){
    struct Complex z_n = c; 
    for(int i = 0; i < iters; i++){
        z_n = m_seq(z_n, c);
        if(sqrt(z_n.real * z_n.real + z_n.imag * z_n.imag) > 2){
            return 1;
        }
    }
    return 0;
}

__device__ void one_val(unsigned char *d_base, int size, int iters, int color, struct Complex c) {
    struct Complex z_n = c;
    int x, y;

    // Check if value escapes within iters
    if (escapes(c, iters) == 0) {
        return;
    }
    
    for (int i = 0; i < iters; i++) {
        // Escape condition
        if (sqrt(z_n.real * z_n.real + z_n.imag * z_n.imag) > 2) {
            return;
        }
        
        c2b(z_n, size, &x, &y);
        x = (x < size - 1) ? x : size - 1;
        y = (y < size - 1) ? y : size - 1;
        
        int idx = ((x * size) + y) * 3;
        int v = d_base[idx + color] + 15;
        d_base[idx + color] = (v > 255) ? 255 : v;
        
        z_n = m_seq(z_n, c);
    }
}

__global__ void get_colors(unsigned char *d_base, int size, int iters) {
    int ilist[3] = {iters * 100, iters * 10, iters};

    // Iterate and run one_val
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < size; j += blockDim.y * gridDim.y) {
            for (int color = 0; color < 3; color++) {
                struct Complex c = b2c(size, i, j);
                one_val(d_base, size, ilist[color], color, c);
            }
        }
    }
}

// Sigmoid correction
__device__ unsigned char sigmoid_correction(unsigned char pixel, double gain, double cut){
    double scaled = (double) pixel / 255.0;
    double corrected = 1.0 / (1.0 + exp(gain * (cut - scaled)));
    return (unsigned char) (corrected * 255);
}

// Run sigmoid on all pixels
__global__ void equalize(unsigned char *d_base, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size) {
        for (int k = 0; k < 3; k++) {
            int idx = ((i * size) + j) * 3 + k;
            d_base[idx] = sigmoid_correction(d_base[idx], GAIN, CUTOFF);
        }
    }
}

// Function to write the output to a PPM file
void write_ppm(unsigned char *base, int size) {
    FILE *fp = fopen("cudabrot.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", size, size);
    fwrite(base, sizeof(unsigned char), size * size * 3, fp);
    fclose(fp);
}

void make_brot(int size, int iters){
    unsigned char *d_base;
    cudaMalloc((void **)&d_base, size * size * 3 * sizeof(unsigned char));

    dim3 blockSize(32, 32);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);

    // Initialize base on GPU
    create_base<<<gridSize, blockSize>>>(d_base, size);
    cudaDeviceSynchronize();

    // Run kernel to compute Buddhabrot set
    get_colors<<<gridSize, blockSize>>>(d_base, size, iters);

    // Perform sigmoid correction
    equalize<<<gridSize, blockSize>>>(d_base, size);

    // Copy data from device to host
    unsigned char *base = (unsigned char *)malloc(size * size * 3 * sizeof(unsigned char));
    cudaMemcpy(base, d_base, size * size * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Write base to ppm
    write_ppm(base, size);

    // Free everything
    cudaFree(d_base);
    free(base);
}

int main(){
    // Launch kernel
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    make_brot(15000, 10000);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Elapsed Time: %.3f seconds\n", elapsedTime / 1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}