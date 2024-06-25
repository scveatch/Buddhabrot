# CUDA Buddhabrot Generator

This repository contains a CUDA implementation for generating the Buddhabrot fractal. The Buddhabrot fractal is a variation of the Mandelbrot set, displaying the probability distribution of escape trajectories of points in the complex plane iterated under a given mapping. Mathematically, the Mandlebrot set is represented by the set of points $c$ in the complex plane for which the following sequence does not tend to infinity as $n$ goes to infinity for

$$z_{n+1} = z_n^2 + c$$

## Introduction

The Buddhabrot is an extension of the Mandelbrot, iterating the function $z_{n+1} = z_n^2 + c$ where $c$ is each point in the image plane. However, instead of recording the behavior of the series at each point $c$, we only consider points which escape to infinity and create a density plot of terms in the series. The result is a 2D plot of escaping points. The point $c$, in this implementation, is taken from the region $(-2 - 2i)$ to $(2 + 2i)$. Pixels are colored based on how many iterations were necessary before the $abs(z_{n}) > 2$ escape condition was encountered. This is an intensely iterative process for many random starting points, requiring computational efficiency. This implementation utilizes CUDA (Compute Unified Device Architecture) to parallelize the computation, significantly speeding up the generation process. There are many possible extensions of this code which I leave to you. 

## Prerequisites

To compile and run the CUDA fractal generator, you need:

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed on your system
- NVIDIA CUDA Compiler (nvcc)

## Setting Up CUDA Environment

1. **Install CUDA Toolkit**: Download and install the CUDA Toolkit from the [NVIDIA Developer website](https://developer.nvidia.com/cuda-downloads).

2. **Verify Installation**: After installation, verify that CUDA is correctly installed by running `nvcc --version` in your terminal or command prompt. This should display the version of the NVIDIA CUDA Compiler.

## Compiling and Running

To compile and run the CUDA fractal generator:

1. Navigate to the project directory containing the CUDA source code.

2. Compile the code using nvcc:
    ```bash
    nvcc brot.cu -o fractal
    ```

3. Run the compiled executable:
    ```bash
    ./fractal
    ```

## About CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model developed by NVIDIA. It enables developers to harness the computational power of NVIDIA GPUs for general-purpose processing, including tasks such as scientific simulations, image processing, and deep learning.

## About nvcc

nvcc is the NVIDIA CUDA Compiler, which translates CUDA C/C++ code into executable binaries that can run on NVIDIA GPUs. It integrates CUDA code compilation with standard C/C++ compilation and linking processes, providing developers with a seamless development experience for GPU-accelerated applications.

For more information about CUDA and nvcc, refer to the [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/).

## Usage

The `cuda_buddhabrot` executable generates the Buddhabrot fractal with customizable parameters such as size and iterations. By default, it generates a Buddhabrot image with a size of 15000x15000 pixels and 10000 iterations.

## Contributors

- [Spencer Veatch](https://github.com/scveatch]) - *Initial work*
