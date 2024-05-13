# CUDA Buddhabrot Generator

This repository contains a CUDA implementation for generating the Buddhabrot fractal. The Buddhabrot fractal is a variation of the Mandelbrot set, displaying the probability distribution of escape trajectories of points in the complex plane iterated under a given mapping.

## Introduction

The Buddhabrot is a visualization of the paths that points in the complex plane take as they are iteratively mapped according to a specific formula. This implementation utilizes CUDA (Compute Unified Device Architecture) to parallelize the computation, significantly speeding up the generation process.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/[username]/cuda-buddhabrot.git
```

2. Navigate to the cloned directory:

```bash
cd cuda-buddhabrot
```

3. Compile the CUDA script using `nvcc`:

```bash
nvcc cuda_buddhabrot.cu -o cuda_buddhabrot
```

4. Run the compiled executable:

```bash
./cuda_buddhabrot
```

## Usage

The `cuda_buddhabrot` executable generates the Buddhabrot fractal with customizable parameters such as size and iterations. By default, it generates a Buddhabrot image with a size of 15000x15000 pixels and 10000 iterations.

## Contributors

- [Your Name](https://github.com/[your-username]) - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
