#include <CL/cl.h>
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

int ceil_mod(int x, int m) {
    int r = x % m;
    return x + (r == 0 ? 0 : m - r);
}

int main() {
    int N, M;

    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);

    std::cin >> N >> M;

    double a[N * N];
    double b[M * M];
    double c[N * N];
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cin >> a[i * N + j];
        }
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            std::cin >> b[i * M + j];
        }
    }

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("matrix_conv.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        size_t const block_size = 16;
        program.build(devices, "-D BLOCK_SIZE=16");

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(double) * N * N);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(double) * M * M);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * N * N);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N * N, a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * M * M, b);

        // load named kernel from opencl source
        cl::Kernel kernel(program, "matrix_conv");
        cl::KernelFunctor matrix_conv(kernel, queue, cl::NullRange,
                cl::NDRange(ceil_mod(N, block_size), ceil_mod(N, block_size)),
                cl::NDRange(block_size, block_size));
        matrix_conv(dev_a, dev_b, dev_c, (int)N, (int)M);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * N * N, c);

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                std::cout << c[i * N + j] << " ";
            }
            std::cout << "\n";
        }
    }
    catch (cl::Error& e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}