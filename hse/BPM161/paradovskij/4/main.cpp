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

size_t const BLOCK_SIZE = 256;

void add_to_every(double* a, double* ds, int N, int m, int part_size, cl::Context& context, cl::CommandQueue& queue, cl::Program& program) {
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(double) * N);
    cl::Buffer dev_ds(context, CL_MEM_READ_ONLY,  sizeof(double) * m);
    cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * N);

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N, a);
    queue.enqueueWriteBuffer(dev_ds, CL_TRUE, 0, sizeof(double) * m, ds);
    queue.finish();

    int data_size = ceil_mod(N, BLOCK_SIZE);
    cl::Kernel kernel(program, "add");
    cl::KernelFunctor add(kernel, queue, cl::NullRange, cl::NDRange(data_size), cl::NDRange(BLOCK_SIZE));

    add(dev_a, dev_ds, dev_c, (int)part_size, (int)N);

    queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * N, a);
}

void scan_by_blocks(double* a, double* c, int N, cl::Context& context, cl::CommandQueue& queue, cl::Program& program) {
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(double) * N);
    cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * N);

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N, a);
    queue.finish();

    int data_size = ceil_mod(N, BLOCK_SIZE);

    cl::Kernel kernel(program, "scan");
    cl::KernelFunctor scan(kernel, queue, cl::NullRange, cl::NDRange(data_size), cl::NDRange(BLOCK_SIZE));

    scan(dev_a, dev_c, cl::__local(sizeof(double) * BLOCK_SIZE), cl::__local(sizeof(double) * BLOCK_SIZE), (int)N);

    queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * N, c);

    if (N > BLOCK_SIZE) {
        int num_of_blocks = ceil_mod(N, BLOCK_SIZE) / BLOCK_SIZE;
        double sums[num_of_blocks];
        sums[0] = 0;
        for (int i = 1; i < num_of_blocks; i++) {
            sums[i] = c[BLOCK_SIZE * i - 1];
        }
        double scan_sums[num_of_blocks];
        scan_by_blocks(sums, scan_sums, num_of_blocks, context, queue, program);

        add_to_every(c, scan_sums, N, num_of_blocks, BLOCK_SIZE, context, queue, program);
    }
}


int main() {
    int N;

    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);

    std::cin >> N;

    double a[N];
    double c[N];
    for (size_t i = 0; i < N; ++i) {
        std::cin >> a[i];
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
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        scan_by_blocks(a, c, N, context, queue, program);

        for (size_t i = 0; i < N; ++i) {
            std::cout << c[i] << " ";
        }
    }
    catch (cl::Error& e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}