#include <cstdio>
#include <device_launch_parameters.h>

#include "../common.h"

#define REPETITIONS 1000
#define MEMORY_SIZE 4096

using Type = uint32_t;

__global__ void kernel(int offset)
{
    __shared__ Type sharedMem[MEMORY_SIZE];

    int threadId = threadIdx.x;

    // init shared memory
    if (threadId == 0)
    {
        for (int i = 0; i < MEMORY_SIZE; i++) sharedMem[i] = 0;
    }
    __syncthreads();

    // repeatedly read and write to shared memory
    uint32_t index = threadId * offset;
    for (int i = 0; i < 10000; i++)
    {
        sharedMem[index] += index * i;
        index += 32;
        index %= MEMORY_SIZE;
    }
}

static void benchmark(int offset)
{
    float time = 0;
    for (int i = 0; i < REPETITIONS; i++)
    {
        CudaTimer timer;
        kernel<<<1, 32>>>(offset);  // launch exactly one warp
        CHECK_CUDA_CALL(cudaPeekAtLastError());
        timer.stop_wait();
        time += timer.get_time();
    }

    std::cerr << time / REPETITIONS << std::endl;
}

void run(int offset)
{
    auto prop = initGPU();
    std::cout << "Warp size: " << prop.warpSize << std::endl;

    // query shared memory bank size
    cudaSharedMemConfig sharedMemConfig;
    CHECK_CUDA_CALL(cudaDeviceGetSharedMemConfig(&sharedMemConfig));
    std::cout << "Bank size: " << (sharedMemConfig == cudaSharedMemBankSizeEightByte ? 8 : 4) << std::endl;

    // set it to four, just in case
    CHECK_CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

    benchmark(offset);
}
