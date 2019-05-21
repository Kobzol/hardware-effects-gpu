#include <cstdio>
#include <device_launch_parameters.h>
#include <cmath>

#include "../common.h"

#define REPETITIONS 1000
#define MEMORY_SIZE 4096

using Type = uint32_t;

__global__ void kernel(Type* memory)
{
    int threadId = threadIdx.x;

    // emulate some work
    for (int i = 0; i < 200; i++)
    {
        memory[threadId] += i * threadId;
        threadId += 32;
        threadId %= MEMORY_SIZE;
    }
}

static void benchmark(int smCount, int threadsPerSm, int sharedMemorySize)
{
    CudaMemory<Type> memory(MEMORY_SIZE);

    int numBlocks = smCount * 100;
    auto dim = static_cast<unsigned int>(sqrt(numBlocks));

    dim3 gridDim{ dim, dim, 1 };
    int threadsPerBlocks = threadsPerSm / 16;

    std::cout << "Launching " << dim * dim << " blocks, each with " << threadsPerBlocks << " threads " << std::endl;

    float time = 0;
    for (int i = 0; i < REPETITIONS; i++)
    {
        CudaTimer timer;
        kernel<<<gridDim, threadsPerBlocks, sharedMemorySize>>>(memory.pointer());
        CHECK_CUDA_CALL(cudaPeekAtLastError());
        timer.stop_wait();
        time += timer.get_time();
    }

    std::cerr << time / REPETITIONS << std::endl;
}

void run(int sharedMemorySize)
{
    auto prop = initGPU();
    std::cout << "SM per block: " << prop.sharedMemPerBlock << ", SM per multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << ", max threads per processor: " << prop.maxThreadsPerMultiProcessor << std::endl;

    benchmark(prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, sharedMemorySize);
}
