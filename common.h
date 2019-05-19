#pragma once

#include <iostream>
#include <cuda_runtime_api.h>

inline void checkCudaCall(cudaError_t error, const char* file, int line)
{
    if (error)
    {
        std::cout << "CUDA error at " << file << ":" << line << std::endl;
        std::cout << cudaGetErrorName(error) << " :: " << cudaGetErrorString(error) << std::endl;
    }
}
#define CHECK_CUDA_CALL(err) (checkCudaCall(err, __FILE__, __LINE__))
#define DISABLE_COPY(T) T(const T& other) = delete;\
T& operator=(const T& other) = delete;\
T(const T&& other) = delete;


inline cudaDeviceProp initGPU()
{
    int deviceCount;
    CHECK_CUDA_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA device was found" << std::endl;
        std::exit(1);
    }

    cudaDeviceProp prop{};
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
    return prop;
}

class CudaTimer
{
public:
    explicit CudaTimer(bool automatic = false) : automatic(automatic)
    {
        CHECK_CUDA_CALL(cudaEventCreate(&this->startEvent));
        CHECK_CUDA_CALL(cudaEventCreate(&this->stopEvent));

        this->start();
    }
    ~CudaTimer()
    {
        if (this->automatic)
        {
            this->stop_wait();
            std::cerr << this->get_time() << " ms" << std::endl;
        }

        CHECK_CUDA_CALL(cudaEventDestroy(this->startEvent));
        CHECK_CUDA_CALL(cudaEventDestroy(this->stopEvent));
    }

    CudaTimer(const CudaTimer& other) = delete;
    CudaTimer& operator=(const CudaTimer& other) = delete;
    CudaTimer(CudaTimer&& other) = delete;

    void start() const
    {
        CHECK_CUDA_CALL(cudaEventRecord(this->startEvent));
    }
    void stop_wait() const
    {
        CHECK_CUDA_CALL(cudaEventRecord(this->stopEvent));
        CHECK_CUDA_CALL(cudaEventSynchronize(this->stopEvent));
    }
    float get_time() const
    {
        float time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&time, this->startEvent, this->stopEvent));
        return time;
    }

    void print(const std::string& message)
    {
        std::cerr << message << this->get_time() << " ms" << std::endl;
    }

private:
    cudaEvent_t startEvent, stopEvent;
    bool automatic;
};

template <typename T>
class CudaMemory
{
public:
    explicit CudaMemory(size_t count = 1, T* mem = nullptr) : count(count)
    {
        CHECK_CUDA_CALL(cudaMalloc(&this->devicePtr, sizeof(T) * count));

        if (mem)
        {
            this->store(mem, count);
        }
        else this->zero();
    }
    CudaMemory(size_t count, T value) : count(count)
    {
        CHECK_CUDA_CALL(cudaMalloc(&this->devicePtr, sizeof(T) * count));
        CHECK_CUDA_CALL(cudaMemset(this->devicePtr, value, sizeof(T) * count));
    }
    ~CudaMemory()
    {
        CHECK_CUDA_CALL(cudaFree(this->devicePtr));
        this->devicePtr = nullptr;
    }

    DISABLE_COPY(CudaMemory);

    T* operator*()
    {
        return this->devicePtr;
    }
    T* pointer() const
    {
        return this->devicePtr;
    }

    void load(T* dest, size_t count = 1) const
    {
        if (count == 0)
        {
            count = this->count;
        }

        CHECK_CUDA_CALL(cudaMemcpy(dest, this->devicePtr, sizeof(T) * count, cudaMemcpyDeviceToHost));
    }
    void store(const T* src, size_t count = 1, size_t start_index = 0)
    {
        if (count == 0)
        {
            count = this->count;
        }

        CHECK_CUDA_CALL(cudaMemcpy(this->devicePtr + start_index, src, sizeof(T) * count, cudaMemcpyHostToDevice));
    }

    void zero()
    {
        CHECK_CUDA_CALL(cudaMemset(this->devicePtr, 0, sizeof(T) * count));
    }

private:
    T* devicePtr = nullptr;
    size_t count;
};
