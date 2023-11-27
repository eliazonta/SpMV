#ifndef __utils__
#define __utils__
#pragma once 
#include <chrono>
#include <cmath>
#include <mutex>
#include <iostream>

///////////////////////////////////////////////////////////////
// CUDA error check
//////////////////////////////////////////////////////////////
#define cuda_error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//////////////////////////////////////////////////////////////
// memory allocation
//////////////////////////////////////////////////////////////
template <typename T>
T* malloc_device(size_t n)
{
    void* p;
    auto status = cudaMalloc(&p, n * sizeof(T));
    cuda_check_status(status);
    return (T*)p;
}

template <typename T>
T* malloc_managed(size_t n, T value = T())
{
    T* p;
    cuda_error_check(cudaMallocManaged(&p, n * sizeof(T)));
    std::fill(p, p + n, value);
    return p;
}

template <typename T>
T* malloc_pinned(size_t n, T value = T())
{
    T* p = nullptr;
    cuda_error_check(((void**)&p, n * sizeof(T), 0));
    std::fill(p, p + n, value);
    return p;
}


///////////////////////////////////////////////////////////////////
// CUDA memory copy
//////////////////////////////////////////////////////////////////
template <typename T>
void copy_to_device(T* from, T* to, size_t n)
{
    cuda_error_check(cudaMemcpy(to, from, n * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void copy_to_host(T* from, T* to, size_t n)
{
    cuda_error_check(cudaMemcpy(to, from, n * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void copy_to_device_async(const T* from, T* to, size_t n, cudaStream_t stream = NULL)
{
    cuda_error_check(cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyHostToDevice, stream));

}

template <typename T>
void copy_to_host_async(const T* from, T* to, size_t n, cudaStream_t stream = NULL)
{
    cuda_error_check(cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

///////////////////////////////////////////////////////////////////
// others
//////////////////////////////////////////////////////////////////
static size_t read_arg(int argc, char** argv, size_t index, int default_value)
{
    if (argc > index)
    {
        try{
            auto n = std::stoi(argv[index]);
            if (n < 0)
            {
                return default_value;
            }
            return n;
        }catch(std::exception& e)
        {
            std::cerr << "error [invalid argument, expected a positive integer] | compiler says :  " 
                        << e.what() << std::endl;
            exit(1);
        }
    }
    return default_value;
}

template <typename T>
T* malloc_host(size_t n, T value = T())
{
    T* p = (T*)malloc(n * sizeof(T));
    std::fill(p, p + n, value);
    return p;
}

//aliases
using clock_type = std::chrono::high_resolution_clock;
using duration_type = std::chrono::duration<double>;

static  double get_time()
{
    static auto start_time = clock_type::now();
    return duration_type(clock_type::now() - start_time).count();
}

#endif