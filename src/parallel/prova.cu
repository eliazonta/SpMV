#include <iostream>

__global__
void kernel()
{
    printf("Hello World from GPU! %d, %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char** argv)
{
    std::cout << "Hello World from CPU!" << std::endl;
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}