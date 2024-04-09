#include <stdio.h>

__global__ void HelloFromGPU(void)
{
    printf("Hello from GPU\n");
}

int main(void)
{
    printf("Hello from CPU\n");
    HelloFromGPU<<<1, 5>>>();
    cudaDeviceReset();
    return 0;
}
