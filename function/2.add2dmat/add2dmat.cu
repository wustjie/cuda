#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<algorithm>

//随机初始化数组
void init(float* ip, float size) {
	for (int i = 0; i < size; i++) {
		ip[i] = float(rand() & 0xff) / 66.6;
	}
}

//打印数组
void printMatrix(float* a, float* b, float* c, const int nx, const int ny) {
	float* ia = a;
	float* ib = b;
	float* ic = c;
	printf("\nMatric:(%d,%d)\n", nx, ny);
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			printf("%f+%f=%f", ia[ix], ib[ix], ic[ix]);
		}
		ia += nx;
		ib += nx;
		ic += nx;
		printf("\n");
	}
	printf("\n");
}
//打印矩阵之差
void printResult(float* c, float* cc, const int nx, const int ny) {
	float* ic = c;
	float* icc = cc;
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			printf("%f", ic[ix] - icc[ix]);
		}
		ic += nx;
		icc += nx;
		printf("\n");
	}
	printf("\n");
}

//验证结果
void checkResult(float* hostRef, float* gpuRef, const int N) {
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Array don't match");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) {
		printf("Array match.\n\n");
		return;
	}
}

//CPU上两个矩阵相加
void sumMatrixOnHost(float* a, float* b, float* c, const int nx, const int ny) {
	float* ia = a;
	float* ib = b;
	float* ic = c;
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}
}
__global__ void sumMatrixOnGPU2D(float* a, float* b, float* c, const int nx, const int ny) {
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int idx = iy * nx + ix;
	if (ix < nx && iy < ny) {
		c[idx] = a[idx] + b[idx];
	}
}

int main() {
	int dev = 0;
	cudaDeviceProp deviceprop;
	cudaGetDeviceProperties(&deviceprop, dev);
	printf("using Device :%d %s\n\n", dev, deviceprop.name);

	//设置矩阵维度
	int nx = 1 << 12;
	int ny = 1 << 12;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	//分配CPU的相关数据内存
	float* h_a, *h_b, *h_c, *h_cc;
	h_a = (float*)malloc(nBytes);
	h_b = (float*)malloc(nBytes);
	h_c = (float*)malloc(nBytes);
	h_cc = (float*)malloc(nBytes);

	//初始化数据
	init(h_a, nxy);
	init(h_b, nxy);

	//开始计时
	clock_t cpuStart = clock();
	sumMatrixOnHost(h_a, h_b, h_c, nx, ny);
	clock_t cpuEnd = clock();
	float cpuTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
	printf("cpu time %f\n", cpuTime);

	//分配gpu内存
	float* d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a, nBytes);
	cudaMalloc((void**)&d_b, nBytes);
	cudaMalloc((void**)&d_c, nBytes);

	//初始化网格以及块大小
	dim3 block(128,1);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	//数据从cpu拷贝gpu
	cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
	//gpu调用核函数
	clock_t gpuStart = clock();
	sumMatrixOnGPU2D << <grid, block >> > (d_a, d_b, d_c, nx, ny);
	cudaDeviceSynchronize();
	clock_t gpuEnd = clock();
	float gpuTime = (float)(gpuEnd - gpuStart) / CLOCKS_PER_SEC;
	printf("GPU time:%f\n", gpuTime);

	// 结果从gpu再拷贝回cpu
	cudaMemcpy(h_cc, d_c, nBytes, cudaMemcpyDeviceToHost);
	checkResult(h_c, h_cc, nxy);
	//释放内存
	cudaFree(d_a); 
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_cc);

	return 0;

}

