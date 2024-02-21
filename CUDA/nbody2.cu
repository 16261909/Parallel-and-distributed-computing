#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define SOFTENING 1e-9f
#define dt (0.01f)
#define blocksize 256
#define dup 8
typedef struct {
	float x, y, z, vx, vy, vz;
} Body;
void randomizeBodies(float *data, int n) {
	for (int i = 0; i < n; i++) {
		data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}
__global__ void bodyForce(Body *p,int n, int numtiles) {
	register int i=threadIdx.x+blockIdx.x*blocksize,st=i/n;
	if(i>=dup*n)return;
	i%=n;
	float3 mypos={p[i].x, p[i].y, p[i].z};
	float3 acc={0.0f,0.0f,0.0f}; 
	const int partsize=blocksize/dup;
	
//	__shared__ float3 sp[blocsize];
//	for (int tile = 0, pos = 0; tile < numtiles/dup; tile++, pos += blockDim.x * dup)
//	{
//		sp[threadIdx.x].x = p[pos + st + dup * threadIdx.x].x;
//		sp[threadIdx.x].y = p[pos + st + dup * threadIdx.x].y;
//		sp[threadIdx.x].z = p[pos + st + dup * threadIdx.x].z;
//		__syncthreads();//Ensure all data are written 
//		#pragma unroll
//		for(int j = 0; j < blocksize; j++)
//		{

	__shared__ float3 sp[partsize];
	for (int tile = 0, pos = 0; tile < numtiles; tile++, pos += blockDim.x)
	{
		if(threadIdx.x%dup==st)
		{
			sp[threadIdx.x/dup].x = p[pos + threadIdx.x].x;
			sp[threadIdx.x/dup].y = p[pos + threadIdx.x].y;
			sp[threadIdx.x/dup].z = p[pos + threadIdx.x].z;
		}
		__syncthreads();//Ensure all data are written 
		#pragma unroll
		for(int j = 0; j < blocksize/dup; j++)
		{
			float3 r={0.0f,0.0f,0.0f};
			r.x=sp[j].x-mypos.x;
			r.y=sp[j].y-mypos.y;
			r.z=sp[j].z-mypos.z;
			float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
			float invDist = rsqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;
			acc.x+=r.x*invDist3;
			acc.y+=r.y*invDist3; 
			acc.z+=r.z*invDist3;
		} 
		__syncthreads();//Prevent writing before others read 
	}
	atomicAdd(&p[i].vx,acc.x*dt);
	atomicAdd(&p[i].vy,acc.y*dt);
	atomicAdd(&p[i].vz,acc.z*dt);
}

__global__ void integrate_position(Body *p,int n)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=n)return;
		p[i].x+=p[i].vx*dt;
		p[i].y+=p[i].vy*dt;
		p[i].z+=p[i].vz*dt;
}

int main(const int argc, const char** argv)
{
	int nBodies = 1<<12;
	int salt = 0;
	if (argc > 1) nBodies = 2<<atoi(argv[1]);
	if (argc > 2) salt = atoi(argv[2]);
	const int nIters = 10;
	int blocknum=(nBodies+blocksize-1)/blocksize;
	int tilenum=(nBodies+blocksize-1)/blocksize;
	int bytes = nBodies * sizeof(Body);
	float *buf,*d_buf;
	cudaMallocHost((void**)&buf,bytes);
	cudaMalloc((void**)&d_buf,bytes);
	Body *d_p,*p;
	p=(Body*)buf;
	d_p=(Body*)d_buf;
	randomizeBodies(buf, 6 * nBodies); // Init pos / vel data
	double totalTime = 0.0;
	cudaMemcpy(d_buf,buf,bytes,cudaMemcpyHostToDevice);
	for (int iter = 0, now = 0; iter < nIters; iter++) {
		StartTimer();
		bodyForce<<<blocknum*dup,blocksize>>>(d_p,nBodies,tilenum);
		integrate_position<<<blocknum,blocksize>>>(d_p,nBodies); 
		if(iter+1==nIters)cudaMemcpy(buf,d_buf,bytes,cudaMemcpyDeviceToHost);
		else cudaDeviceSynchronize(); 
		const double tElapsed = GetTimer() / 1000.0;
		totalTime += tElapsed;
	}
	double avgTime = totalTime / (double)(nIters);
	float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
#ifdef ASSESS
	checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
	checkAccuracy(buf, nBodies);
	printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
	salt += 1;
#endif
	cudaFree(buf);
	cudaFree(d_buf);
}

