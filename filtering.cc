#include "filtering.h"
#include "image.h"
#include "common.h"
#include "conv_cpu.h"
#include "conv_gpu.h"

void filtering(const char *imgfile, int ks)
{
	// === Task 1 ===
	// TODO: Load image
	image_cpu h_img_src(imgfile);
	image_cpu h_img_dst(imgfile);
	// TODO: Generate gaussian filter kernel
	filterkernel_cpu h_kernel(ks);
	// TODO: Blur image on CPU
	conv_h_cpu(h_img_dst, h_img_src, h_kernel);
	conv_v_cpu(h_img_src, h_img_dst, h_kernel);
	h_img_src.save("out_cpu.ppm");

	// === Task 2 ===
	image_cpu h_img_2(imgfile);
	image_gpu d_img_src(h_img_2.width, h_img_2.height);
	h_img_2.upload(d_img_src);
	image_gpu d_img_dst(h_img_2.width, h_img_2.height);
	filterkernel_gpu d_kernel(ks);
	h_kernel.upload(d_kernel);
	// TODO: Blur image on GPU (Global memory)
	conv_h_gpu_gmem(d_img_dst, d_img_src, d_kernel);
	conv_v_gpu_gmem(d_img_src, d_img_dst, d_kernel);
	h_img_2.download(d_img_src);
	h_img_2.save("out_gpu_gmem.ppm");

	// === Task 3 ===
	image_cpu h_img_3(imgfile);
	h_img_3.upload(d_img_src);
	// TODO: Blur image on GPU (Shared memory)
	conv_h_gpu_smem(d_img_dst, d_img_src, d_kernel);
	conv_v_gpu_smem(d_img_src, d_img_dst, d_kernel);
	h_img_3.download(d_img_src);
	h_img_3.save("out_gpu_smem.ppm");

	// === Task 4 ===
	image_cpu h_img_4(imgfile);
	h_img_4.upload(d_img_src);
	// TODO: Blur image on GPU (Constant memory)
	conv_h_gpu_cmem(d_img_dst, d_img_src, d_kernel);
	conv_v_gpu_cmem(d_img_src, d_img_dst, d_kernel);
	h_img_4.download(d_img_src);
	h_img_4.save("out_gpu_cmem.ppm");

	// === Task 5 ===
	image_cpu h_img_5(imgfile);
	h_img_5.upload(d_img_src);
	// TODO: Blur image on GPU (L1/texture cache)
	conv_h_gpu_tmem(d_img_dst, d_img_src, d_kernel);
	conv_v_gpu_tmem(d_img_src, d_img_dst, d_kernel);
	h_img_5.download(d_img_src);
	h_img_5.save("out_gpu_tmem.ppm");

	// === Task 6 ===
	image_cpu h_img_6(imgfile);
	h_img_6.upload(d_img_src);
	// TODO: Blur image on GPU (all memory types)
	conv_h_gpu_all(d_img_dst, d_img_src, d_kernel);
	conv_v_gpu_all(d_img_src, d_img_dst, d_kernel);
	h_img_5.download(d_img_src);
	h_img_5.save("out_gpu_all.ppm");
	
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 7) nvprof output
 * 
 * Answer: TODO
dnguyen@gccg301:~/pmpp_ex2/build$ ~/cuda-10.2/bin/nvprof ./gauss_filter ../cornellBoxSphere_2048x2048.ppm 127
PMPP Hello World!
==21078== NVPROF is profiling process 21078, command: ./gauss_filter ../cornellBoxSphere_2048x2048.ppm 127
==21078== Profiling application: ./gauss_filter ../cornellBoxSphere_2048x2048.ppm 127
==21078== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   18.04%  18.727ms        10  1.8727ms  1.3760us  3.8922ms  [CUDA memcpy HtoD]
                   17.58%  18.250ms         9  2.0278ms  2.2400us  3.7051ms  [CUDA memcpy DtoH]
                    8.99%  9.3289ms         1  9.3289ms  9.3289ms  9.3289ms  conv_h_gpu_smem_kernel(unsigned int*, unsigned int const *, float const *, int, int, int, unsigned long)
                    8.94%  9.2843ms         1  9.2843ms  9.2843ms  9.2843ms  conv_h_gpu_all_kernel(unsigned int*, unsigned int const *, int, int, int, unsigned long)
                    6.32%  6.5570ms         1  6.5570ms  6.5570ms  6.5570ms  conv_v_gpu_cmem_kernel(unsigned int*, unsigned int const *, int, int, int, unsigned long)
                    6.17%  6.4078ms         1  6.4078ms  6.4078ms  6.4078ms  conv_v_gpu_tmem_kernel(unsigned int*, unsigned int const *, float const *, int, int, int, unsigned long)
                    6.17%  6.4046ms         1  6.4046ms  6.4046ms  6.4046ms  conv_v_gpu_gmem_kernel(unsigned int*, unsigned int const *, float const *, int, int, int, unsigned long)
                    5.65%  5.8691ms         1  5.8691ms  5.8691ms  5.8691ms  conv_v_gpu_smem_kernel(unsigned int*, unsigned int const *, float const *, int, int, int, unsigned long)
                    5.63%  5.8469ms         1  5.8469ms  5.8469ms  5.8469ms  conv_v_gpu_all_kernel(unsigned int*, unsigned int const *, int, int, int, unsigned long)
                    5.60%  5.8166ms         1  5.8166ms  5.8166ms  5.8166ms  conv_h_gpu_cmem_kernel(unsigned int*, unsigned int const *, int, int, int, unsigned long)
                    5.46%  5.6667ms         1  5.6667ms  5.6667ms  5.6667ms  conv_h_gpu_gmem_kernel(unsigned int*, unsigned int const *, float const *, int, int, int, unsigned long)
                    5.46%  5.6666ms         1  5.6666ms  5.6666ms  5.6666ms  conv_h_gpu_tmem_kernel(unsigned int*, unsigned int const *, float const *, int, int, int, unsigned long)
      API calls:   65.51%  249.02ms         2  124.51ms  384.09us  248.63ms  cudaMallocPitch
                   24.14%  91.770ms        10  9.1770ms  3.8905ms  19.348ms  cudaMemcpy2D
                    4.03%  15.320ms         5  3.0641ms  23.123us  9.2934ms  cudaMemcpy
                    3.02%  11.476ms         8  1.4345ms  597.83us  1.8725ms  cuDeviceTotalMem
                    2.16%  8.2287ms       776  10.603us     229ns  640.47us  cuDeviceGetAttribute
                    0.73%  2.7727ms         3  924.22us  646.57us  1.1324ms  cudaFree
                    0.20%  761.51us         8  95.188us  83.307us  124.27us  cuDeviceGetName
                    0.12%  440.79us         1  440.79us  440.79us  440.79us  cudaMalloc
                    0.06%  244.44us        10  24.444us  9.4170us  44.774us  cudaLaunchKernel
                    0.01%  37.287us         4  9.3210us  6.5710us  12.639us  cudaMemcpyToSymbol
                    0.01%  32.497us         8  4.0620us  2.4020us  8.4240us  cuDeviceGetPCIBusId
                    0.00%  10.030us        16     626ns     382ns  2.2150us  cuDeviceGet
                    0.00%  10.003us        15     666ns     197ns  1.9960us  cudaGetLastError
                    0.00%  4.2460us         8     530ns     253ns     946ns  cuDeviceGetUuid
                    0.00%  3.1820us         3  1.0600us     550ns  1.9430us  cuDeviceGetCount
 * 
 ************************************************************/
