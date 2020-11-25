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
	// TODO: Blur image on GPU (L1/texture cache)

	// === Task 6 ===
	// TODO: Blur image on GPU (all memory types)
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 7) nvprof output
 * 
 * Answer: TODO
 * 
 ************************************************************/
