#include <cuda_runtime.h>
#include <stdio.h>
#include "image.h"
#include "common.h"
#include "conv_gpu.h"

__global__ void conv_v_gpu_gmem_kernel(unsigned int *dst_data, unsigned int *src_data, float *kernel_data, int width, int height, int kernel_size, size_t pitch)
{
    int tx = blockIdx.x * blockDim.y + threadIdx.x;
    if (tx >= width)
        return;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty >= height)
        return;

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        int yy = ty + (i - kernel_size / 2);

        // Clamp to [0, h-1]
        int tmp = (yy < (height - 1)) ? yy : (height - 1);
        yy = (tmp > 0) ? tmp : 0;

        // unsigned int pixel = src_data[ty * width + xx];
        unsigned int *src_data_row = (unsigned int*)((char*)src_data + yy * pitch);
        unsigned int pixel = src_data_row[tx];

        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * kernel_data[i];
        gg += g * kernel_data[i];
        bb += b * kernel_data[i];
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

    // dst_data[ty * width + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_h_gpu_gmem_kernel(unsigned int *dst_data, unsigned int *src_data, float *kernel_data, int width, int height, int kernel_size, size_t pitch)
{
    int tx = blockIdx.x * blockDim.y + threadIdx.x;
    if (tx >= width)
        return;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty >= height)
        return;

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        int xx = tx + (i - kernel_size / 2);

        // Clamp to [0, w-1]
        int tmp = (xx < (width - 1)) ? xx : (width - 1);
        xx = (tmp > 0) ? tmp : 0;

        // unsigned int pixel = src_data[ty * width + xx];
        unsigned int *src_data_row = (unsigned int*)((char*)src_data + ty * pitch);
        unsigned int pixel = src_data_row[xx];

        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * kernel_data[i];
        gg += g * kernel_data[i];
        bb += b * kernel_data[i];
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

    // dst_data[ty * width + tx] = rr_c | (gg_c << 8) | (bb_c << 16);
    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

void conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_h_gpu_gmem_kernel<<<grid_dim, block_dim>>>(dst.data, src.data, kernel.data, w, h, kernel.ks, src.pitch);
}

void conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_v_gpu_gmem_kernel<<<grid_dim, block_dim>>>(dst.data, src.data, kernel.data, w, h, kernel.ks, src.pitch);
}