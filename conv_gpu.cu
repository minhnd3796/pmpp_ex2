#include <cuda_runtime.h>
#include <stdio.h>
#include "image.h"
#include "common.h"
#include "conv_gpu.h"

__constant__ float d_c_kernel[127];

__global__ void conv_v_gpu_gmem_kernel(unsigned int *dst_data, const unsigned int *src_data, const float *kernel_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
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

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_h_gpu_gmem_kernel(unsigned int *dst_data, const unsigned int *src_data, const float *kernel_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
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

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_h_gpu_smem_kernel(unsigned int *dst_data, const unsigned int *src_data, const float *kernel_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    extern __shared__ unsigned int shared_src_data[];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int shared_src_data_width = kernel_size / 2 + blockDim.x + kernel_size / 2;
    
    if (tx < width && ty < height)
    {
        unsigned int *src_data_row = (unsigned int*)((char*)src_data + ty * pitch);
        shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + threadIdx.x] = src_data_row[tx];
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (tx == 0)
                    shared_src_data[threadIdx.y * shared_src_data_width + i] = shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2];
                else
                {
                    int xx = blockIdx.x * blockDim.x - kernel_size / 2 + i;
                    xx = (xx > 0) ? xx : 0;
                    shared_src_data[threadIdx.y * shared_src_data_width + i] = src_data_row[xx];
                }
            }
        }
        if (threadIdx.x == blockDim.x - 1)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (tx == width - 1)
                    shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + blockDim.x + i] = shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + blockDim.x - 1];
                else
                {
                    int xx = (blockIdx.x + 1) * blockDim.x + i;
                    xx = (xx < (width - 1)) ? xx : (width - 1);
                    shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + blockDim.x + i] = src_data_row[xx];
                }
            }
        }
    }
    __syncthreads();
    if (ty >= height || tx >= width)
        return;

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        unsigned int pixel = shared_src_data[threadIdx.y * shared_src_data_width + threadIdx.x + i];

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

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_v_gpu_smem_kernel(unsigned int *dst_data, const unsigned int *src_data, const float *kernel_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    extern __shared__ unsigned int shared_src_data[];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx < width && ty < height)
    {
        unsigned int *src_data_row = (unsigned int*)((char*)src_data + ty * pitch);
        shared_src_data[(kernel_size / 2 + threadIdx.y) * blockDim.y + threadIdx.x] = src_data_row[tx];
        if (threadIdx.y == 0)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (ty == 0)
                    shared_src_data[(threadIdx.y + i) * blockDim.y + threadIdx.x] = shared_src_data[(threadIdx.y + kernel_size / 2) * blockDim.y + threadIdx.x];
                else
                {
                    int yy = blockIdx.y * blockDim.y - kernel_size / 2 + i;
                    yy = (yy > 0) ? yy : 0;
                    src_data_row = (unsigned int*)((char*)src_data + yy * pitch);
                    shared_src_data[(threadIdx.y + i) * blockDim.y + threadIdx.x] = src_data_row[tx];
                }
            }
        }
        if (threadIdx.y == blockDim.y - 1)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (ty == height - 1)
                    shared_src_data[(kernel_size / 2 + blockDim.y + i) * blockDim.y + threadIdx.x] = shared_src_data[(kernel_size / 2 + blockDim.y - 1) * blockDim.y + threadIdx.x];
                else
                {
                    int yy = (blockIdx.y + 1) * blockDim.y + i;
                    yy = (yy < (height - 1)) ? yy : (height - 1);
                    src_data_row = (unsigned int*)((char*)src_data + yy * pitch);
                    shared_src_data[(kernel_size / 2 + blockDim.y + i) * blockDim.y + threadIdx.x] = src_data_row[tx];
                }
            }
        }
    }
    __syncthreads();
    if (ty >= height || tx >= width)
        return;

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        unsigned int pixel = shared_src_data[(threadIdx.y + i) * blockDim.y + threadIdx.x];


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

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_h_gpu_cmem_kernel(unsigned int *dst_data, const unsigned int *src_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
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

        unsigned int *src_data_row = (unsigned int*)((char*)src_data + ty * pitch);
        unsigned int pixel = src_data_row[xx];

        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * d_c_kernel[i];
        gg += g * d_c_kernel[i];
        bb += b * d_c_kernel[i];
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_v_gpu_cmem_kernel(unsigned int *dst_data, const unsigned int *src_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
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

        unsigned int *src_data_row = (unsigned int*)((char*)src_data + yy * pitch);
        unsigned int pixel = src_data_row[tx];

        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * d_c_kernel[i];
        gg += g * d_c_kernel[i];
        bb += b * d_c_kernel[i];
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_v_gpu_tmem_kernel(unsigned int* __restrict__ dst_data, const unsigned int* __restrict__ src_data, const float* __restrict__ kernel_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
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

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_h_gpu_tmem_kernel(unsigned int* __restrict__ dst_data, const unsigned int* __restrict__ src_data, const float* __restrict__ kernel_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
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

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_h_gpu_all_kernel(unsigned int* __restrict__ dst_data, const unsigned int* __restrict__ src_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    extern __shared__ unsigned int shared_src_data[];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int shared_src_data_width = kernel_size / 2 + blockDim.x + kernel_size / 2;
    
    if (tx < width && ty < height)
    {
        unsigned int *src_data_row = (unsigned int*)((char*)src_data + ty * pitch);
        shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + threadIdx.x] = src_data_row[tx];
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (tx == 0)
                    shared_src_data[threadIdx.y * shared_src_data_width + i] = shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2];
                else
                {
                    int xx = blockIdx.x * blockDim.x - kernel_size / 2 + i;
                    xx = (xx > 0) ? xx : 0;
                    shared_src_data[threadIdx.y * shared_src_data_width + i] = src_data_row[xx];
                }
            }
        }
        if (threadIdx.x == blockDim.x - 1)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (tx == width - 1)
                    shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + blockDim.x + i] = shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + blockDim.x - 1];
                else
                {
                    int xx = (blockIdx.x + 1) * blockDim.x + i;
                    xx = (xx < (width - 1)) ? xx : (width - 1);
                    shared_src_data[threadIdx.y * shared_src_data_width + kernel_size / 2 + blockDim.x + i] = src_data_row[xx];
                }
            }
        }
    }
    __syncthreads();
    if (ty >= height || tx >= width)
        return;

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        unsigned int pixel = shared_src_data[threadIdx.y * shared_src_data_width + threadIdx.x + i];

        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * d_c_kernel[i];
        gg += g * d_c_kernel[i];
        bb += b * d_c_kernel[i];
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

    unsigned int *dst_data_row = (unsigned int*)((char*)dst_data + ty * pitch);
    dst_data_row[tx] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_v_gpu_all_kernel(unsigned int* __restrict__ dst_data, const unsigned int* __restrict__ src_data, const int width, const int height, const int kernel_size, const size_t pitch)
{
    extern __shared__ unsigned int shared_src_data[];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx < width && ty < height)
    {
        unsigned int *src_data_row = (unsigned int*)((char*)src_data + ty * pitch);
        shared_src_data[(kernel_size / 2 + threadIdx.y) * blockDim.y + threadIdx.x] = src_data_row[tx];
        if (threadIdx.y == 0)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (ty == 0)
                    shared_src_data[(threadIdx.y + i) * blockDim.y + threadIdx.x] = shared_src_data[(threadIdx.y + kernel_size / 2) * blockDim.y + threadIdx.x];
                else
                {
                    int yy = blockIdx.y * blockDim.y - kernel_size / 2 + i;
                    yy = (yy > 0) ? yy : 0;
                    src_data_row = (unsigned int*)((char*)src_data + yy * pitch);
                    shared_src_data[(threadIdx.y + i) * blockDim.y + threadIdx.x] = src_data_row[tx];
                }
            }
        }
        if (threadIdx.y == blockDim.y - 1)
        {
            for (int i = 0; i < kernel_size / 2; i++)
            {
                if (ty == height - 1)
                    shared_src_data[(kernel_size / 2 + blockDim.y + i) * blockDim.y + threadIdx.x] = shared_src_data[(kernel_size / 2 + blockDim.y - 1) * blockDim.y + threadIdx.x];
                else
                {
                    int yy = (blockIdx.y + 1) * blockDim.y + i;
                    yy = (yy < (height - 1)) ? yy : (height - 1);
                    src_data_row = (unsigned int*)((char*)src_data + yy * pitch);
                    shared_src_data[(kernel_size / 2 + blockDim.y + i) * blockDim.y + threadIdx.x] = src_data_row[tx];
                }
            }
        }
    }
    __syncthreads();
    if (ty >= height || tx >= width)
        return;

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        unsigned int pixel = shared_src_data[(threadIdx.y + i) * blockDim.y + threadIdx.x];


        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * d_c_kernel[i];
        gg += g * d_c_kernel[i];
        bb += b * d_c_kernel[i];
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

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

void conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    int shared_src_data_width = kernel.ks / 2 + block_width + kernel.ks / 2;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_h_gpu_smem_kernel<<<grid_dim, block_dim, shared_src_data_width * block_height * sizeof(unsigned int)>>>(dst.data, src.data, kernel.data, w, h, kernel.ks, src.pitch);
}

void conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    int shared_src_data_height = kernel.ks / 2 + block_height + kernel.ks / 2;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_v_gpu_smem_kernel<<<grid_dim, block_dim, block_width * shared_src_data_height * sizeof(unsigned int)>>>(dst.data, src.data, kernel.data, w, h, kernel.ks, src.pitch);
}

void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    float *h_kernel_data = new float[kernel.ks];
    cudaMemcpy(h_kernel_data, kernel.data, kernel.ks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_c_kernel, h_kernel_data, sizeof(float) * kernel.ks);
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_h_gpu_cmem_kernel<<<grid_dim, block_dim>>>(dst.data, src.data, w, h, kernel.ks, src.pitch);
}

void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    float *h_kernel_data = new float[kernel.ks];
    cudaMemcpy(h_kernel_data, kernel.data, kernel.ks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_c_kernel, h_kernel_data, sizeof(float) * kernel.ks);
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_v_gpu_cmem_kernel<<<grid_dim, block_dim>>>(dst.data, src.data, w, h, kernel.ks, src.pitch);
}

void conv_h_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_h_gpu_tmem_kernel<<<grid_dim, block_dim>>>(dst.data, src.data, kernel.data, w, h, kernel.ks, src.pitch);
}

void conv_v_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_v_gpu_tmem_kernel<<<grid_dim, block_dim>>>(dst.data, src.data, kernel.data, w, h, kernel.ks, src.pitch);
}

void conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    float *h_kernel_data = new float[kernel.ks];
    cudaMemcpy(h_kernel_data, kernel.data, kernel.ks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_c_kernel, h_kernel_data, sizeof(float) * kernel.ks);
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    int shared_src_data_width = kernel.ks / 2 + block_width + kernel.ks / 2;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_h_gpu_all_kernel<<<grid_dim, block_dim, shared_src_data_width * block_height * sizeof(unsigned int)>>>(dst.data, src.data, w, h, kernel.ks, src.pitch);
}

void conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel)
{
    float *h_kernel_data = new float[kernel.ks];
    cudaMemcpy(h_kernel_data, kernel.data, kernel.ks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(d_c_kernel, h_kernel_data, sizeof(float) * kernel.ks);
    int w = src.width, h = src.height;
    int block_width = 32;
    int block_height = 32;
    int shared_src_data_height = kernel.ks / 2 + block_height + kernel.ks / 2;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim(div_up(w, block_width), div_up(h, block_height));
    conv_v_gpu_all_kernel<<<grid_dim, block_dim, block_width * shared_src_data_height * sizeof(unsigned int)>>>(dst.data, src.data, w, h, kernel.ks, src.pitch);
}