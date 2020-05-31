#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_fft.h"

#define M_PI 3.14159265358979323846

#define FFT_SAMPLES_PER_KEY 10
#define FFT_WIDTH 10000
#define PRECISION_MULTIPLIER 5
#define WAVELENGTHS_PER_SAMPLE 20
#define FFT_RESULT_LEN 128 * FFT_SAMPLES_PER_KEY
#define OUTPUT_SIZE FFT_WIDTH * FFT_RESULT_LEN * sizeof(float)

__device__ int processed = 0;

__global__ void do_fft(const float* __restrict__  input, float* __restrict__  output, size_t len, uint32_t sample_rate) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (int i = index; i < FFT_RESULT_LEN; i += stride) {
    const float key = i / (float)FFT_SAMPLES_PER_KEY - 0.5f / FFT_SAMPLES_PER_KEY;
    const float freq = __powf(2, (key - 69 + 9 - 7 * 3 + 12 * 1) / 12) * 440;
    int wave_size = (int)(sample_rate / freq) * WAVELENGTHS_PER_SAMPLE;
    double wave_step = (double)wave_size * FFT_WIDTH / len;
    wave_step /= PRECISION_MULTIPLIER;
    wave_step = max(wave_step, 1.0f);
    wave_step = wave_step / FFT_WIDTH * len;
    wave_size = max(wave_size, 1000);
    for (double _l = 0; _l + wave_size < len; _l += wave_step) {
      int l = _l;
      float mult = freq / sample_rate * M_PI * 2;
      float sum_r = 0;
      float sum_i = 0;
      for (int j = 0; j < wave_size; j++) {
        float a = mult * j + M_PI;
        sum_r += cosf(a) * input[l + j];
        sum_i += sinf(a) * input[l + j];
      }
      float val = (fabsf(sum_r) + fabsf(sum_i)) / wave_size;
      int start = (int)((double)l * FFT_WIDTH / len);
      int end = (int)((double)(l + wave_size) * FFT_WIDTH / len);
      for (int p = start; p <= end; p++)
        output[p * FFT_RESULT_LEN + i] = val;
    }
    printf("processed %d bands\n", atomicAdd(&processed, 1) + 1);
  }
}

extern "C" float* process_audio(float* audio, size_t samples, uint32_t sample_rate) {
  // prepare to process the data
  float* gpu_input;
  float* gpu_output;
  cudaMallocManaged(&gpu_input, samples * sizeof(float));
  cudaMallocManaged(&gpu_output, OUTPUT_SIZE);
  memcpy(gpu_input, audio, samples * sizeof(float));

  // go!
  printf("doing the thing\n");
  int block_size = 128;
  int num_blocks = (samples + block_size - 1) / block_size;
  do_fft<<<num_blocks, block_size>>>(gpu_input, gpu_output, samples, sample_rate);
  //do_fft(gpu_input, gpu_output, samples / 2);
  printf("time to wait for the gpu\n");
  cudaDeviceSynchronize();
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  /*
  // write out the output data
  FILE* output = fopen("output.bin", "wb");
  fwrite(gpu_output, FFT_WIDTH * FFT_RESULT_LEN, sizeof(float), output);
  fclose(output);
  */

  // copy the output data to host-only memory
  float* output = (float*)malloc(OUTPUT_SIZE);
  memcpy(output, gpu_output, OUTPUT_SIZE);

  // clean up
  cudaFree(gpu_input);
  cudaFree(gpu_output);
  cudaDeviceReset();

  printf("success!\n");

  return output;
}