// CUDA smoke test (plan native-g0-g1 step 5, NG-5): a real device query and a vector add on a non-default stream.
// Every CUDA return code is checked; the printed summary is what docs/environment.md records.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../check.hpp"

#define MM_CUDA(call)                                                                                   \
  do {                                                                                                  \
    const cudaError_t err_ = (call);                                                                    \
    if (err_ != cudaSuccess) {                                                                          \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err_)); \
      return 1;                                                                                         \
    }                                                                                                   \
  } while (0)

__global__ void add(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int main() {
  int devices = 0;
  MM_CUDA(cudaGetDeviceCount(&devices));
  MM_CHECK(devices > 0);
  if (devices == 0) return 1;

  int driver = 0;
  int runtime = 0;
  MM_CUDA(cudaDriverGetVersion(&driver));
  MM_CUDA(cudaRuntimeGetVersion(&runtime));
  cudaDeviceProp prop{};
  MM_CUDA(cudaGetDeviceProperties(&prop, 0));
  size_t free_b = 0;
  size_t total_b = 0;
  MM_CUDA(cudaMemGetInfo(&free_b, &total_b));
  std::printf("device 0: %s, compute capability %d.%d, %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("memory: %zu MiB total, %zu MiB free\n", total_b >> 20, free_b >> 20);
  std::printf("CUDA driver API %d, runtime %d\n", driver, runtime);

  constexpr int n = 1 << 20;
  std::vector<float> a(n);
  std::vector<float> b(n);
  std::vector<float> out(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 0.5f * static_cast<float>(i);
  }

  cudaStream_t stream = nullptr;
  MM_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_out = nullptr;
  const size_t bytes = n * sizeof(float);
  MM_CUDA(cudaMallocAsync(&d_a, bytes, stream));
  MM_CUDA(cudaMallocAsync(&d_b, bytes, stream));
  MM_CUDA(cudaMallocAsync(&d_out, bytes, stream));
  MM_CUDA(cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream));
  MM_CUDA(cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream));
  add<<<(n + 255) / 256, 256, 0, stream>>>(d_a, d_b, d_out, n);
  MM_CUDA(cudaGetLastError());
  MM_CUDA(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream));
  MM_CUDA(cudaFreeAsync(d_a, stream));
  MM_CUDA(cudaFreeAsync(d_b, stream));
  MM_CUDA(cudaFreeAsync(d_out, stream));
  MM_CUDA(cudaStreamSynchronize(stream));
  MM_CUDA(cudaStreamDestroy(stream));

  int wrong = 0;
  for (int i = 0; i < n; ++i) {
    if (out[i] != 1.5f * static_cast<float>(i)) ++wrong;
  }
  MM_CHECK(wrong == 0);
  std::printf("vector add on a non-default stream: %d elements, %d wrong\n", n, wrong);
  return motionman::test::failures == 0 ? 0 : 1;
}
