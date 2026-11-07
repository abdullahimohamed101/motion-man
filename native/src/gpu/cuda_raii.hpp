// Owning wrappers for CUDA resources (plan native-g0-g1 step 9, spec C§7): every allocation, stream and event is
// released by its destructor, and every CUDA call is checked and turned into an exception naming the call.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#define MM_CUDA_CHECK(call)                                                                                \
  do {                                                                                                     \
    const cudaError_t mm_err_ = (call);                                                                    \
    if (mm_err_ != cudaSuccess) {                                                                          \
      throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(mm_err_) + " at " + \
                               __FILE__ + ":" + std::to_string(__LINE__));                                 \
    }                                                                                                      \
  } while (0)

namespace motionman::gpu {

// Move-only owner of one handle; Free is a stateless deleter that must not throw.
template <typename T, void (*Free)(T)>
class Owned {
 public:
  Owned() = default;
  explicit Owned(T h) : h_(h) {}
  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;
  Owned(Owned&& o) noexcept : h_(std::exchange(o.h_, T{})) {}
  Owned& operator=(Owned&& o) noexcept {
    if (this != &o) {
      reset();
      h_ = std::exchange(o.h_, T{});
    }
    return *this;
  }
  ~Owned() { reset(); }
  T get() const { return h_; }
  void reset() {
    if (h_) Free(std::exchange(h_, T{}));
  }

 private:
  T h_{};
};

inline void free_device(void* p) { cudaFree(p); }
inline void free_pinned(void* p) { cudaFreeHost(p); }
inline void destroy_stream(cudaStream_t s) { cudaStreamDestroy(s); }
inline void destroy_event(cudaEvent_t e) { cudaEventDestroy(e); }

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    void* p = nullptr;
    MM_CUDA_CHECK(cudaMalloc(&p, bytes));
    p_ = Owned<void*, free_device>(p);
  }
  void* get() const { return p_.get(); }
  std::size_t bytes() const { return bytes_; }

 private:
  Owned<void*, free_device> p_;
  std::size_t bytes_ = 0;
};

class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t bytes) : bytes_(bytes) {
    void* p = nullptr;
    MM_CUDA_CHECK(cudaMallocHost(&p, bytes));
    p_ = Owned<void*, free_pinned>(p);
  }
  void* get() const { return p_.get(); }
  std::size_t bytes() const { return bytes_; }

 private:
  Owned<void*, free_pinned> p_;
  std::size_t bytes_ = 0;
};

class Stream {
 public:
  Stream() {
    cudaStream_t s = nullptr;
    MM_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    s_ = Owned<cudaStream_t, destroy_stream>(s);
  }
  cudaStream_t get() const { return s_.get(); }
  void sync() const { MM_CUDA_CHECK(cudaStreamSynchronize(s_.get())); }

 private:
  Owned<cudaStream_t, destroy_stream> s_;
};

class Event {
 public:
  Event() {
    cudaEvent_t e = nullptr;
    MM_CUDA_CHECK(cudaEventCreate(&e));
    e_ = Owned<cudaEvent_t, destroy_event>(e);
  }
  cudaEvent_t get() const { return e_.get(); }
  void record(const Stream& s) const { MM_CUDA_CHECK(cudaEventRecord(e_.get(), s.get())); }

 private:
  Owned<cudaEvent_t, destroy_event> e_;
};

}  // namespace motionman::gpu
