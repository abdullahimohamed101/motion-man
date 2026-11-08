#include "trt_detector.hpp"

#include <NvInfer.h>
#include <nvml.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

#include "cuda_raii.hpp"
#include "motionman/json.hpp"
#include "motionman/preprocess.hpp"
#include "motionman/sha256.hpp"

namespace motionman::gpu {
namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
  }
};

Logger& logger() {
  static Logger l;
  return l;
}

std::string compiled_tensorrt() {
  return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH) + "." + std::to_string(NV_TENSORRT_BUILD);
}

std::string driver_version() {
  if (nvmlInit_v2() != NVML_SUCCESS) return "unknown (NVML init failed)";
  char buf[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
  const bool ok = nvmlSystemGetDriverVersion(buf, sizeof buf) == NVML_SUCCESS;
  nvmlShutdown();
  return ok ? std::string(buf) : "unknown (NVML query failed)";
}

std::vector<char> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::size_t element_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    default: throw std::runtime_error("unexpected engine tensor data type");
  }
}

struct Io {
  const char* name;
  nvinfer1::TensorIOMode mode;
  nvinfer1::DataType type;
};

// The deploy contract (C§5, ADR-003): names, directions and types the engine must expose, in this order.
constexpr Io kContract[] = {
    {"images", nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kFLOAT},
    {"orig_target_sizes", nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kINT64},
    {"labels", nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kINT64},
    {"boxes", nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT},
    {"scores", nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT},
};

}  // namespace

struct TrtDetector::Impl {
  // Destruction runs bottom-up: buffers, context, engine, then runtime, as TensorRT requires.
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  Stream stream;
  DeviceBuffer device[5];
  PinnedBuffer host[5];
  std::int64_t queries = 0;
  int width = 0;
  int height = 0;
  std::vector<std::string> names;
  std::string engine_sha;
  std::string gpu;
  std::string driver;
};

std::unique_ptr<TrtDetector> TrtDetector::open(const DetectorPaths& paths) {
  const auto profile = json::parse_file(paths.profile_json);
  const auto model = json::parse_file(paths.model_dir + "/manifest.json");
  const std::string em_path = paths.model_dir + "/engine_manifest." + paths.precision + ".json";
  const auto em = json::parse_file(em_path);
  const auto& id = em.at("identity");

  int device = 0;
  MM_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  MM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::vector<std::string> problems;
  const auto expect = [&](const char* field, const std::string& now) {
    const json::Value* v = id.find(field);
    const std::string built = v && v->kind == json::Value::Kind::String ? v->string : "<missing>";
    if (built != now) problems.push_back(std::string(field) + ": engine built with '" + built + "', now '" + now + "'");
  };
  expect("profile", profile.at("profile").str());
  expect("precision", paths.precision);
  expect("onnx_sha256", model.at("onnx").at("sha256").str());
  expect("checkpoint_sha256", profile.at("checkpoint").at("sha256").str());
  expect("preprocessing_revision", kPreprocessingRevision);
  if (profile.at("preprocessing").at("revision").str() != kPreprocessingRevision) {
    problems.push_back("profile preprocessing revision '" + profile.at("preprocessing").at("revision").str() +
                       "' is not the one this binary implements ('" + kPreprocessingRevision + "')");
  }
  expect("tensorrt", compiled_tensorrt());
  if (getInferLibVersion() != NV_TENSORRT_VERSION) {
    problems.push_back("TensorRT library " + std::to_string(getInferLibVersion()) + " differs from headers " +
                       std::to_string(NV_TENSORRT_VERSION));
  }
  expect("gpu", prop.name);
  expect("compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
  const std::string driver = driver_version();
  expect("driver", driver);
  const std::string cuda = std::to_string(CUDART_VERSION / 1000) + "." + std::to_string(CUDART_VERSION % 1000 / 10);
  const json::Value* ct = id.find("cuda_toolkit");
  if (!ct || ct->kind != json::Value::Kind::String || ct->string.rfind(cuda + ".", 0) != 0) {
    problems.push_back("cuda_toolkit: engine built with '" + (ct ? ct->string : "<missing>") +
                       "', this binary uses CUDA " + cuda);
  }
  if (!id.at("plugins").items.empty()) problems.push_back("engine needs plugins; none are loaded");

  const std::string engine_path = paths.model_dir + "/" + em.at("engine").at("file").str();
  const std::vector<char> blob = read_bytes(engine_path);
  const std::string sha = sha256_hex(reinterpret_cast<const std::uint8_t*>(blob.data()), blob.size());
  if (sha != em.at("engine").at("sha256").str()) {
    problems.push_back("engine file " + engine_path + " is " + sha + ", manifest says " +
                       em.at("engine").at("sha256").str());
  }
  if (!problems.empty()) {
    std::string msg = "refusing engine (" + std::to_string(problems.size()) + " mismatches):";
    for (const auto& p : problems) msg += "\n  " + p;
    throw IdentityError(msg);
  }

  auto impl = std::make_unique<Impl>();
  impl->engine_sha = sha;
  impl->gpu = prop.name;
  impl->driver = driver;
  const auto& names = model.at("classes").at("names").items;
  for (const auto& n : names) impl->names.push_back(n.str());
  impl->width = impl->height = static_cast<int>(profile.at("export").at("input_size").num());

  impl->runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (!impl->runtime) throw std::runtime_error("createInferRuntime failed");
  impl->engine.reset(impl->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl->engine) throw std::runtime_error("deserializeCudaEngine failed for " + engine_path);
  impl->context.reset(impl->engine->createExecutionContext());
  if (!impl->context) throw std::runtime_error("createExecutionContext failed");

  if (impl->engine->getNbIOTensors() != 5) throw std::runtime_error("engine does not have 5 I/O tensors");
  for (const Io& io : kContract) {
    if (impl->engine->getTensorIOMode(io.name) != io.mode || impl->engine->getTensorDataType(io.name) != io.type) {
      throw std::runtime_error(std::string("engine tensor '") + io.name + "' breaks the deploy contract");
    }
  }
  nvinfer1::Dims img{};
  img.nbDims = 4;
  img.d[0] = 1;
  img.d[1] = 3;
  img.d[2] = impl->height;
  img.d[3] = impl->width;
  nvinfer1::Dims sz{};
  sz.nbDims = 2;
  sz.d[0] = 1;
  sz.d[1] = 2;
  if (!impl->context->setInputShape("images", img) || !impl->context->setInputShape("orig_target_sizes", sz)) {
    throw std::runtime_error("engine rejected the batch-1 input shapes");
  }
  for (int i = 0; i < 5; ++i) {
    const nvinfer1::Dims d = impl->context->getTensorShape(kContract[i].name);
    std::size_t n = 1;
    for (int k = 0; k < d.nbDims; ++k) {
      if (d.d[k] < 0) throw std::runtime_error(std::string("unresolved shape for ") + kContract[i].name);
      n *= static_cast<std::size_t>(d.d[k]);
    }
    if (i == 4) impl->queries = d.d[1];
    const std::size_t bytes = n * element_size(kContract[i].type);
    impl->device[i] = DeviceBuffer(bytes);
    impl->host[i] = PinnedBuffer(bytes);
    if (!impl->context->setTensorAddress(kContract[i].name, impl->device[i].get())) {
      throw std::runtime_error(std::string("setTensorAddress failed for ") + kContract[i].name);
    }
  }
  return std::unique_ptr<TrtDetector>(new TrtDetector(std::move(impl)));
}

TrtDetector::TrtDetector(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TrtDetector::~TrtDetector() = default;

std::vector<Detection> TrtDetector::infer(const float* input_nchw, int src_w, int src_h) {
  Impl& m = *impl_;
  const cudaStream_t s = m.stream.get();
  std::memcpy(m.host[0].get(), input_nchw, m.host[0].bytes());
  const std::int64_t sizes[2] = {src_w, src_h};  // orig_target_sizes is (width, height), C§5
  std::memcpy(m.host[1].get(), sizes, sizeof sizes);
  for (int i = 0; i < 2; ++i) {
    MM_CUDA_CHECK(cudaMemcpyAsync(m.device[i].get(), m.host[i].get(), m.host[i].bytes(), cudaMemcpyHostToDevice, s));
  }
  if (!m.context->enqueueV3(s)) throw std::runtime_error("enqueueV3 failed");
  for (int i = 2; i < 5; ++i) {
    MM_CUDA_CHECK(cudaMemcpyAsync(m.host[i].get(), m.device[i].get(), m.host[i].bytes(), cudaMemcpyDeviceToHost, s));
  }
  m.stream.sync();

  const auto* labels = static_cast<const std::int64_t*>(m.host[2].get());
  const auto* boxes = static_cast<const float*>(m.host[3].get());
  const auto* scores = static_cast<const float*>(m.host[4].get());
  std::vector<Detection> out(static_cast<std::size_t>(m.queries));
  for (std::int64_t q = 0; q < m.queries; ++q) {
    Detection& d = out[static_cast<std::size_t>(q)];
    d.class_id = static_cast<std::int32_t>(labels[q]);
    d.confidence = scores[q];
    for (int k = 0; k < 4; ++k) d.bbox_xyxy[k] = boxes[q * 4 + k];
  }
  return out;
}

int TrtDetector::input_width() const { return impl_->width; }
int TrtDetector::input_height() const { return impl_->height; }
const std::vector<std::string>& TrtDetector::class_names() const { return impl_->names; }
const std::string& TrtDetector::engine_sha256() const { return impl_->engine_sha; }
const std::string& TrtDetector::gpu_name() const { return impl_->gpu; }
const std::string& TrtDetector::driver() const { return impl_->driver; }
std::string TrtDetector::tensorrt_version() { return compiled_tensorrt(); }

}  // namespace motionman::gpu
