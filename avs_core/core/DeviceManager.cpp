
#include "internal.h"
#include "DeviceManager.h"
#include "InternalEnvironment.h"
#include <avs/minmax.h>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include "LruCache.h"
#include "ThreadPool.h"
#include "AVSMap.h"

#define ENABLE_CUDA_COMPUTE_STREAM 0

#ifdef ENABLE_CUDA

#include <cuda_runtime_api.h>

#endif // #ifdef ENABLE_CUDA

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err__ = call; \
    if (err__ != cudaSuccess) { \
      env->ThrowError("[CUDA Error] %d: %s", err__, cudaGetErrorString(err__)); \
    } \
  } while (0)

int GetDeviceTypes(const PClip& child)
{
  if (child->GetVersion() < 5) {
    return DEV_TYPE_CPU;
  }
  int deviceflags = child->SetCacheHints(CACHE_GET_DEV_TYPE, 0);
  if (deviceflags == 0) {
    // if not implement CACHE_GET_DEVICE_TYPE, we assume CPU only filter.
    deviceflags = DEV_TYPE_CPU;
  }
  return deviceflags;
}

int GetTargetDeviceTypes(const PClip& clip)
{
  if (clip->GetVersion() < 5) {
    return DEV_TYPE_CPU;
  }
  int deviceflags = clip->SetCacheHints(CACHE_GET_CHILD_DEV_TYPE, 0);
  if (deviceflags == 0) {
    deviceflags = clip->SetCacheHints(CACHE_GET_DEV_TYPE, 0);
    if (deviceflags == 0) {
      // if not implement CACHE_GET_DEVICE_TYPE, we assume CPU only filter.
      deviceflags = DEV_TYPE_CPU;
    }
  }
  return deviceflags;
}

std::string DeviceTypesString(int devicetypes)
{
  std::vector<const char*> typesstr;
  if (devicetypes & DEV_TYPE_CPU) {
    typesstr.push_back("CPU");
  }
  if (devicetypes & DEV_TYPE_CUDA) {
    typesstr.push_back("CUDA");
  }
  std::ostringstream oss;
  for (int i = 0; i < (int)typesstr.size(); ++i) {
    if (i > 0) oss << ",";
    oss << typesstr[i];
  }
  return oss.str();
}

static void CheckDeviceTypes(const char* name, int devicetypes, const AVSValue& arr, IScriptEnvironment2* env)
{
  for (int i = 0; i < arr.ArraySize(); ++i) {
    const AVSValue& val = arr[i];
    if (val.IsClip()) {
      int childtypes = GetDeviceTypes(val.AsClip());
      if ((devicetypes & childtypes) == 0) {
        std::string parentdevstr = DeviceTypesString(devicetypes);
        std::string childdevstr = DeviceTypesString(childtypes);
        env->ThrowError(
          "Device unmatch: %s[%s] does not support [%s] frame",
          name, parentdevstr.c_str(), childdevstr.c_str());
      }
    }
    else if (val.IsArray()) {
      CheckDeviceTypes(name, devicetypes, val, env);
    }
  }
}

void CheckChildDeviceTypes(const PClip& clip, const char* name, const AVSValue& args, const char* const* argnames, IScriptEnvironment2* env)
{
  int deviceflags = GetTargetDeviceTypes(clip);
  if (args.IsArray()) {
    CheckDeviceTypes(name, deviceflags, args, env);
  }
  else {
    CheckDeviceTypes(name, deviceflags, AVSValue(&args, 1), env);
  }
}

class CPUDevice : public Device {
public:
  CPUDevice(InternalEnvironment* env) : Device(DEV_TYPE_CPU, 0, 0, env) { }

  virtual int SetMemoryMax(int mem)
  {
    // memory_max for CPU device is not implemented here.
    env->ThrowError("Not implemented ...");
    return 0;
  }

  virtual BYTE* Allocate(size_t size)
  {
#ifdef _DEBUG
    BYTE* data = new BYTE[size + 16];
    int *pInt = (int *)(data + size);
    pInt[0] = 0xDEADBEEF;
    pInt[1] = 0xDEADBEEF;
    pInt[2] = 0xDEADBEEF;
    pInt[3] = 0xDEADBEEF;

    static const BYTE filler[] = { 0x0A, 0x11, 0x0C, 0xA7, 0xED };
    BYTE* pByte = data;
    BYTE* q = pByte + size / 5 * 5;
    for (; pByte < q; pByte += 5)
    {
      pByte[0] = filler[0];
      pByte[1] = filler[1];
      pByte[2] = filler[2];
      pByte[3] = filler[3];
      pByte[4] = filler[4];
    }
    return data;
#else
    return new BYTE[size + 16];
#endif
  }

  virtual void Free(BYTE* ptr)
  {
    if (ptr != nullptr) {
      delete[] ptr;
    }
  }

  virtual const char* GetName() const { return "CPU"; }

  virtual void AddCompleteCallback(DeviceCompleteCallbackData cbdata)
  {
    // no need to delay, call immediately
    cbdata.cb(cbdata.user_data);
  }

  virtual std::unique_ptr<std::vector<DeviceCompleteCallbackData>> GetAndClearCallbacks()
  {
    return nullptr;
  }

  virtual void SetActiveToCurrentThread(InternalEnvironment* env)
  {
    // do nothing
  }

  virtual void* GetComputeStream()
  {
    return nullptr;
  }
};

#ifdef ENABLE_CUDA
class CUDACPUDevice : public CPUDevice {
public:
  CUDACPUDevice(InternalEnvironment* env) : CPUDevice(env) { }

  virtual BYTE* Allocate(size_t size)
  {
    unsigned int flags = cudaHostAllocMapped;
    BYTE* data = nullptr;
#ifdef _DEBUG
    CUDA_CHECK(cudaHostAlloc((void**)&data, size + 16, flags));
    int *pInt = (int *)(data + size);
    pInt[0] = 0xDEADBEEF;
    pInt[1] = 0xDEADBEEF;
    pInt[2] = 0xDEADBEEF;
    pInt[3] = 0xDEADBEEF;

    static const BYTE filler[] = { 0x0A, 0x11, 0x0C, 0xA7, 0xED };
    BYTE* pByte = data;
    BYTE* q = pByte + size / 5 * 5;
    for (; pByte < q; pByte += 5)
    {
      pByte[0] = filler[0];
      pByte[1] = filler[1];
      pByte[2] = filler[2];
      pByte[3] = filler[3];
      pByte[4] = filler[4];
    }
#else
    CUDA_CHECK(cudaHostAlloc((void**)&data, size + 16, flags));
#endif
    return data;
  }

  virtual void Free(BYTE* ptr)
  {
    if (ptr != nullptr) {
      CUDA_CHECK(cudaFreeHost(ptr));
    }
  }

  virtual const char* GetName() const { return "CPU(CUDAAware)"; }
};
#endif

#ifdef ENABLE_CUDA
class CUDADevice : public Device {
  class ScopedCUDADevice
  {
    int old_device;
    int tgt_device;
  public:
    ScopedCUDADevice(int device_index, IScriptEnvironment* env)
      : tgt_device(device_index)
    {
      CUDA_CHECK(cudaGetDevice(&old_device));
      if (tgt_device != old_device) {
        CUDA_CHECK(cudaSetDevice(tgt_device));
      }
    }
    ~ScopedCUDADevice()
    {
      if (tgt_device != old_device) {
        cudaSetDevice(old_device);
      }
    }
  };

  char name[32];

  std::mutex mutex;
  std::vector<DeviceCompleteCallbackData> callbacks;

#if ENABLE_CUDA_COMPUTE_STREAM
  cudaStream_t computeStream;
  cudaEvent_t computeEvent;
#endif

public:
  CUDADevice(int id, int n, InternalEnvironment* env) :
    Device(DEV_TYPE_CUDA, id, n, env)
  {
    sprintf_s(name, "CUDA %d", n);

    SetMemoryMax(768); // start with 768MB
#if ENABLE_CUDA_COMPUTE_STREAM
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaEventCreate(&computeEvent));
#endif
  }

  ~CUDADevice()
  {
#if ENABLE_CUDA_COMPUTE_STREAM
    cudaStreamDestroy(computeStream);
    cudaEventDestroy(computeEvent);
#endif
  }

  virtual int SetMemoryMax(int mem)
  {
    if (mem > 0) {
      unsigned __int64 requested = mem * 1048576ull;
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
      unsigned __int64 mem_limit = prop.totalGlobalMem;
      memory_max = clamp(requested, 64 * 1024 * 1024ull, mem_limit - 128 * 1024 * 1024ull);
    }
    return (int)(memory_max / 1048576ull);
  }

  virtual BYTE* Allocate(size_t size)
  {
    BYTE* data = nullptr;
    ScopedCUDADevice d(device_index, env);
    CUDA_CHECK(cudaMalloc((void**)&data, size));
    return data;
  }

  virtual void Free(BYTE* ptr)
  {
    if (ptr != NULL) {
      ScopedCUDADevice d(device_index, env);
      CUDA_CHECK(cudaFree(ptr));
    }
  }

  virtual const char* GetName() const { return name; }

  virtual void AddCompleteCallback(DeviceCompleteCallbackData cbdata)
  {
    std::lock_guard<std::mutex> lock(mutex);

    callbacks.push_back(cbdata);
  }

  virtual std::unique_ptr<std::vector<DeviceCompleteCallbackData>> GetAndClearCallbacks()
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (callbacks.size() > 0) {
      auto *ret = new std::vector<DeviceCompleteCallbackData>(std::move(callbacks));
      callbacks.clear();
      return std::unique_ptr<std::vector<DeviceCompleteCallbackData>>(ret);
    }

    return nullptr;
  }

  virtual void SetActiveToCurrentThread(InternalEnvironment* env)
  {
    CUDA_CHECK(cudaSetDevice(device_index));
  }

  virtual void* GetComputeStream() {
#if ENABLE_CUDA_COMPUTE_STREAM
    return computeStream;
#else
    return nullptr;
#endif
  }

  void MakeStreamWaitCompute(cudaStream_t stream, InternalEnvironment* env)
  {
#if ENABLE_CUDA_COMPUTE_STREAM
    std::lock_guard<std::mutex> lock(mutex);

    CUDA_CHECK(cudaEventRecord(computeEvent, computeStream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, computeEvent, 0));
#endif
  }
};
#endif

DeviceManager::DeviceManager(InternalEnvironment* env) :
  env(env)
{
  // 0 is CPU device, start from 1 for other devices.
  int next_device_id = 1;

#ifdef ENABLE_CUDA
  int cuda_device_count = 0;
  if (cudaGetDeviceCount(&cuda_device_count) == cudaSuccess) {
    for (int i = 0; i < cuda_device_count; ++i) {
      cudaDevices.emplace_back(new CUDADevice(next_device_id++, i, env));
    }
  }
  // do not modify CUDADevices after this since it causes pointer change

  // if cuda capable device is available, we always allocate with page locked memory.
  // is it OK ???
  cpuDevice = std::unique_ptr<Device>((cuda_device_count > 0)
      ? new CUDACPUDevice(env)
      : new CPUDevice(env));
#else // ENABLE_CUDA
  cpuDevice = std::unique_ptr<CPUDevice>(new CPUDevice(env));
#endif // ENABLE_CUDA

  numDevices = next_device_id;
}

Device* DeviceManager::GetDevice(int device_type, int device_index)
{
  switch (device_type) {

  case DEV_TYPE_CPU:
    return cpuDevice.get();

#ifdef ENABLE_CUDA
  case DEV_TYPE_CUDA:
    if (device_index < 0) {
      env->ThrowError("Invalid device index %d", device_index);
    }
    if (cudaDevices.size() == 0) {
      env->ThrowError("No CUDA devices ...");
    }
    // wrap index
    device_index %= (int)cudaDevices.size();
    return cudaDevices[device_index].get();
#endif // #ifdef ENABLE_CUDA

  default:
    env->ThrowError("Not supported memory type %d", device_type);
  }
  return nullptr;
}

class QueuePrefetcher
{
  PClip child;
  VideoInfo vi;

  int prefetchFrames;
  int numThreads;

  ThreadPool* threadPool;
  Device* device;

  typedef LruCache<size_t, PVideoFrame> CacheType;

  std::shared_ptr<CacheType> videoCache;

  std::mutex mutex; // do not acceess to videoCache during locked by this mutex
  std::deque<std::pair<size_t, CacheType::handle>> prefetchQueue;
  int numWorkers;
  std::exception_ptr workerException;
  bool workerExceptionPresent;

  static AVSValue ThreadWorker_(IScriptEnvironment2* env, void* data)
  {
    return static_cast<QueuePrefetcher*>(data)->ThreadWorker(
      static_cast<InternalEnvironment*>(env));
  }
    
  AVSValue ThreadWorker(InternalEnvironment* env)
  {
    device->SetActiveToCurrentThread(env);

    while (true) {
      std::pair<size_t, CacheType::handle> work;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (prefetchQueue.size() == 0) {
          // there are no prefetch work
          --numWorkers;
          break;
        }
        work = prefetchQueue.front();
        prefetchQueue.pop_front();
      }

      try
      {
        work.second.first->value = child->GetFrame((int)work.first, env);
        videoCache->commit_value(&work.second);
      }
      catch (...)
      {
        videoCache->rollback(&work.second);

        std::lock_guard<std::mutex> lock(mutex);
        workerException = std::current_exception();
        workerExceptionPresent = true;
      }
    }

    return AVSValue();
  }

  int SchedulePrefetch(int currentN, int prefetchStart, InternalEnvironment* env)
  {
    int numQueued = 0;
    int n = prefetchStart;
    for (; n < currentN + prefetchFrames && n < vi.num_frames; ++n)
    {
      PVideoFrame result;
      CacheType::handle cacheHandle;
			bool increaseCache = true;
      switch (videoCache->lookup(n, &cacheHandle, false, result, increaseCache))
      {
        case LRU_LOOKUP_NOT_FOUND:
        {
            std::lock_guard<std::mutex> lock(mutex);
            prefetchQueue.emplace_back(n, cacheHandle);
            ++numQueued;
            break;
        }
        case LRU_LOOKUP_FOUND_AND_READY:      // Fall-through intentional
        case LRU_LOOKUP_NO_CACHE:             // Fall-through intentional
        case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:
        {
            break;
        }
        default:
        {
            env->ThrowError("Invalid Program");
            break;
        }
      }
    }
    if (numQueued > 0) {
        std::lock_guard<std::mutex> lock(mutex);
        for (; numWorkers < numThreads && numQueued > 0; ++numWorkers, --numQueued) {
            threadPool->QueueJob(ThreadWorker_, this, env, nullptr);
        }
    }
    return n;
  }

public:
  QueuePrefetcher(PClip child, int prefetchFrames, int numThreads, Device* device, InternalEnvironment* env) :
      child(child),
      vi(child->GetVideoInfo()),
      prefetchFrames(prefetchFrames),
      numThreads(numThreads),
      threadPool(NULL),
      device(device),
      videoCache(new CacheType(prefetchFrames*2, CACHE_DEFAULT)),
      numWorkers(0),
      workerExceptionPresent(false)
  {
    threadPool = env->NewThreadPool(numThreads);
  }

  ~QueuePrefetcher()
  {
    // finish threadpool
    threadPool->Finish();

    // cancel queue
    while (prefetchQueue.size() > 0) {
      videoCache->rollback(&prefetchQueue.front().second);
      prefetchQueue.pop_front();
    }
  }

  VideoInfo GetVideoInfo() const { return vi; }

  PVideoFrame GetFrame(int n, InternalEnvironment* env)
  {
    if (prefetchFrames == 0) {
      return child->GetFrame(n, env);
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      if (workerExceptionPresent)
      {
          std::rethrow_exception(workerException);
      }
    }

    // Prefetch 1
    int prefetchPos = SchedulePrefetch(n, n, env);

    // Get requested frame
    PVideoFrame result;
    CacheType::handle cacheHandle;
		bool increaseCache = true;
    // fill result if LRU_LOOKUP_FOUND_AND_READY
    switch (videoCache->lookup(n, &cacheHandle, true, result, increaseCache))
    {
      case LRU_LOOKUP_FOUND_AND_READY:
      {
          break;
      }
      case LRU_LOOKUP_NO_CACHE:
      {
          result = child->GetFrame(n, env);
          break;
      }
      case LRU_LOOKUP_NOT_FOUND:             // Fall-through intentional
      case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:    // Fall-through intentional
      default:
      {
          env->ThrowError("Invalid Program");
          break;
      }
    }

    // Prefetch 2
    SchedulePrefetch(n, prefetchPos, env);

    return result;
  }
};

class FrameTransferEngine
{
public:
    QueuePrefetcher& child;
    VideoInfo vi;

  Device* upstreamDevice;
  Device* downstreamDevice;

  FrameTransferEngine(QueuePrefetcher& child, Device* upstreamDevice, Device* downstreamDevice) :
        child(child),
        vi(child.GetVideoInfo()),
    upstreamDevice(upstreamDevice),
    downstreamDevice(downstreamDevice)
  { }

  virtual ~FrameTransferEngine() { }

  virtual PVideoFrame GetFrame(int n, InternalEnvironment* env) = 0;
};

class CUDAFrameTransferEngine : public FrameTransferEngine
{
    typedef LruCache<size_t, PVideoFrame> CacheType;

    struct QueueItem {
        size_t n;
        PVideoFrame src;
        CacheType::handle cacheHandle;
        cudaEvent_t completeEvent;
        std::unique_ptr<std::vector<DeviceCompleteCallbackData>> completeCallbacks;
    };

  int prefetchFrames;

    std::shared_ptr<CacheType> videoCache;

  std::mutex mutex;
  cudaStream_t stream;
  std::deque<QueueItem> prefetchQueue;

  cudaMemcpyKind GetMemcpyKind()
  {
    if (upstreamDevice->device_type == DEV_TYPE_CPU && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Host to Device
      return cudaMemcpyHostToDevice;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CPU) {
      // Device to Host
      return cudaMemcpyDeviceToHost;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Device to Device
      return cudaMemcpyDeviceToDevice;
    }
    _ASSERT(false);
    return cudaMemcpyDefault;
  }

  void ExecuteCallbacks(const std::vector<DeviceCompleteCallbackData>* callbacks)
  {
    if (callbacks != nullptr) {
      for (auto cbdata : *callbacks) {
        cbdata.cb(cbdata.user_data);
      }
    }
  }

  void TransferFrameData(PVideoFrame& dst, PVideoFrame& src, bool async, InternalEnvironment* env)
  {
    VideoFrameBuffer* srcvfb = src->GetFrameBuffer();
    VideoFrameBuffer* dstvfb = dst->GetFrameBuffer();

    const BYTE* srcptr = srcvfb->GetReadPtr();
    BYTE* dstptr = dstvfb->GetWritePtr();
    int sz = srcvfb->GetDataSize();
    cudaMemcpyKind kind = GetMemcpyKind();

    if (async) {
      CUDA_CHECK(cudaMemcpyAsync(dstptr, srcptr, sz, kind, stream));
    }
    else {
      CUDA_CHECK(cudaMemcpy(dstptr, srcptr, sz, kind));
    }
  }

  PVideoFrame GetFrameImmediate(int n, InternalEnvironment* env)
  {
    PVideoFrame src = child.GetFrame(n, env);
    PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
    TransferFrameData(dst, src, false, env);

    AVSMap* mapv = env->GetAVSMap(dst);
    for (auto it = mapv->begin(), end = mapv->end(); it != end; ++it) {
      if (it->second.IsFrame()) {
        PVideoFrame src = it->second.GetFrame();
        PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
        TransferFrameData(dst, src, false, env);
        it->second = dst;
      }
    }

    ExecuteCallbacks(downstreamDevice->GetAndClearCallbacks().get());

    return dst;
  }

  QueueItem SetupTransfer(int n, CacheType::handle& cacheHandle, InternalEnvironment* env)
  {
    QueueItem item = { (size_t)n, child.GetFrame(n, env), cacheHandle, nullptr, nullptr };
    cacheHandle.first->value = env->GetOnDeviceFrame(item.src, downstreamDevice);
    CUDA_CHECK(cudaEventCreate(&item.completeEvent));

    item.completeCallbacks = upstreamDevice->GetAndClearCallbacks();

    if (upstreamDevice->device_type == DEV_TYPE_CUDA) {
      static_cast<CUDADevice*>(upstreamDevice)->MakeStreamWaitCompute(stream, env);
    }

    TransferFrameData(cacheHandle.first->value, item.src, true, env);

    AVSMap* mapv = env->GetAVSMap(cacheHandle.first->value);
    for (auto it = mapv->begin(), end = mapv->end(); it != end; ++it) {
      if (it->second.IsFrame()) {
        PVideoFrame src = it->second.GetFrame();
        PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
        TransferFrameData(dst, src, true, env);
        it->second = dst;
      }
    }

    CUDA_CHECK(cudaEventRecord(item.completeEvent, stream));

    return std::move(item);
  }

  int SchedulePrefetch(int currentN, int prefetchStart, InternalEnvironment* env)
  {
    int numQueued = 0;
    int n = prefetchStart;
    for (; n < currentN + prefetchFrames && n < vi.num_frames; ++n)
    {
      PVideoFrame result;
      CacheType::handle cacheHandle;
			bool increaseCache = true;
      switch (videoCache->lookup(n, &cacheHandle, false, result, increaseCache))
      {
        case LRU_LOOKUP_NOT_FOUND:
        {
          try {
            prefetchQueue.emplace_back(SetupTransfer(n, cacheHandle, env));
          }
          catch(...) {
            videoCache->rollback(&cacheHandle);
            throw;
          }
          break;
        }
        case LRU_LOOKUP_FOUND_AND_READY:      // Fall-through intentional
        case LRU_LOOKUP_NO_CACHE:             // Fall-through intentional
        case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:
        {
            break;
        }
        default:
        {
          env->ThrowError("Invalid Program");
            break;
        }
      }
    }
    return n;
  }

  void FinishCompleted(InternalEnvironment* env)
  {
    while (prefetchQueue.size() > 0) {
      QueueItem& item = prefetchQueue.front();
      cudaError_t err = cudaEventQuery(item.completeEvent);
      if (err == cudaErrorNotReady) {
        break;
      }
      try {
        CUDA_CHECK(err);

        // transfer is complete
        CUDA_CHECK(cudaEventDestroy(item.completeEvent));
        ExecuteCallbacks(item.completeCallbacks.get());

        videoCache->commit_value(&item.cacheHandle);
      }
      catch (...) {
        videoCache->rollback(&item.cacheHandle);
        throw;
      }
      prefetchQueue.pop_front();
    }
  }

  PVideoFrame WaitUntil(int n, InternalEnvironment* env)
  {
    while (prefetchQueue.size() > 0) {
      QueueItem& item = prefetchQueue.front();
      try {
        CUDA_CHECK(cudaEventSynchronize(item.completeEvent));
        CUDA_CHECK(cudaEventDestroy(item.completeEvent));
        ExecuteCallbacks(item.completeCallbacks.get());

        PVideoFrame frame = item.cacheHandle.first->value; // fill before Commit !!!

        videoCache->commit_value(&item.cacheHandle);

        if (item.n == n) {
          prefetchQueue.pop_front();
          return frame;
        }

        prefetchQueue.pop_front();
      }
      catch (...) {
        videoCache->rollback(&item.cacheHandle);
        throw;
      }
    }
    env->ThrowError("invalid program");
    return PVideoFrame();
  }

  void CheckDevicePair(InternalEnvironment* env)
  {
    if (upstreamDevice->device_type == DEV_TYPE_CPU && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Host to Device
      return;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CPU) {
      // Device to Host
      return;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Device to Device
      return;
    }
    env->ThrowError("[CUDAFrameTransferEngine] invalid device pair up:%s down:%d",
      upstreamDevice->GetName(), downstreamDevice->GetName());
  }

public:
  CUDAFrameTransferEngine(QueuePrefetcher& child, Device* upstreamDevice, Device* downstreamDevice, int prefetchFrames, InternalEnvironment* env) :
    FrameTransferEngine(child, upstreamDevice, downstreamDevice),
    prefetchFrames(prefetchFrames),
    videoCache(new CacheType(prefetchFrames*2, CACHE_DEFAULT)),
    stream(nullptr)
  {
    CheckDevicePair(env);

    // note: stream belongs to a device
    upstreamDevice->SetActiveToCurrentThread(env);
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  ~CUDAFrameTransferEngine()
  {
    for (auto& item : prefetchQueue) {
      cudaEventSynchronize(item.completeEvent);
      cudaEventDestroy(item.completeEvent);
      ExecuteCallbacks(item.completeCallbacks.get());
      videoCache->commit_value(&item.cacheHandle);
    }
    prefetchQueue.clear();

    cudaStreamDestroy(stream);
  }

  virtual PVideoFrame GetFrame(int n, InternalEnvironment* env)
  {
    // Giant lock. This is OK because all transfer is done asynchronously
    std::lock_guard<std::mutex> lock(mutex);

    // set upstream device
    upstreamDevice->SetActiveToCurrentThread(env);

    if (prefetchFrames == 0) {
      return GetFrameImmediate(n, env);
    }

    FinishCompleted(env);

    // Prefetch 1
    int prefetchPos = SchedulePrefetch(n, n, env);

    // Get requested frame
    PVideoFrame result;
    CacheType::handle cacheHandle;
		bool increaseCache = true;
    // fill result if LRU_LOOKUP_FOUND_AND_READY
    switch (videoCache->lookup(n, &cacheHandle, false, result, increaseCache))
    {
      case LRU_LOOKUP_FOUND_AND_READY:
      {
          break;
      }
      case LRU_LOOKUP_NO_CACHE:
      {
          result = GetFrameImmediate(n, env);
          break;
      }
      case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:
      {
        // now transferring, wait until completion
        result = WaitUntil(n, env);
        break;
      }
      default:
      {
        env->ThrowError("Invalid Program");
          break;
      }
    }

    // Prefetch 2
    SchedulePrefetch(n, prefetchPos, env);

    // set downstreamdevice
    downstreamDevice->SetActiveToCurrentThread(env);

    return result;
  }
};

FrameTransferEngine* CreateTransferEngine(QueuePrefetcher& child,
  Device* upstreamDevice, Device* downstreamDevice, int prefetchFrames, InternalEnvironment* env)
{
  if (upstreamDevice->device_type == DEV_TYPE_CPU && downstreamDevice->device_type == DEV_TYPE_CUDA) {
    // CPU to CUDA
    return new CUDAFrameTransferEngine(child, upstreamDevice, downstreamDevice, prefetchFrames, env);
  }
  if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CPU) {
    // CUDA to CPU
    return new CUDAFrameTransferEngine(child, upstreamDevice, downstreamDevice, prefetchFrames, env);
  }
  if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CUDA) {
    // CUDA to CUDA
    return new CUDAFrameTransferEngine(child, upstreamDevice, downstreamDevice, prefetchFrames, env);
  }
  env->ThrowError("Not supported frame data transfer. up:%s down:%d",
    upstreamDevice->GetName(), downstreamDevice->GetName());
  return nullptr;
}

class OnDevice : public GenericVideoFilter
{
  Device* upstreamDevice;
  int prefetchFrames;

  QueuePrefetcher prefetcher;

  std::mutex mutex;
  std::map<Device*, std::unique_ptr<FrameTransferEngine>> transferEngines;

  FrameTransferEngine* GetOrCreateTransferEngine(Device* downstreamDevice, InternalEnvironment* env)
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = transferEngines.find(downstreamDevice);
    if (it != transferEngines.end()) {
      return it->second.get();
    }
    auto pEngine = CreateTransferEngine(prefetcher, upstreamDevice, downstreamDevice, prefetchFrames, env);
    transferEngines[downstreamDevice] = std::unique_ptr<FrameTransferEngine>(pEngine);
    return pEngine;
  }

public:
  OnDevice(PClip child, int prefetchFrames, Device* upstreamDevice, InternalEnvironment* env) :
    GenericVideoFilter(child),
    upstreamDevice(upstreamDevice),
    prefetchFrames(prefetchFrames),
    prefetcher(child, prefetchFrames ? 2 : 0, 1, upstreamDevice, env)
  { }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);
    Device* downstreamDevice = env->SetCurrentDevice(upstreamDevice);

    if (downstreamDevice == nullptr) {
      env->ThrowError("This thread is not created by AviSynth. It is not allowed to call GetFrame on this thread ...");
    }
    
    if (downstreamDevice == upstreamDevice) {
      // shortcut
      return child->GetFrame(n, env);
    }

    // Get frame via transfer engine
    PVideoFrame frame = GetOrCreateTransferEngine(downstreamDevice, env)->GetFrame(n, env);

    env->SetCurrentDevice(downstreamDevice);
    return frame;
  }

  void __stdcall GetAudio(void* buf, __int64 start, __int64 count, IScriptEnvironment* env_)
  {
    InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);
    Device* downstreamDevice = env->SetCurrentDevice(upstreamDevice);
    child->GetAudio(buf, start, count, env);
    env->SetCurrentDevice(downstreamDevice);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range)
  {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return 0xFF; // any devices
    }
    if (cachehints == CACHE_GET_CHILD_DEV_TYPE) {
      return upstreamDevice->device_type;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    int upstreamType = (int)(size_t)user_data;
    InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);

    PClip clip = args[0].AsClip();
    int numPrefetch = args[1].Defined() ? args[1].AsInt() : 4;
    int upstreamIndex = (args.ArraySize() >= 3 && args[2].Defined()) ? args[2].AsInt() : 0;

    if (numPrefetch < 0) {
      numPrefetch = 0;
    }

    switch (upstreamType) {
    case DEV_TYPE_CPU:
      return new OnDevice(clip, numPrefetch, env->GetDevice(DEV_TYPE_CPU, 0), env);
    case DEV_TYPE_CUDA:
      return new OnDevice(clip, numPrefetch, env->GetDevice(DEV_TYPE_CUDA, upstreamIndex), env);
    }

    env->ThrowError("Not supported device ...");
    return AVSValue();
  }

  static AVSValue __cdecl Eval(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
     int upstreamType = (int)(size_t)user_data;
     InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);

     const char* str = args[0].AsString();
     int upstreamIndex = (args.ArraySize() >= 2 && args[1].Defined()) ? args[1].AsInt() : 0;

     Device* upstreamDevice = nullptr;
     switch (upstreamType) {
     case DEV_TYPE_CPU:
        upstreamDevice = env->GetDevice(DEV_TYPE_CPU, 0);
        break;
     case DEV_TYPE_CUDA:
        upstreamDevice = env->GetDevice(DEV_TYPE_CUDA, upstreamIndex);
        break;
     default:
        env->ThrowError("Not supported device ...");
        break;
     }

     Device* downstreamDevice = env->SetCurrentDevice(upstreamDevice);

     if (downstreamDevice == nullptr) {
        env->ThrowError("This thread is not created by AviSynth. It is not allowed to invoke script on this thread ...");
     }

     try {
        ScriptParser parser(env, args[0].AsString(), "EvalOnDevice");
        PExpression exp = parser.Parse();
        AVSValue ret = exp->Evaluate(env);
        env->SetCurrentDevice(downstreamDevice);
        return ret;
     }
     catch (...) {
        env->SetCurrentDevice(downstreamDevice);
        throw;
     }
  }
};

extern const AVSFunction Device_filters[] = {
  { "OnCPU", BUILTIN_FUNC_PREFIX, "c[num_prefetch]i", OnDevice::Create, (void*)DEV_TYPE_CPU },
  { "OnCUDA", BUILTIN_FUNC_PREFIX, "c[num_prefetch]i[device_index]i", OnDevice::Create, (void*)DEV_TYPE_CUDA },
  { "EvalOnCPU", BUILTIN_FUNC_PREFIX, "s", OnDevice::Eval, (void*)DEV_TYPE_CPU },
  { "EvalOnCUDA", BUILTIN_FUNC_PREFIX, "s[device_index]i", OnDevice::Eval, (void*)DEV_TYPE_CUDA },
  { 0 }
};
