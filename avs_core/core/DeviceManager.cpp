
#include "internal.h"
#include "DeviceManager.h"
#include "InternalEnvironment.h"
#include <avs/minmax.h>
#include <deque>
#include <map>
#include <mutex>
#include "LruCache.h"
#include "ThreadPool.h"

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

DeviceManager::DeviceManager(InternalEnvironment* env) :
	env(env)
{

#ifdef ENABLE_CUDA
    int cuda_device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    for (int i = 0; i < cuda_device_count; ++i) {
        cudaDevices.emplace_back(new CUDADevice(i, env));
    }
    // do not modify CUDADevices after this since it causes pointer change

    // if cuda capable device is available, we always allocate with page locked memory.
    // is it OK ???
    cpuDevice = std::unique_ptr<CPUDevice>((cuda_device_count > 0)
        ? new CUDACPUDevice(env)
        : new CPUDevice(env));
#else // ENABLE_CUDA
    cpuDevice = std::unique_ptr<CPUDevice>(new CPUDevice(env));
#endif // ENABLE_CUDA

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
		// wrap index
		device_index %= (int)cudaDevices.size();
        return cudaDevices[device_index].get();
#endif // #ifdef ENABLE_CUDA

    default:
		env->ThrowError("Not supported memory type %d", device_type);
    }
    return nullptr;
}

int DeviceManager::CPUDevice::SetMemoryMax(int mem)
{
    // memory_max for CPU device is not implemented here.
	env->ThrowError("Not implemented ...");
    return 0;
}

BYTE* DeviceManager::CPUDevice::Allocate(size_t size)
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
    return new BYTE[size];
#endif
}

void DeviceManager::CPUDevice::Free(BYTE* ptr)
{
    if (ptr != nullptr) {
        delete[] ptr;
    }
}

#ifdef ENABLE_CUDA

BYTE* DeviceManager::CUDACPUDevice::Allocate(size_t size)
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
    CUDA_CHECK(cudaHostAlloc((void**)&data, size, flags));
#endif
    return data;
}

void DeviceManager::CUDACPUDevice::Free(BYTE* ptr)
{
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

DeviceManager::CUDADevice::CUDADevice(int n, InternalEnvironment* env) :
    Device(DEV_TYPE_CUDA, n, env)
{
    sprintf_s(name, "CUDA %d", n);

    SetMemoryMax(768); // start with 768MB
}

int DeviceManager::CUDADevice::SetMemoryMax(int mem)
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

BYTE* DeviceManager::CUDADevice::Allocate(size_t size)
{
    BYTE* data = nullptr;
    CUDA_CHECK(cudaSetDevice(device_index));
    CUDA_CHECK(cudaMalloc((void**)&data, size));
    return data;
}

void DeviceManager::CUDADevice::Free(BYTE* ptr)
{
    if (ptr != NULL) {
        CUDA_CHECK(cudaSetDevice(device_index));
        CUDA_CHECK(cudaFree(ptr));
    }
}

#endif // #ifdef ENABLE_CUDA

class QueuePrefetcher
{
    PClip child;
    VideoInfo vi;

    int prefetchFrames;
    int numThreads;

    ThreadPool threadPool;

    typedef LruCache<size_t, PVideoFrame> CacheType;

    std::shared_ptr<CacheType> videoCache;

    std::mutex mutex; // do not acceess to videoCache during locked by this mutex
    std::deque<std::pair<size_t, CacheType::handle>> prefetchQueue;
    int numWorkers;
    std::exception_ptr workerException;
    bool workerExceptionPresent;

    static AVSValue ThreadWorker_(IScriptEnvironment2* env, void* data)
    {
        return static_cast<QueuePrefetcher*>(data)->ThreadWorker(env);
    }
    
    AVSValue ThreadWorker(IScriptEnvironment2* env)
    {
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
            switch (videoCache->lookup(n, &cacheHandle, false, result))
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
                assert(0);
                break;
            }
            }
        }
        if (numQueued > 0) {
            std::lock_guard<std::mutex> lock(mutex);
            for (; numWorkers < numThreads && numQueued > 0; ++numWorkers, --numQueued) {
                threadPool.QueueJob(ThreadWorker_, this, env, nullptr);
            }
        }
        return n;
    }

public:
    QueuePrefetcher(PClip child, int prefetchFrames, int numThreads, InternalEnvironment* env) :
        child(child),
        vi(child->GetVideoInfo()),
        prefetchFrames(prefetchFrames),
        numThreads(numThreads),
        threadPool(numThreads),
        videoCache(new CacheType(prefetchFrames*2)),
        numWorkers(0),
        workerExceptionPresent(false)
    { }

    VideoInfo GetVideoInfo() const { return vi; }

    PVideoFrame GetFrame(int n, InternalEnvironment* env)
    {
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
        // fill result if LRU_LOOKUP_FOUND_AND_READY
        switch (videoCache->lookup(n, &cacheHandle, true, result))
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
            assert(0);
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

	PVideoFrame GetFrameImmediate(int n, InternalEnvironment* env)
	{
		PVideoFrame src = child.GetFrame(n, env);
		PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);

		VideoFrameBuffer* srcvfb = src->GetFrameBuffer();
		VideoFrameBuffer* dstvfb = dst->GetFrameBuffer();

		const BYTE* srcptr = srcvfb->GetReadPtr();
		BYTE* dstptr = dstvfb->GetWritePtr();
		int sz = srcvfb->GetDataSize();
		cudaMemcpyKind kind = GetMemcpyKind();

		CUDA_CHECK(cudaMemcpy(dstptr, srcptr, sz, kind));

		return dst;
	}

	QueueItem SetupTransfer(int n, CacheType::handle& cacheHandle, InternalEnvironment* env)
	{
		QueueItem item = { (size_t)n, child.GetFrame(n, env), cacheHandle, nullptr };
		cacheHandle.first->value = env->GetOnDeviceFrame(item.src, downstreamDevice);
		CUDA_CHECK(cudaEventCreate(&item.completeEvent));

		VideoFrameBuffer* srcvfb = item.src->GetFrameBuffer();
		VideoFrameBuffer* dstvfb = cacheHandle.first->value->GetFrameBuffer();

		const BYTE* srcptr = srcvfb->GetReadPtr();
		BYTE* dstptr = dstvfb->GetWritePtr();
		int sz = srcvfb->GetDataSize();
		cudaMemcpyKind kind = GetMemcpyKind();

		CUDA_CHECK(cudaMemcpyAsync(dstptr, srcptr, sz, kind, stream));
		CUDA_CHECK(cudaEventRecord(item.completeEvent, stream));

		return item;
	}

    int SchedulePrefetch(int currentN, int prefetchStart, InternalEnvironment* env)
    {
        int numQueued = 0;
        int n = prefetchStart;
        for (; n < currentN + prefetchFrames && n < vi.num_frames; ++n)
        {
            PVideoFrame result;
            CacheType::handle cacheHandle;
            switch (videoCache->lookup(n, &cacheHandle, false, result))
            {
            case LRU_LOOKUP_NOT_FOUND:
            {
				try {
					prefetchQueue.push_back(SetupTransfer(n, cacheHandle, env));
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
                assert(0);
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
		videoCache(new CacheType(prefetchFrames*2)),
		stream(nullptr)
	{
		CheckDevicePair(env);

		CUDA_CHECK(cudaStreamCreate(&stream));
	}

	~CUDAFrameTransferEngine()
	{
		for (auto& item : prefetchQueue) {
			cudaEventSynchronize(item.completeEvent);
			cudaEventDestroy(item.completeEvent);
			videoCache->commit_value(&item.cacheHandle);
		}
		prefetchQueue.clear();

		cudaStreamDestroy(stream);
	}

	virtual PVideoFrame GetFrame(int n, InternalEnvironment* env)
    {
		// Giant lock. This is OK because all transfer is done asynchronously
		std::lock_guard<std::mutex> lock(mutex);

		FinishCompleted(env);

        // Prefetch 1
        int prefetchPos = SchedulePrefetch(n, n, env);

        // Get requested frame
        PVideoFrame result;
        CacheType::handle cacheHandle;
        // fill result if LRU_LOOKUP_FOUND_AND_READY
        switch (videoCache->lookup(n, &cacheHandle, false, result))
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
            assert(0);
            break;
        }
        }

		// Prefetch 2
		SchedulePrefetch(n, prefetchPos, env);

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
        prefetcher(child, prefetchFrames ? 2 : 0, 1, env)
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
};

extern const AVSFunction Device_filters[] = {
	{ "OnCPU", BUILTIN_FUNC_PREFIX, "c[num_prefetch]i", OnDevice::Create, (void*)DEV_TYPE_CPU },
	{ "OnCUDA", BUILTIN_FUNC_PREFIX, "c[num_prefetch]i[device_index]i", OnDevice::Create, (void*)DEV_TYPE_CUDA },
	{ 0 }
};
