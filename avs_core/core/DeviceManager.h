#pragma once

#include <avisynth.h>
#include <vector>
#include <atomic>
#include <memory>

class InternalEnvironment;

enum {
    DEV_TYPE_CPU = 0,
    DEV_TYPE_CUDA,
};

class Device {
protected:
    InternalEnvironment* env;

public:
    const int device_type;
    const int device_index;

    unsigned __int64 memory_max;
    std::atomic<unsigned __int64> memory_used;

    Device(int type, int index, InternalEnvironment* env) :
		env(env),
        device_type(type),
        device_index(index),
		memory_max(0),
		memory_used(0)
    { }

    virtual ~Device() { }

    virtual int SetMemoryMax(int mem) = 0;
    virtual BYTE* Allocate(size_t sz) = 0;
    virtual void Free(BYTE* ptr) = 0;
    virtual const char* GetName() const = 0;
};

class DeviceManager {
private:
    class CPUDevice : public Device {
    public:
        CPUDevice(InternalEnvironment* env) : Device(DEV_TYPE_CPU, 0, env) { }

        virtual int SetMemoryMax(int mem);
        virtual BYTE* Allocate(size_t sz);
        virtual void Free(BYTE* ptr);
        virtual const char* GetName() const { return "CPU"; }
    };

#ifdef ENABLE_CUDA
    class CUDACPUDevice : public CPUDevice {
    public:
        CUDACPUDevice(InternalEnvironment* env) : CPUDevice(env) { }

        virtual BYTE* Allocate(size_t sz);
        virtual void Free(BYTE* ptr);
        virtual const char* GetName() const { return "CPU(CUDAAware)"; }
    };
#endif

    InternalEnvironment *env;
    std::unique_ptr<CPUDevice> cpuDevice;

#ifdef ENABLE_CUDA
    class CUDADevice : public Device {
        char name[32];
    public:
        CUDADevice(int n, InternalEnvironment* env);

        virtual int SetMemoryMax(int mem);
        virtual BYTE* Allocate(size_t sz);
        virtual void Free(BYTE* ptr);
        virtual const char* GetName() const { return name; }
    };

    std::vector<std::unique_ptr<CUDADevice>> cudaDevices;
#endif

public:
    DeviceManager(InternalEnvironment* env);
    ~DeviceManager() { }

    Device* GetDevice(int device_type, int device_index);

    Device* GetCPUDevice() { return GetDevice(0, 0); }

};
