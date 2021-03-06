// Avisynth v2.6.  Copyright 2002-2009 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

#include <avisynth.h>
#include "../core/internal.h"
#include "InternalEnvironment.h"
#include "./parser/script.h"
#include <avs/minmax.h>
#include <avs/alignment.h>
#include "strings.h"
#include <avs/cpuid.h>
#include <unordered_set>
#include "bitblt.h"
#include "FilterConstructor.h"
#include "PluginManager.h"
#include "MappedList.h"
#include <vector>
#include <iostream>
#include <fstream>

#include <avs/win.h>
#include <objbase.h>

#include <string>
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include "MTGuard.h"
#include "cache.h"
#include <clocale>

#include "FilterGraph.h"
#include "DeviceManager.h"
#include "AVSMap.h"

#ifndef YieldProcessor // low power spin idle
  #define YieldProcessor() __nop(void)
#endif

extern const AVSFunction Audio_filters[], Combine_filters[], Convert_filters[],
                   Convolution_filters[], Edit_filters[], Field_filters[],
                   Focus_filters[], Fps_filters[], Histogram_filters[],
                   Layer_filters[], Levels_filters[], Misc_filters[],
                   Plugin_functions[], Resample_filters[], Resize_filters[],
                   Script_functions[], Source_filters[], Text_filters[],
                   Transform_filters[], Merge_filters[], Color_filters[],
                   Debug_filters[], Turn_filters[],
                   Conditional_filters[], Conditional_funtions_filters[],
                   Cache_filters[], Greyscale_filters[],
                   Swap_filters[], Overlay_filters[], Exprfilter_filters[],
                   FilterGraph_filters[], Device_filters[];


const AVSFunction* const builtin_functions[] = {
                   Audio_filters, Combine_filters, Convert_filters,
                   Convolution_filters, Edit_filters, Field_filters,
                   Focus_filters, Fps_filters, Histogram_filters,
                   Layer_filters, Levels_filters, Misc_filters,
                   Resample_filters, Resize_filters,
                   Script_functions, Source_filters, Text_filters,
                   Transform_filters, Merge_filters, Color_filters,
                   Debug_filters, Turn_filters,
                   Conditional_filters, Conditional_funtions_filters,
                   Plugin_functions, Cache_filters,
                   Overlay_filters, Greyscale_filters, Swap_filters,
                   FilterGraph_filters, Device_filters, Exprfilter_filters };

// Global statistics counters
struct {
  unsigned int CleanUps;
  unsigned int Losses;
  unsigned int PlanA1;
  unsigned int PlanA2;
  unsigned int PlanB;
  unsigned int PlanC;
  unsigned int PlanD;
  char tag[36];
} g_Mem_stats = {0, 0, 0, 0, 0, 0, 0, "CleanUps, Losses, Plan[A1,A2,B,C,D]"};

const _PixelClip PixelClip;


// Helper function to count set bits in the processor mask.
static DWORD CountSetBits(ULONG_PTR bitMask)
{
  DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
  DWORD bitSetCount = 0;
  ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;
  DWORD i;

  for (i = 0; i <= LSHIFT; ++i)
  {
    bitSetCount += ((bitMask & bitTest)?1:0);
    bitTest/=2;
  }

  return bitSetCount;
}

static size_t GetNumPhysicalCPUs()
{
#ifdef MSVC
  typedef BOOL (WINAPI *LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);
  LPFN_GLPI glpi;
  BOOL done = FALSE;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
  DWORD returnLength = 0;
  DWORD logicalProcessorCount = 0;
  DWORD numaNodeCount = 0;
  DWORD processorCoreCount = 0;
  DWORD processorL1CacheCount = 0;
  DWORD processorL2CacheCount = 0;
  DWORD processorL3CacheCount = 0;
  DWORD processorPackageCount = 0;
  DWORD byteOffset = 0;
  PCACHE_DESCRIPTOR Cache;

  glpi = (LPFN_GLPI) GetProcAddress(
    GetModuleHandle(TEXT("kernel32")),
    "GetLogicalProcessorInformation");
  if (NULL == glpi)
  {
//    _tprintf(TEXT("\nGetLogicalProcessorInformation is not supported.\n"));
    return (0);
  }

  while (!done)
  {
    BOOL rc = glpi(buffer, &returnLength);

    if (FALSE == rc)
    {
      if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
      {
        if (buffer)
          free(buffer);

        buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
          returnLength);

        if (NULL == buffer)
        {
//          _tprintf(TEXT("\nError: Allocation failure\n"));
          return (0);
        }
      }
      else
      {
//        _tprintf(TEXT("\nError %d\n"), GetLastError());
        return (0);
      }
    }
    else
    {
      done = TRUE;
    }
  }

  ptr = buffer;

  while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength)
  {
    switch (ptr->Relationship)
    {
    case RelationNumaNode:
      // Non-NUMA systems report a single record of this type.
      numaNodeCount++;
      break;

    case RelationProcessorCore:
      processorCoreCount++;

      // A hyperthreaded core supplies more than one logical processor.
      logicalProcessorCount += CountSetBits(ptr->ProcessorMask);
      break;

    case RelationCache:
      // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache.
      Cache = &ptr->Cache;
      if (Cache->Level == 1)
      {
        processorL1CacheCount++;
      }
      else if (Cache->Level == 2)
      {
        processorL2CacheCount++;
      }
      else if (Cache->Level == 3)
      {
        processorL3CacheCount++;
      }
      break;

    case RelationProcessorPackage:
      // Logical processors share a physical package.
      processorPackageCount++;
      break;

    default:
//      _tprintf(TEXT("\nError: Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.\n"));
      return (0);
    }
    byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    ptr++;
  }

  /*
  _tprintf(TEXT("\nGetLogicalProcessorInformation results:\n"));
  _tprintf(TEXT("Number of NUMA nodes: %d\n"),
    numaNodeCount);
  _tprintf(TEXT("Number of physical processor packages: %d\n"),
    processorPackageCount);
  _tprintf(TEXT("Number of processor cores: %d\n"),
    processorCoreCount);
  _tprintf(TEXT("Number of logical processors: %d\n"),
    logicalProcessorCount);
  _tprintf(TEXT("Number of processor L1/L2/L3 caches: %d/%d/%d\n"),
    processorL1CacheCount,
    processorL2CacheCount,
    processorL3CacheCount);
  */

  free(buffer);

  return processorCoreCount;
#else
  return 4; // GCC TODO
#endif
}

#ifdef MSVC
static std::string FormatString(const char *fmt, va_list args)
{
  va_list args2;
  va_copy(args2, args);
  _locale_t locale = _create_locale(LC_NUMERIC, "C"); // decimal point: dot

  int count = _vscprintf_l(fmt, locale, args2);
  // don't use _vsnprintf_l(NULL, 0, fmt, locale, args) here,
  // it returns -1 instead of the buffer size under Wine (February, 2017)
  std::vector<char> buf(count + 1);
  _vsnprintf_l(buf.data(), buf.size(), fmt, locale, args2);

  _free_locale(locale);
  va_end(args2);

  return std::string(buf.data());
}
#else
static std::string FormatString(const char *fmt, va_list args)
{
  va_list args2;
  va_copy(args2, args);

  int count = vsnprintf(NULL, 0, fmt, args);
  std::vector<char> buf(count + 1);
  vsnprintf(buf.data(), buf.size(), fmt, args2);

  va_end(args2);

  return std::string(buf.data());
}
#endif

void* VideoFrame::operator new(size_t size) {
  return ::operator new(size);
}

#ifdef SIZETMOD
VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, size_t _offset, int _pitch, int _row_size, int _height)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
  offsetU(_offset), offsetV(_offset), pitchUV(0), row_sizeUV(0), heightUV(0)  // PitchUV=0 so this doesn't take up additional space
  , offsetA(0), pitchA(0), row_sizeA(0)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, size_t _offset, int _pitch, int _row_size, int _height,
  size_t _offsetU, size_t _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
  offsetU(_offsetU), offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV)
  , offsetA(0), pitchA(0), row_sizeA(0)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, size_t _offset, int _pitch, int _row_size, int _height,
  size_t _offsetU, size_t _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV, size_t _offsetA)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
  offsetU(_offsetU), offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV)
  , offsetA(_offsetA), pitchA(_pitch), row_sizeA(_row_size)
{
  InterlockedIncrement(&vfb->refcount);
}
#else
VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, AVSMap* avsmap, int _offset, int _pitch, int _row_size, int _height)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
    offsetU(_offset), offsetV(_offset), pitchUV(0), row_sizeUV(0), heightUV(0)  // PitchUV=0 so this doesn't take up additional space
    ,offsetA(0), pitchA(0), row_sizeA(0), avsmap(avsmap)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, AVSMap* avsmap, int _offset, int _pitch, int _row_size, int _height,
                       int _offsetU, int _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
    offsetU(_offsetU), offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV)
    ,offsetA(0), pitchA(0), row_sizeA(0), avsmap(avsmap)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, AVSMap* avsmap, int _offset, int _pitch, int _row_size, int _height,
    int _offsetU, int _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV, int _offsetA)
    : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
    offsetU(_offsetU), offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV)
    ,offsetA(_offsetA), pitchA(_pitch), row_sizeA(_row_size), avsmap(avsmap)
{
    InterlockedIncrement(&vfb->refcount);
}
#endif
// Hack note :- Use of SubFrame will require an "InterlockedDecrement(&retval->refcount);" after
// assignement to a PVideoFrame, the same as for a "New VideoFrame" to keep the refcount consistant.
// P.F. ?? so far it works automatically

VideoFrame* VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height) const {
  return new VideoFrame(vfb, new AVSMap(), offset+rel_offset, new_pitch, new_row_size, new_height);
}


VideoFrame* VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height,
                                 int rel_offsetU, int rel_offsetV, int new_pitchUV) const {
    // Maintain plane size relationship
    const int new_row_sizeUV = !row_size ? 0 : MulDiv(new_row_size, row_sizeUV, row_size);
    const int new_heightUV   = !height   ? 0 : MulDiv(new_height,   heightUV,   height);

    return new VideoFrame(vfb, new AVSMap(), offset+rel_offset, new_pitch, new_row_size, new_height,
        rel_offsetU+offsetU, rel_offsetV+offsetV, new_pitchUV, new_row_sizeUV, new_heightUV);
}

// alpha support
VideoFrame* VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height,
    int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA) const {
    // Maintain plane size relationship
    const int new_row_sizeUV = !row_size ? 0 : MulDiv(new_row_size, row_sizeUV, row_size);
    const int new_heightUV   = !height   ? 0 : MulDiv(new_height,   heightUV,   height);

    return new VideoFrame(vfb, new AVSMap(), offset+rel_offset, new_pitch, new_row_size, new_height,
        rel_offsetU+offsetU, rel_offsetV+offsetV, new_pitchUV, new_row_sizeUV, new_heightUV, rel_offsetA+offsetA);
}

VideoFrameBuffer::VideoFrameBuffer() : refcount(1), data(NULL), data_size(0), sequence_number(0) {}


VideoFrameBuffer::VideoFrameBuffer(int size, int margin, Device* device) :
  data(device->Allocate(size, margin)),
  data_size(size),
  sequence_number(0),
  refcount(0),
  device(device)
{
}

VideoFrameBuffer::~VideoFrameBuffer() {
//  _ASSERTE(refcount == 0);
  InterlockedIncrement(&sequence_number); // HACK : Notify any children with a pointer, this buffer has changed!!!
  if (data) device->Free(data);
  data = nullptr; // and mark it invalid!!
  data_size = 0;   // and don't forget to set the size to 0 as well!
  device = nullptr; // no longer related to a device
}


class AtExiter {
  struct AtExitRec {
    const IScriptEnvironment::ShutdownFunc func;
    void* const user_data;
    AtExitRec* const next;
    AtExitRec(IScriptEnvironment::ShutdownFunc _func, void* _user_data, AtExitRec* _next)
      : func(_func), user_data(_user_data), next(_next) {}
  };
  AtExitRec* atexit_list;

public:
  AtExiter() {
    atexit_list = 0;
  }

  void Add(IScriptEnvironment::ShutdownFunc f, void* d) {
    atexit_list = new AtExitRec(f, d, atexit_list);
  }

  void Execute(IScriptEnvironment* env) {
    while (atexit_list) {
      AtExitRec* next = atexit_list->next;
      atexit_list->func(atexit_list->user_data, env);
      delete atexit_list;
      atexit_list = next;
    }
  }
};


static std::string NormalizeString(const std::string& str)
{
  // lowercase
  std::string ret = str;
  for (size_t i = 0; i < ret.size(); ++i)
    ret[i] = tolower(ret[i]);

  // trim trailing spaces
  size_t endpos = ret.find_last_not_of(" \t");
  if (std::string::npos != endpos)
    ret = ret.substr(0, endpos + 1);

  // trim leading spaces
  size_t startpos = ret.find_first_not_of(" \t");
  if (std::string::npos != startpos)
    ret = ret.substr(startpos);

  return ret;
}

typedef enum class _MtWeight
{
  MT_WEIGHT_0_DEFAULT,
  MT_WEIGHT_1_USERSPEC,
  MT_WEIGHT_2_USERFORCE,
  MT_WEIGHT_MAX
} MtWeight;

class ClipDataStore
{
public:

  // The clip instance that we hold data for.
  IClip *Clip = nullptr;

  // Clip was created directly by an Invoke() call
  bool CreatedByInvoke = false;

  ClipDataStore(IClip *clip) : Clip(clip) {};
};

class MtModeEvaluator
{
public:
  int NumChainedNice = 0;
  int NumChainedMulti = 0;
  int NumChainedSerial = 0;

  MtMode GetFinalMode(MtMode topInvokeMode)
  {
    if (NumChainedSerial > 0)
    {
      return MT_SERIALIZED;
    }
    else if (NumChainedMulti > 0)
    {
      if (MT_SERIALIZED == topInvokeMode)
      {
        return MT_SERIALIZED;
      }
      else
      {
        return MT_MULTI_INSTANCE;
      }
    }
    else
    {
      return topInvokeMode;
    }
  }

  void Accumulate(const MtModeEvaluator &other)
  {
    NumChainedNice += other.NumChainedNice;
    NumChainedMulti += other.NumChainedMulti;
    NumChainedSerial += other.NumChainedSerial;
  }

  void Accumulate(MtMode mode)
  {
    switch (mode)
    {
    case MT_NICE_FILTER:
      ++NumChainedNice;
      break;
    case MT_MULTI_INSTANCE:
      ++NumChainedMulti;
      break;
    case MT_SERIALIZED:
      ++NumChainedSerial;
      break;
    default:
      assert(0);
      break;
    }
  }

  static bool ClipSpecifiesMtMode(const PClip &clip)
  {
    int val = clip->SetCacheHints(CACHE_GET_MTMODE, 0);
    return (clip->GetVersion() >= 5) && (val > MT_INVALID) && (val < MT_MODE_COUNT);
  }

  static MtMode GetInstanceMode(const PClip &clip, MtMode defaultMode)
  {
    return ClipSpecifiesMtMode(clip) ? (MtMode)clip->SetCacheHints(CACHE_GET_MTMODE, 0) : defaultMode;
  }

  static MtMode GetInstanceMode(const PClip &clip)
  {
    return (MtMode)clip->SetCacheHints(CACHE_GET_MTMODE, 0);
  }

  static MtMode GetMtMode(const PClip &clip, const Function *invokeCall, const InternalEnvironment* env)
  {
    bool invokeModeForced;

    MtMode invokeMode = env->GetFilterMTMode(invokeCall, &invokeModeForced);
    if (invokeModeForced) {
      return invokeMode;
    }

    bool hasInstanceMode = ClipSpecifiesMtMode(clip);
    if (hasInstanceMode) {
      return GetInstanceMode(clip);
        } else {
      return invokeMode;
    }
  }

  static bool UsesDefaultMtMode(const PClip &clip, const Function *invokeCall, const InternalEnvironment* env)
  {
    return !ClipSpecifiesMtMode(clip) && !env->FilterHasMtMode(invokeCall);
  }

  void AddChainedFilter(const PClip &clip, MtMode defaultMode)
  {
    MtMode mode = GetInstanceMode(clip, defaultMode);
    Accumulate(mode);
  }
};


OneTimeLogTicket::OneTimeLogTicket(ELogTicketType type)
  : _type(type)
{}

OneTimeLogTicket::OneTimeLogTicket(ELogTicketType type, const Function *func)
  : _type(type), _function(func)
{}

OneTimeLogTicket::OneTimeLogTicket(ELogTicketType type, const std::string &str)
  : _type(type), _string(str)
{}

bool OneTimeLogTicket::operator==(const OneTimeLogTicket &other) const
{
  return (_type == other._type)
    && (_function == other._function)
    && (_string.compare(other._string) == 0);
}

namespace std
{
template <>
struct hash<OneTimeLogTicket>
{
  std::size_t operator()(const OneTimeLogTicket& k) const
  {
    // TODO: This is a pretty poor combination function for hashes.
    // Find something better than a simple XOR.
    return hash<int>()(k._type)
      ^ hash<void*>()((void*)k._function)
      ^ hash<std::string>()((std::string)k._string);
  }
};
}

#include "vartable.h"
#include "ThreadPool.h"
#include <map>
#include <unordered_set>
#include <atomic>
#include <stack>
#include "Prefetcher.h"
#include "BufferPool.h"
#include "ScriptEnvironmentTLS.h"
class ThreadScriptEnvironment;
class ScriptEnvironment {
public:
  ScriptEnvironment();
  void CheckVersion(int version);
  int GetCPUFlags();
  void AddFunction(const char* name, const char* params, INeoEnv::ApplyFunc apply, void* user_data = 0);
  bool FunctionExists(const char* name);
  PVideoFrame NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device);
  PVideoFrame NewVideoFrameOnDevice(int row_size, int height, int align, Device* device);
  PVideoFrame NewVideoFrame(const VideoInfo& vi, const PDevice& device);
  PVideoFrame NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, Device* device);
	bool MakeWritable(PVideoFrame* pvf);
	bool MakePropertyWritable(PVideoFrame* pvf);
  void BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height);
  void AtExit(IScriptEnvironment::ShutdownFunc function, void* user_data);
  PVideoFrame Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height);
  int SetMemoryMax(int mem);
  int SetWorkingDir(const char * newdir);
  AVSC_CC ~ScriptEnvironment();
  void* ManageCache(int key, void* data);
  bool PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key);
  PVideoFrame SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV);
  void DeleteScriptEnvironment();
  void ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor);
  const AVS_Linkage* GetAVSLinkage();

  // alpha support
  PVideoFrame NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, bool alpha, Device* device);
  PVideoFrame SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA);

  /* IScriptEnvironment2 */
  bool LoadPlugin(const char* filePath, bool throwOnError, AVSValue *result);
  void AddAutoloadDir(const char* dirPath, bool toFront);
  void ClearAutoloadDirs();
  void AutoloadPlugins();
  void AddFunction(const char* name, const char* params, INeoEnv::ApplyFunc apply, void* user_data, const char *exportVar);
  bool InternalFunctionExists(const char* name);
  void AdjustMemoryConsumption(size_t amount, bool minus);
  void SetFilterMTMode(const char* filter, MtMode mode, bool force);
  void SetFilterMTMode(const char* filter, MtMode mode, MtWeight weight);
  MtMode GetFilterMTMode(const Function* filter, bool* is_forced) const;
  void ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion);
  IJobCompletion* NewCompletion(size_t capacity);
  size_t  GetProperty(AvsEnvProperty prop);
  ClipDataStore* ClipData(IClip *clip);
  MtMode GetDefaultMtMode() const;
  bool FilterHasMtMode(const Function* filter) const;
  void SetLogParams(const char *target, int level);
  void LogMsg(int level, const char* fmt, ...);
  void LogMsg_valist(int level, const char* fmt, va_list va);
  void LogMsgOnce(const OneTimeLogTicket &ticket, int level, const char* fmt, ...);
  void LogMsgOnce_valist(const OneTimeLogTicket &ticket, int level, const char* fmt, va_list va);
  PVideoFrame SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA);

  /* INeoEnv */
  bool Invoke_(AVSValue *result, const AVSValue& implicit_last,
    const char* name, const Function *f, const AVSValue& args, const char* const* arg_names,
    InternalEnvironment* env_thread, bool is_runtime);

  PDevice GetDevice(AvsDeviceType device_type, int device_index) const;
  int SetMemoryMax(AvsDeviceType type, int index, int mem);

  PVideoFrame GetOnDeviceFrame(const PVideoFrame& src, Device* device);
  void CopyFrameProps(PVideoFrame src, PVideoFrame dst) const;
  void ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion, InternalEnvironment *env);
  ThreadPool* NewThreadPool(size_t nThreads);
  AVSMap* GetAVSMap(PVideoFrame& frame) { return frame->avsmap; }

  void SetGraphAnalysis(bool enable) { graphAnalysisEnable = enable; }

  void IncEnvCount() { InterlockedIncrement(&EnvCount); }
  void DecEnvCount() { InterlockedDecrement(&EnvCount); }

  ConcurrentVarStringFrame* GetTopFrame() { return &top_frame; }
  void SetCacheMode(CacheMode mode) { cacheMode = mode; }
  CacheMode GetCacheMode() { return cacheMode; }
  void SetDeviceOpt(DeviceOpt opt, int val);

  void UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar);

  ThreadScriptEnvironment* GetMainThreadEnv() { return threadEnv.get(); }

private:

  typedef IScriptEnvironment::NotFound NotFound;
  typedef IScriptEnvironment::ApplyFunc ApplyFunc;

  // Tritical May 2005
  // Note order here!!
  // AtExiter has functions which
  // rely on StringDump elements.
  ConcurrentVarStringFrame top_frame;
  std::unique_ptr<ThreadScriptEnvironment> threadEnv;
  std::mutex string_mutex;
  char * vsprintf_buf;
  size_t vsprintf_len;

  AtExiter at_exit;
  ThreadPool * thread_pool;

  PluginManager *plugin_manager;
  std::recursive_mutex plugin_mutex;

  long EnvCount; // for ThreadScriptEnvironment leak detection

  void ThrowError(const char* fmt, ...);
  void VThrowError(const char* fmt, va_list va);

  const Function* Lookup(const char* search_name, const AVSValue* args, size_t num_args,
    bool &pstrict, size_t args_names_count, const char* const* arg_names, IScriptEnvironment2* env_thread);
  bool CheckArguments(const Function* f, const AVSValue* args, size_t num_args,
    bool &pstrict, size_t args_names_count, const char* const* arg_names);
  std::unordered_map<IClip*, ClipDataStore> clip_data;

  void ExportBuiltinFilters();

  bool PlanarChromaAlignmentState;

  HRESULT hrfromcoinit;
  DWORD coinitThreadId;

  struct DebugTimestampedFrame
  {
    VideoFrame *frame;
    AVSMap *avsmap;
#ifdef _DEBUG
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
#endif

    DebugTimestampedFrame(VideoFrame *_frame, AVSMap *_avsmap)
      : frame(_frame)
      , avsmap(_avsmap)
#ifdef _DEBUG
      , timestamp(std::chrono::high_resolution_clock::now())
#endif
    {}
  };
  class VFBStorage : public VideoFrameBuffer {
  public:
    int free_count;
    int margin;
    PGraphMemoryNode memory_node;
    VFBStorage()
      : VideoFrameBuffer(),
      free_count(0)
    { }
    VFBStorage(int size, int margin, Device* device)
      : VideoFrameBuffer(size, margin, device),
      free_count(0),
      margin(margin)
    { }
    void Attach(FilterGraphNode* node) {
      if (memory_node) {
        memory_node->OnFree(data_size, device);
        memory_node = nullptr;
      }
      if (node != nullptr) {
        memory_node = node->GetMemoryNode();
        memory_node->OnAllocate(data_size, device);
      }
    }
    ~VFBStorage() {
      if (memory_node) {
        memory_node->OnFree(data_size, device);
        memory_node = nullptr;
      }
#ifdef _DEBUG
      if (data && device->device_type == DEV_TYPE_CPU) {
        // check buffer overrun
        int *pInt = (int *)(data + margin + data_size);
        if (pInt[0] != 0xDEADBEEF ||
          pInt[1] != 0xDEADBEEF ||
          pInt[2] != 0xDEADBEEF ||
          pInt[3] != 0xDEADBEEF)
        {
          printf("Buffer overrun!!!\n");
        }
      }
#endif
    }
  };
  typedef std::vector<DebugTimestampedFrame> VideoFrameArrayType;
  typedef std::map<VideoFrameBuffer *, VideoFrameArrayType> FrameBufferRegistryType;
  typedef std::map<size_t, FrameBufferRegistryType> FrameRegistryType2; // post r1825 P.F.
  typedef mapped_list<Cache*> CacheRegistryType;


  FrameRegistryType2 FrameRegistry2; // P.F.
#ifdef _DEBUG
  void ListFrameRegistry(size_t min_size, size_t max_size, bool someframes);
#endif

  std::unique_ptr<DeviceManager> Devices;
  CacheRegistryType CacheRegistry;
  Cache* FrontCache;
  VideoFrame* GetNewFrame(size_t vfb_size, size_t margin, Device* device);
  VideoFrame* GetFrameFromRegistry(size_t vfb_size, Device* device);
  void ShrinkCache(Device* device);
  VideoFrame* AllocateFrame(size_t vfb_size, size_t margin, Device* device);
  std::recursive_mutex memory_mutex;

	int frame_align;
	int plane_align;

  //BufferPool BufferPool;

  typedef std::vector<MTGuard*> MTGuardRegistryType;
  MTGuardRegistryType MTGuardRegistry;

  std::vector <std::unique_ptr<ThreadPool>> ThreadPoolRegistry;
  size_t nTotalThreads;
  size_t nMaxFilterInstances;

  // Members used to reconstruct Association between Invoke() calls and filter instances
  std::stack<MtModeEvaluator*> invoke_stack;

  // MT mode specifications
  std::unordered_map<std::string, std::pair<MtMode, MtWeight>> MtMap;
  MtMode DefaultMtMode = MtMode::MT_MULTI_INSTANCE;
  static const std::string DEFAULT_MODE_SPECIFIER;

  // Logging-related members
  int LogLevel;
  std::string LogTarget;
  std::ofstream LogFileStream;
  std::unordered_set<OneTimeLogTicket> LogTickets;

  // filter graph
  bool graphAnalysisEnable;

	typedef std::vector<FilterGraphNode*> GraphNodeRegistryType;
	GraphNodeRegistryType GraphNodeRegistry;

  CacheMode cacheMode;

  void InitMT();
};
const std::string ScriptEnvironment::DEFAULT_MODE_SPECIFIER = "DEFAULT_MT_MODE";

// Only ThrowError and Sprintf is implemented(Used by destructor)
class MinimumScriptEnvironment : public IScriptEnvironment {
	VarTable var_table;
public:
	MinimumScriptEnvironment(ConcurrentVarStringFrame* top_frame) : var_table(top_frame) { }
	virtual ~MinimumScriptEnvironment() {}
	virtual int __stdcall GetCPUFlags() {
		throw AvisynthError("Not Implemented");
	}
	virtual char* __stdcall SaveString(const char* s, int length = -1) {
		return var_table.SaveString(s, length);
	}
	virtual char* __stdcall Sprintf(const char* fmt, ...) {
		va_list val;
		va_start(val, fmt);
		char* result = VSprintf(fmt, val);
		va_end(val);
		return result;
	}
	virtual char* __stdcall VSprintf(const char* fmt, void* val) {
		try {
			std::string str = FormatString(fmt, (va_list)val);
			return var_table.SaveString(str.c_str(), int(str.size())); // SaveString will add the NULL in len mode.
		}
		catch (...) {
			return NULL;
		}
	}
	__declspec(noreturn) virtual void __stdcall ThrowError(const char* fmt, ...) {
		va_list val;
		va_start(val, fmt);
		VThrowError(fmt, val);
		va_end(val);
	}
	void __stdcall VThrowError(const char* fmt, va_list va)
	{
		std::string msg;
		try {
			msg = FormatString(fmt, va);
		}
		catch (...) {
			msg = "Exception while processing ScriptEnvironment::ThrowError().";
		}
		// Throw...
		throw AvisynthError(var_table.SaveString(msg.c_str()));
	}
	virtual void __stdcall AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data) {
		throw AvisynthError("Not Implemented");
	}
	virtual bool __stdcall FunctionExists(const char* name) {
		throw AvisynthError("Not Implemented");
	}
	virtual AVSValue __stdcall Invoke(const char* name, const AVSValue args, const char* const* arg_names = 0) {
		throw AvisynthError("Not Implemented");
	}
	virtual AVSValue __stdcall GetVar(const char* name) {
		throw AvisynthError("Not Implemented");
	}
	virtual bool __stdcall SetVar(const char* name, const AVSValue& val) {
		throw AvisynthError("Not Implemented");
	}
	virtual bool __stdcall SetGlobalVar(const char* name, const AVSValue& val) {
		throw AvisynthError("Not Implemented");
	}
	virtual void __stdcall PushContext(int level = 0) {
		throw AvisynthError("Not Implemented");
	}
	virtual void __stdcall PopContext() {
		throw AvisynthError("Not Implemented");
	}
	virtual PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, int align = FRAME_ALIGN) {
		throw AvisynthError("Not Implemented");
	}
	virtual bool __stdcall MakeWritable(PVideoFrame* pvf) {
		throw AvisynthError("Not Implemented");
	}
	virtual void __stdcall BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height) {
		throw AvisynthError("Not Implemented");
	}
	virtual void __stdcall AtExit(ShutdownFunc function, void* user_data) {
		throw AvisynthError("Not Implemented");
	}
	virtual void __stdcall CheckVersion(int version = AVISYNTH_INTERFACE_VERSION) {
		throw AvisynthError("Not Implemented");
	}
	virtual PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) {
		throw AvisynthError("Not Implemented");
	}
	virtual int __stdcall SetMemoryMax(int mem) {
		throw AvisynthError("Not Implemented");
	}
	virtual int __stdcall SetWorkingDir(const char * newdir) {
		throw AvisynthError("Not Implemented");
	}
	virtual void* __stdcall ManageCache(int key, void* data) {
		throw AvisynthError("Not Implemented");
	}
	virtual bool __stdcall PlanarChromaAlignment(PlanarChromaAlignmentMode key) {
		throw AvisynthError("Not Implemented");
	}
	virtual PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
		int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) {
		throw AvisynthError("Not Implemented");
	}
	virtual void __stdcall DeleteScriptEnvironment() {
		throw AvisynthError("Not Implemented");
	}
	virtual void __stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
		int textcolor, int halocolor, int bgcolor) {
		throw AvisynthError("Not Implemented");
	}
	virtual const AVS_Linkage* __stdcall GetAVSLinkage() {
		throw AvisynthError("Not Implemented");
	}
	virtual AVSValue __stdcall GetVarDef(const char* name, const AVSValue& def = AVSValue()) {
		throw AvisynthError("Not Implemented");
	}
};

/* ---------------------------------------------------------------------------------
*  Per thread data
* ---------------------------------------------------------------------------------
*/
struct ScriptEnvironmentTLS
{
  const int thread_id;
  // PF 161223 why do we need thread-local global variables?
  // comment remains here until it gets cleared, anyway, I make it of no use
  VarTable var_table;
  BufferPool BufferPool;
  Device* currentDevice;
  bool closing;                 // Used to avoid deadlock, if vartable is being accessed while shutting down (Popcontext)
  bool supressCaching;
  int ImportDepth;
	int getFrameRecursiveCount;

	// Concurrent GetFrame with Invoke causes deadlock.
	// Increment this variable when Invoke running
	// to prevent submitting job to threadpool
	int suppressThreadCount;
	
	FilterGraphNode* currentGraphNode;
  volatile long refcount;

  ScriptEnvironmentTLS(int thread_id, InternalEnvironment* core)
    : thread_id(thread_id)
    , var_table(core->GetTopFrame())
    , BufferPool(core)
    , currentDevice(NULL)
    , closing(false)
    , supressCaching(false)
    , ImportDepth(0)
		, getFrameRecursiveCount(0)
		, suppressThreadCount(0)
		, currentGraphNode(nullptr)
    , refcount(1)
  {
  }
};

// per thread data is bound to a thread (not ThreadScriptEnvironment)
// since some filter (e.g. svpflow1) ignores env given for GetFrame, and always use main thread's env.
// this is a work-around for that.
__declspec(thread) ScriptEnvironmentTLS* g_TLS;

class ThreadScriptEnvironment : public InternalEnvironment
{
  ScriptEnvironment* core;
  ScriptEnvironmentTLS* coreTLS;
  ScriptEnvironmentTLS myTLS;
public:

  ThreadScriptEnvironment(int thread_id, ScriptEnvironment* core, ScriptEnvironmentTLS* coreTLS)
    : core(core)
    , coreTLS(coreTLS)
    , myTLS(thread_id, this)
  {
    if (coreTLS == nullptr) {
      // when this is main thread TLS
      this->coreTLS = &myTLS;
    }
    if (thread_id != 0) {
      // thread pool thread
      if (g_TLS != nullptr) {
        ThrowError("Detected multiple ScriptEnvironmentTLSs for a single thread");
      }
      g_TLS = &myTLS;
    }
    core->IncEnvCount(); // for leak detection
  }

  ~ThreadScriptEnvironment() {
    core->DecEnvCount(); // for leak detection
  }

  ScriptEnvironmentTLS* GetTLS() { return &myTLS; }

#define DISPATCH(name) (g_TLS ? g_TLS : coreTLS)->name

  AVSValue __stdcall GetVar(const char* name)
  {
    if (DISPATCH(closing)) return AVSValue();  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (DISPATCH(var_table).Get(name, &val))
      return val;
    else
      throw IScriptEnvironment::NotFound();
  }

  bool __stdcall SetVar(const char* name, const AVSValue& val)
  {
    if (DISPATCH(closing)) return true;  // We easily risk  being inside the critical section below, while deleting variables.
    return DISPATCH(var_table).Set(name, val);
  }

  bool __stdcall SetGlobalVar(const char* name, const AVSValue& val)
  {
    if (DISPATCH(closing)) return true;  // We easily risk  being inside the critical section below, while deleting variables.
    return DISPATCH(var_table).SetGlobal(name, val);
  }

  void __stdcall PushContext(int level = 0)
  {
    DISPATCH(var_table).Push();
  }

  void __stdcall PopContext()
  {
    DISPATCH(var_table).Pop();
  }

  void __stdcall PushContextGlobal()
  {
    DISPATCH(var_table).PushGlobal();
  }

  void __stdcall PopContextGlobal()
  {
    DISPATCH(var_table).PopGlobal();
  }

  bool __stdcall GetVar(const char* name, AVSValue *val) const
  {
    if (DISPATCH(closing)) return false;  // We easily risk  being inside the critical section below, while deleting variables.
    return DISPATCH(var_table).Get(name, val);
  }

  AVSValue __stdcall GetVarDef(const char* name, const AVSValue& def)
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVar(name, &val))
      return val;
    else
      return def;
  }

  bool __stdcall GetVar(const char* name, bool def) const
  {
    if (DISPATCH(closing)) return false;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVar(name, &val))
      return val.AsBool(def);
    else
      return def;
  }

  int __stdcall GetVar(const char* name, int def) const
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVar(name, &val))
      return val.AsInt(def);
    else
      return def;
  }

  double __stdcall GetVar(const char* name, double def) const
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVar(name, &val))
      return val.AsDblDef(def);
    else
      return def;
  }

  const char* __stdcall GetVar(const char* name, const char* def) const
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVar(name, &val))
      return val.AsString(def);
    else
      return def;
  }

  void* __stdcall Allocate(size_t nBytes, size_t alignment, AvsAllocType type)
  {
    if ((type != AVS_NORMAL_ALLOC) && (type != AVS_POOLED_ALLOC))
      return NULL;
    return DISPATCH(BufferPool).Allocate(nBytes, alignment, type == AVS_POOLED_ALLOC);
  }

  void __stdcall Free(void* ptr)
  {
    DISPATCH(BufferPool).Free(ptr);
  }

  Device* __stdcall GetCurrentDevice() const
  {
    return DISPATCH(currentDevice);
  }

  Device* __stdcall SetCurrentDevice(Device* device)
  {
    Device* old = DISPATCH(currentDevice);
    DISPATCH(currentDevice) = device;
    return old;
  }

  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, int align)
  {
    return core->NewVideoFrameOnDevice(vi, align, DISPATCH(currentDevice));
  }

  void* __stdcall GetDeviceStream()
  {
    return DISPATCH(currentDevice)->GetComputeStream();
  }

  void __stdcall DeviceAddCallback(void(*cb)(void*), void* user_data)
  {
    DeviceCompleteCallbackData cbdata = { cb, user_data };
    DISPATCH(currentDevice)->AddCompleteCallback(cbdata);
  }

  PVideoFrame __stdcall GetFrame(PClip c, int n, const PDevice& device)
  {
    DeviceSetter setter(this, (Device*)(void*)device);
    return c->GetFrame(n, this);
  }


  /* ---------------------------------------------------------------------------------
  *             S T U B S
  * ---------------------------------------------------------------------------------
  */

  bool __stdcall InternalFunctionExists(const char* name)
  {
    return core->InternalFunctionExists(name);
  }

  void __stdcall AdjustMemoryConsumption(size_t amount, bool minus)
  {
    core->AdjustMemoryConsumption(amount, minus);
  }

  void __stdcall CheckVersion(int version)
  {
    core->CheckVersion(version);
  }

  int __stdcall GetCPUFlags()
  {
    return core->GetCPUFlags();
  }

  char* __stdcall SaveString(const char* s, int length = -1)
  {
    return DISPATCH(var_table).SaveString(s, length);
  }

  char* __stdcall SaveString(const char* s, int length, bool escape)
  {
    return DISPATCH(var_table).SaveString(s, length, escape);
  }

  char* __stdcall Sprintf(const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    // do not call core->Sprintf, because cannot pass ... further
    char* result = VSprintf(fmt, val);
    va_end(val);
    return result;
  }

  char* __stdcall VSprintf(const char* fmt, void* val)
  {
    try {
      std::string str = FormatString(fmt, (va_list)val);
      return DISPATCH(var_table).SaveString(str.c_str(), int(str.size())); // SaveString will add the NULL in len mode.
    }
    catch (...) {
      return NULL;
    }
  }

  void __stdcall ThrowError(const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    VThrowError(fmt, val);
    va_end(val);
  }

  void __stdcall VThrowError(const char* fmt, va_list va)
  {
    std::string msg;
    try {
      msg = FormatString(fmt, va);
    }
    catch (...) {
      msg = "Exception while processing ScriptEnvironment::ThrowError().";
    }

    // Also log the error before throwing
    this->LogMsg(LOGLEVEL_ERROR, msg.c_str());

    // Throw...
    throw AvisynthError(DISPATCH(var_table).SaveString(msg.c_str()));
  }

  PVideoFrame __stdcall SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA)
  {
    return core->SubframePlanarA(src, rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV, rel_offsetA);
  }

  void __stdcall AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data = 0)
  {
    core->AddFunction(name, params, apply, user_data);
  }

  bool __stdcall FunctionExists(const char* name)
  {
    return core->FunctionExists(name);
  }

  bool IsRuntime() {
    // When invoked from GetFrame/GetAudio, skip all cache and mt mecanism
    bool is_runtime = true;

    if (g_TLS == nullptr) { // not called by thread
      if (GetFrameRecursiveCount() == 0) { // not called by GetFrame
        is_runtime = false;
      }
    }

    return is_runtime;
  }

  AVSValue __stdcall Invoke(const char* name,
    const AVSValue args, const char* const* arg_names)
  {
    AVSValue result;
    if (!core->Invoke_(&result, AVSValue(), name, nullptr, args, arg_names, this, IsRuntime()))
    {
      throw NotFound();
    }
    return result;
  }

  bool __stdcall Invoke(AVSValue* result,
    const char* name, const AVSValue& args, const char* const* arg_names)
  {
    return core->Invoke_(result, AVSValue(), name, nullptr, args, arg_names, this, IsRuntime());
  }

  bool __stdcall Invoke(AVSValue* result, const AVSValue& implicit_last,
    const char* name, const AVSValue args, const char* const* arg_names)
  {
    return core->Invoke_(result, implicit_last,
      name, nullptr, args, arg_names, this, IsRuntime());
  }

  AVSValue __stdcall Invoke(const AVSValue& implicit_last,
    const PFunction& func, const AVSValue args, const char* const* arg_names)
  {
    AVSValue result;
    if (!core->Invoke_(&result, implicit_last,
      func->GetLegacyName(), func->GetDefinition(), args, arg_names, this, IsRuntime()))
    {
      throw NotFound();
    }
    return result;
  }

  bool __stdcall Invoke(AVSValue *result, const AVSValue& implicit_last,
    const PFunction& func, const AVSValue args, const char* const* arg_names)
  {
    return core->Invoke_(result, implicit_last,
      func->GetLegacyName(), func->GetDefinition(), args, arg_names, this, IsRuntime());
  }

  bool __stdcall Invoke_(AVSValue *result, const AVSValue& implicit_last,
    const char* name, const Function *f, const AVSValue& args, const char* const* arg_names)
  {
    return core->Invoke_(result, implicit_last, name, f, args, arg_names, this, IsRuntime());
  }

  bool __stdcall MakeWritable(PVideoFrame* pvf)
  {
    return core->MakeWritable(pvf);
  }

  void __stdcall BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height)
  {
    core->BitBlt(dstp, dst_pitch, srcp, src_pitch, row_size, height);
  }

  void __stdcall AtExit(IScriptEnvironment::ShutdownFunc function, void* user_data)
  {
    core->AtExit(function, user_data);
  }

  PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height)
  {
    return core->Subframe(src, rel_offset, new_pitch, new_row_size, new_height);
  }

  int __stdcall SetMemoryMax(int mem)
  {
    return core->SetMemoryMax(mem);
  }

  int __stdcall SetWorkingDir(const char * newdir)
  {
    return core->SetWorkingDir(newdir);
  }

  void* __stdcall ManageCache(int key, void* data)
  {
    return core->ManageCache(key, data);
  }

  bool __stdcall PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key)
  {
    return core->PlanarChromaAlignment(key);
  }

  PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV)
  {
    return core->SubframePlanar(src, rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV);
  }

  void __stdcall DeleteScriptEnvironment()
  {
    if (g_TLS != nullptr) {
      ThrowError("Cannot delete environment from a TLS proxy.");
    }
    core->DeleteScriptEnvironment();
  }

  void __stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor)
  {
    core->ApplyMessage(frame, vi, message, size, textcolor, halocolor, bgcolor);
  }

  const AVS_Linkage* __stdcall GetAVSLinkage()
  {
    return core->GetAVSLinkage();
  }

  /* IScriptEnvironment2 */
  bool __stdcall LoadPlugin(const char* filePath, bool throwOnError, AVSValue *result)
  {
    return core->LoadPlugin(filePath, throwOnError, result);
  }

  void __stdcall AddAutoloadDir(const char* dirPath, bool toFront)
  {
    core->AddAutoloadDir(dirPath, toFront);
  }

  void __stdcall ClearAutoloadDirs()
  {
    core->ClearAutoloadDirs();
  }

  void __stdcall AutoloadPlugins()
  {
    core->AutoloadPlugins();
  }

  void __stdcall AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data, const char *exportVar)
  {
    core->AddFunction(name, params, apply, user_data, exportVar);
  }

  int __stdcall IncrImportDepth()
  {
    return ++DISPATCH(ImportDepth);
  }

  int __stdcall DecrImportDepth()
  {
    return --DISPATCH(ImportDepth);
  }

  size_t  __stdcall GetProperty(AvsEnvProperty prop)
  {
    switch (prop)
    {
    case AEP_THREAD_ID:
      return DISPATCH(thread_id);
    case AEP_SUPPRESS_THREAD:
      return DISPATCH(suppressThreadCount);
    case AEP_GETFRAME_RECURSIVE:
      return DISPATCH(getFrameRecursiveCount);
    default:
      return core->GetProperty(prop);
    }
  }

  void __stdcall SetFilterMTMode(const char* filter, MtMode mode, bool force)
  {
    core->SetFilterMTMode(filter, mode, force);
  }

  MtMode __stdcall GetFilterMTMode(const Function* filter, bool* is_forced) const
  {
    return core->GetFilterMTMode(filter, is_forced);
  }

  bool __stdcall FilterHasMtMode(const Function* filter) const
  {
    return core->FilterHasMtMode(filter);
  }

  IJobCompletion* __stdcall NewCompletion(size_t capacity)
  {
    return core->NewCompletion(capacity);
  }

  void __stdcall ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion)
  {
    core->ParallelJob(jobFunc, jobData, completion, this);
  }

  void __stdcall ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion, InternalEnvironment *env)
  {
    core->ParallelJob(jobFunc, jobData, completion, env);
  }

  ClipDataStore* __stdcall ClipData(IClip *clip)
  {
    return core->ClipData(clip);
  }

  MtMode __stdcall GetDefaultMtMode() const
  {
    return core->GetDefaultMtMode();
  }

  void __stdcall SetLogParams(const char *target, int level)
  {
    core->SetLogParams(target, level);
  }

  void __stdcall LogMsg(int level, const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    core->LogMsg_valist(level, fmt, val);
    va_end(val);
  }

  void __stdcall LogMsg_valist(int level, const char* fmt, va_list va)
  {
    core->LogMsg_valist(level, fmt, va);
  }

  void __stdcall LogMsgOnce(const OneTimeLogTicket &ticket, int level, const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    core->LogMsgOnce_valist(ticket, level, fmt, val);
    va_end(val);
  }

  void __stdcall LogMsgOnce_valist(const OneTimeLogTicket &ticket, int level, const char* fmt, va_list va)
  {
    core->LogMsgOnce_valist(ticket, level, fmt, va);
  }

  void __stdcall SetGraphAnalysis(bool enable)
  {
    core->SetGraphAnalysis(enable);
  }

  int __stdcall SetMemoryMax(AvsDeviceType type, int index, int mem)
  {
    return core->SetMemoryMax(type, index, mem);
  }

  PDevice __stdcall GetDevice(AvsDeviceType device_type, int device_index) const
  {
    return core->GetDevice(device_type, device_index);
  }

  PDevice __stdcall GetDevice() const
  {
    return DISPATCH(currentDevice);
  }

  AvsDeviceType __stdcall GetDeviceType() const
  {
    return DISPATCH(currentDevice)->device_type;
  }

  int __stdcall GetDeviceId() const
  {
    return DISPATCH(currentDevice)->device_id;
  }

  int __stdcall GetDeviceIndex() const
  {
    return DISPATCH(currentDevice)->device_index;
  }

  void* __stdcall GetDeviceStream() const
  {
    return DISPATCH(currentDevice)->GetComputeStream();;
  }

  PVideoFrame __stdcall NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device)
  {
    return core->NewVideoFrameOnDevice(vi, align, device);
  }

  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi)
  {
    return NewVideoFrameOnDevice(vi, FRAME_ALIGN, DISPATCH(currentDevice));
  }

  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, const PDevice& device)
  {
    return NewVideoFrameOnDevice(vi, FRAME_ALIGN, (Device*)(void*)device);
  }

  PVideoFrame __stdcall GetOnDeviceFrame(const PVideoFrame& src, Device* device)
  {
    return core->GetOnDeviceFrame(src, device);
  }

  void __stdcall CopyFrameProps(PVideoFrame src, PVideoFrame dst) const
  {
    core->CopyFrameProps(src, dst);
  }

  ThreadPool* __stdcall NewThreadPool(size_t nThreads)
  {
    return core->NewThreadPool(nThreads);
  }

  AVSMap* __stdcall GetAVSMap(PVideoFrame& frame)
  {
    return core->GetAVSMap(frame);
  }

  void __stdcall AddRef() {
    InterlockedIncrement(&DISPATCH(refcount));
  }

  void __stdcall Release() {
    if (InterlockedDecrement(&DISPATCH(refcount)) == 0) {
      delete this;
    }
  }

  void __stdcall IncEnvCount() {
    core->IncEnvCount();
  }

  void __stdcall DecEnvCount() {
    core->DecEnvCount();
  }

  ConcurrentVarStringFrame* __stdcall GetTopFrame()
  {
    return core->GetTopFrame();
  }

  void __stdcall SetCacheMode(CacheMode mode)
  {
    core->SetCacheMode(mode);
  }

  CacheMode __stdcall GetCacheMode()
  {
    return core->GetCacheMode();
  }

	bool& __stdcall GetSupressCaching()
	{
		return DISPATCH(supressCaching);
	}

  void __stdcall SetDeviceOpt(DeviceOpt opt, int val)
  {
    core->SetDeviceOpt(opt, val);
  }

  void __stdcall UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar)
  {
    if (GetThreadId() != 0 || GetFrameRecursiveCount() != 0) {
      // no need to export function at runtime
      return;
    }
    core->UpdateFunctionExports(funcName, funcParams, exportVar);
  }

	bool __stdcall MakePropertyWritable(PVideoFrame* pvf)
	{
		return core->MakePropertyWritable(pvf);
	}

  InternalEnvironment* __stdcall NewThreadScriptEnvironment(int thread_id)
  {
    return new ThreadScriptEnvironment(thread_id, core, coreTLS);
  }

	int __stdcall GetThreadId() {
		return DISPATCH(thread_id);
	}

	int& __stdcall GetFrameRecursiveCount() {
		return DISPATCH(getFrameRecursiveCount);
	}

	int& __stdcall GetSuppressThreadCount() {
		return DISPATCH(suppressThreadCount);
	}

	FilterGraphNode*& GetCurrentGraphNode() {
		return DISPATCH(currentGraphNode);
	}

#undef DISPATCH
};

static unsigned __int64 ConstrainMemoryRequest(unsigned __int64 requested)
{
  // Get system memory information
  MEMORYSTATUSEX memstatus;
  memstatus.dwLength = sizeof(memstatus);
  GlobalMemoryStatusEx(&memstatus);

  // mem_limit is the largest amount of memory that makes sense to use.
  // We don't want to use more than the virtual address space,
  // and we also don't want to start paging to disk.
  unsigned __int64 mem_limit = min(memstatus.ullTotalVirtual, memstatus.ullTotalPhys);

  unsigned __int64 mem_sysreserve = 0;
  if (memstatus.ullTotalPhys > memstatus.ullTotalVirtual)
  {
    // We are probably running on a 32bit OS system where the virtual space is capped to
    // much less than what the system can use, so it is enough to reserve only a small amount.
    mem_sysreserve = 128 * 1024 * 1024ull;
  }
  else
  {
    // We could probably use up all the RAM in our single application,
    // so reserve more to leave some RAM for other apps and the OS too.
    mem_sysreserve = 1024 * 1024 * 1024ull;
  }

  // Cap memory_max to at most mem_sysreserve less than total, but at least to 64MB.
  return clamp(requested, 64 * 1024 * 1024ull, mem_limit - mem_sysreserve);
}

IJobCompletion* ScriptEnvironment::NewCompletion(size_t capacity)
{
  return new JobCompletion(capacity);
}

ScriptEnvironment::ScriptEnvironment()
  : threadEnv(),
  at_exit(),
  vsprintf_buf(NULL),
  vsprintf_len(0),
  plugin_manager(NULL),
  hrfromcoinit(E_FAIL), coinitThreadId(0),
  PlanarChromaAlignmentState(true),   // Change to "true" for 2.5.7
  thread_pool(NULL),
  EnvCount(0),
  Devices(),
  FrontCache(NULL),
  graphAnalysisEnable(false),
  nTotalThreads(1),
  nMaxFilterInstances(1)
{
  try {
    // Make sure COM is initialised
    hrfromcoinit = CoInitialize(NULL);

    threadEnv = std::unique_ptr<ThreadScriptEnvironment>(new ThreadScriptEnvironment(0, this, nullptr));
    Devices = std::unique_ptr<DeviceManager>(new DeviceManager(threadEnv.get()));

		// calc frame align
		frame_align = plane_align = FRAME_ALIGN;
		for (int i = 0, end = Devices->GetNumDevices(DEV_TYPE_CUDA); i < end; ++i) {
			int align, pitchAlign;
			Devices->GetDevice(DEV_TYPE_CUDA, i)->GetAlignmentRequirement(&align, &pitchAlign);
			frame_align = max(frame_align, pitchAlign);
			plane_align = max(plane_align, align);
		}

    // If it was already init'd then decrement
    // the use count and leave it alone!
    if (hrfromcoinit == S_FALSE) {
      hrfromcoinit = E_FAIL;
      CoUninitialize();
    }
    // Remember our threadId.
    coinitThreadId = GetCurrentThreadId();

    auto cpuDevice = Devices->GetCPUDevice();
		threadEnv->GetTLS()->currentDevice = cpuDevice;

    MEMORYSTATUSEX memstatus;
    memstatus.dwLength = sizeof(memstatus);
    GlobalMemoryStatusEx(&memstatus);
    cpuDevice->memory_max = ConstrainMemoryRequest(memstatus.ullTotalPhys / 4);
    const bool isX64 = sizeof(void *) == 8;
    cpuDevice->memory_max = min(cpuDevice->memory_max, (isX64 ? 4096 : 1024)*(1024*1024ull));  // at start, cap memory usage to 1GB(x86)/4GB (x64)
    cpuDevice->memory_used = 0ull;

    top_frame.Set("true", true);
    top_frame.Set("false", false);
    top_frame.Set("yes", true);
    top_frame.Set("no", false);
    top_frame.Set("last", AVSValue());

    top_frame.Set("$ScriptName$", AVSValue());
    top_frame.Set("$ScriptFile$", AVSValue());
    top_frame.Set("$ScriptDir$",  AVSValue());
    top_frame.Set("$ScriptNameUtf8$", AVSValue());
    top_frame.Set("$ScriptFileUtf8$", AVSValue());
    top_frame.Set("$ScriptDirUtf8$", AVSValue());

    plugin_manager = new PluginManager(threadEnv.get());
    plugin_manager->AddAutoloadDir("USER_PLUS_PLUGINS", false);
    plugin_manager->AddAutoloadDir("MACHINE_PLUS_PLUGINS", false);
    plugin_manager->AddAutoloadDir("USER_CLASSIC_PLUGINS", false);
    plugin_manager->AddAutoloadDir("MACHINE_CLASSIC_PLUGINS", false);

    top_frame.Set("LOG_ERROR",   (int)LOGLEVEL_ERROR);
    top_frame.Set("LOG_WARNING", (int)LOGLEVEL_WARNING);
    top_frame.Set("LOG_INFO",    (int)LOGLEVEL_INFO);
    top_frame.Set("LOG_DEBUG",   (int)LOGLEVEL_DEBUG);

    top_frame.Set("DEV_TYPE_CPU", (int)DEV_TYPE_CPU);
    top_frame.Set("DEV_TYPE_CUDA", (int)DEV_TYPE_CUDA);

    top_frame.Set("CACHE_FAST_START", (int)CACHE_FAST_START);
    top_frame.Set("CACHE_OPTIMAL_SIZE", (int)CACHE_OPTIMAL_SIZE);
    top_frame.Set("DEV_CUDA_PINNED_HOST", (int)DEV_CUDA_PINNED_HOST);
    top_frame.Set("DEV_FREE_THRESHOLD", (int)DEV_FREE_THRESHOLD);

    InitMT();
    thread_pool = new ThreadPool(std::thread::hardware_concurrency(), 1, threadEnv.get());

    ExportBuiltinFilters();

    clip_data.max_load_factor(0.8f);
    LogTickets.max_load_factor(0.8f);
  }
  catch (const AvisynthError &err) {
    if (SUCCEEDED(hrfromcoinit)) {
      hrfromcoinit = E_FAIL;
      CoUninitialize();
    }
    // Needs must, to not loose the text we
    // must leak a little memory.
    throw AvisynthError(_strdup(err.msg));
  }
}

MtMode ScriptEnvironment::GetDefaultMtMode() const
{
  return DefaultMtMode;
}

void ScriptEnvironment::InitMT()
{
  top_frame.Set("MT_NICE_FILTER", (int)MT_NICE_FILTER);
  top_frame.Set("MT_MULTI_INSTANCE", (int)MT_MULTI_INSTANCE);
  top_frame.Set("MT_SERIALIZED", (int)MT_SERIALIZED);
  top_frame.Set("MT_SPECIAL_MT", (int)MT_SPECIAL_MT);
}

ScriptEnvironment::~ScriptEnvironment() {

  _RPT0(0, "~ScriptEnvironment() called.\n");

  auto tls = threadEnv->GetTLS();
  tls->closing = true;

  // Before we start to pull the world apart
  // give every one their last wish.
  at_exit.Execute(threadEnv.get());

  delete thread_pool;

  tls->var_table.Clear();
  top_frame.Clear();

  // There can be a circular reference between the Prefetcher and the
  // TLS PopContext() variables of the threads started by it. Normally
  // this doesn't happen, but it can for example when somebody
  // sets 'last' in a TLS (see ScriptClip for a specific example).
  // This circular reference causes leaks, so we call
  // Destroy() on the prefetcher, which will in turn terminate all
  // its TLS stuff and break the chain.
  for (auto& pool : ThreadPoolRegistry) {
    pool->Join();
  }
  ThreadPoolRegistry.clear();

  // delete ThreadScriptEnvironment
  threadEnv = nullptr;

  // check ThreadScriptEnvironment leaks
  if (EnvCount > 0) {
    LogMsg(LOGLEVEL_WARNING, "ThreadScriptEnvironment leaks.");
  }

#if 0
	// check clip leaks DoDumpGraph
	if (std::find_if(GraphNodeRegistry.begin(), GraphNodeRegistry.end(),
		[](FilterGraphNode* node) { return node != nullptr; }) != GraphNodeRegistry.end())
	{
		// This is dangerous operation because thread's string is destroyed
		// and may be there are dangling string pointer which results in access violation.
		MinimumScriptEnvironment env(&top_frame);
		DoDumpGraph(GraphNodeRegistry, "clip_leaks.txt", &env);
	}
#endif

  // delete avsmap
  for (FrameRegistryType2::iterator it = FrameRegistry2.begin(), end_it = FrameRegistry2.end();
    it != end_it;
    ++it)
  {
    for (FrameBufferRegistryType::iterator it2 = (it->second).begin(), end_it2 = (it->second).end();
      it2 != end_it2;
      ++it2)
    {
      for (VideoFrameArrayType::iterator it3 = it2->second.begin(), end_it3 = it2->second.end();
        it3 != end_it3;
        ++it3)
      {
        delete it3->avsmap;
        it3->avsmap = 0;
        it3->frame->avsmap = 0;
      }
    }
  }

  // and deleting the frame buffer from FrameRegistry2 as well
  bool somethingLeaks = false;
  for (FrameRegistryType2::iterator it = FrameRegistry2.begin(), end_it = FrameRegistry2.end();
    it != end_it;
    ++it)
  {
    for (FrameBufferRegistryType::iterator it2 = (it->second).begin(), end_it2 = (it->second).end();
      it2 != end_it2;
      ++it2)
    {
      VFBStorage *vfb = static_cast<VFBStorage*>(it2->first);
      delete vfb;
      // iterate through frames belonging to this vfb
      for (VideoFrameArrayType::iterator it3 = it2->second.begin(), end_it3 = it2->second.end();
        it3 != end_it3;
        ++it3)
      {
        VideoFrame *frame = it3->frame;

        frame->vfb = 0;

        //assert(0 == frame->refcount);
        if (0 == frame->refcount)
        {
          delete frame;
        }
        else
        {
          somethingLeaks = true;
        }
      } // it3
    } // it2
  } // it

  if (somethingLeaks) {
    LogMsg(LOGLEVEL_WARNING, "A plugin or the host application might be causing memory leaks.");
  }

  delete plugin_manager;
  delete[] vsprintf_buf;

  // If we init'd COM and this is the right thread then release it
  // If it's the wrong threadId then tuff, nothing we can do.
  if (SUCCEEDED(hrfromcoinit) && (coinitThreadId == GetCurrentThreadId())) {
    hrfromcoinit = E_FAIL;
    CoUninitialize();
  }
}

void ScriptEnvironment::SetLogParams(const char *target, int level)
{
  if (nullptr == target) {
    target = "stderr";
  }

  if (-1 == level) {
    level = LOGLEVEL_INFO;
  }

  if (LogFileStream.is_open()) {
    LogFileStream.close();
  }

  LogLevel = LOGLEVEL_NONE;

  if (!streqi(target, "stderr") && !streqi(target, "stdout")) {
    LogFileStream.open(target, std::ofstream::out | std::ofstream::app);
    if (LogFileStream.fail()) {
      this->ThrowError("SetLogParams: Could not open file \"%s\" for writing.", target);
      return;
    }
  }

  LogLevel = level;
  LogTarget = target;
}

void ScriptEnvironment::LogMsg(int level, const char* fmt, ...)
{
  va_list val;
  va_start(val, fmt);
  LogMsg_valist(level, fmt, val);
  va_end(val);
}

void ScriptEnvironment::LogMsg_valist(int level, const char *fmt, va_list va)
{
  // Don't output message if our logging level is not high enough
  if (level > LogLevel) {
    return;
  }

  // Setup string prefixes for output messages
  const char *levelStr = nullptr;
  WORD levelAttr;
  switch (level)
  {
  case LOGLEVEL_ERROR:
    levelStr = "ERROR: ";
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_RED;
    break;
  case LOGLEVEL_WARNING:
    levelStr = "WARNING: ";
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED;
    break;
  case LOGLEVEL_INFO:
    levelStr = "INFO: ";
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE;
    break;
  case LOGLEVEL_DEBUG:
    levelStr = "DEBUG: ";
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_BLUE | FOREGROUND_RED;
    break;
  default:
    this->ThrowError("LogMsg: level argument must be between 1 and 4.");
    break;
  }

  // Prepare message output target
  std::ostream *targetStream = nullptr;
  HANDLE hConsole = GetStdHandle(STD_ERROR_HANDLE);

  if (streqi("stderr", LogTarget.c_str()))
  {
    hConsole = GetStdHandle(STD_ERROR_HANDLE);
    targetStream = &std::cerr;
  }
  else if (streqi("stdout", LogTarget.c_str()))
  {
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    targetStream = &std::cout;
  }
  else if (LogFileStream.is_open())
  {
    targetStream = &LogFileStream;
  }
  else
  {
    // Logging not yet set up (SetLogParams() not yet called).
    // Do nothing.
    return;
  }

  // Format our message string
  std::string msg = FormatString(fmt, va);

  // Save current console attributes so that we can restore them later
  CONSOLE_SCREEN_BUFFER_INFO Info;
  GetConsoleScreenBufferInfo(hConsole, &Info);

  // Do the output
  std::lock_guard<std::mutex> lock(string_mutex);
  *targetStream << "---------------------------------------------------------------------" << std::endl;
  SetConsoleTextAttribute(hConsole, levelAttr);
  *targetStream << levelStr;
  SetConsoleTextAttribute(hConsole, Info.wAttributes);
  *targetStream << msg << std::endl;
  targetStream->flush();
}

void ScriptEnvironment::LogMsgOnce(const OneTimeLogTicket &ticket, int level, const char* fmt, ...)
{
  va_list val;
  va_start(val, fmt);
  LogMsgOnce_valist(ticket, level, fmt, val);
  va_end(val);
}

void ScriptEnvironment::LogMsgOnce_valist(const OneTimeLogTicket &ticket, int level, const char *fmt, va_list va)
{
  if (LogTickets.end() == LogTickets.find(ticket))
  {
    LogMsg_valist(level, fmt, va);
    LogTickets.insert(ticket);
  }
}

ClipDataStore* ScriptEnvironment::ClipData(IClip *clip)
{
#if ( !defined(_MSC_VER) || (_MSC_VER < 1900) )
  return &(clip_data.emplace(clip, clip).first->second);
#else
  return &(clip_data.try_emplace(clip, clip).first->second);
#endif
}

void ScriptEnvironment::AdjustMemoryConsumption(size_t amount, bool minus)
{
  if (minus)
    Devices->GetCPUDevice()->memory_used -= amount;
  else
    Devices->GetCPUDevice()->memory_used += amount;
}

void ScriptEnvironment::ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion)
{
  thread_pool->QueueJob(jobFunc, jobData, threadEnv.get(), static_cast<JobCompletion*>(completion));
}

void ScriptEnvironment::ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion, InternalEnvironment* env)
{
  thread_pool->QueueJob(jobFunc, jobData, env, static_cast<JobCompletion*>(completion));
}

void ScriptEnvironment::SetFilterMTMode(const char* filter, MtMode mode, bool force)
{
  this->SetFilterMTMode(filter, mode, force ? MtWeight::MT_WEIGHT_2_USERFORCE : MtWeight::MT_WEIGHT_1_USERSPEC);
}

void ScriptEnvironment::SetFilterMTMode(const char* filter, MtMode mode, MtWeight weight)
{
  assert(NULL != filter);
  assert(strcmp("", filter) != 0);

  if (((int)mode <= (int)MT_INVALID)
    || ((int)mode >= (int)MT_MODE_COUNT))
  {
    throw AvisynthError("Invalid MT mode specified.");
  }

  if (streqi(filter, DEFAULT_MODE_SPECIFIER.c_str()))
  {
    DefaultMtMode = mode;
    return;
  }

  std::string name_to_register;
  std::string loading;
  {
    std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
    loading = plugin_manager->PluginLoading();
  }
  if (loading.empty())
    name_to_register = filter;
  else
    name_to_register = loading.append("_").append(filter);

  name_to_register = NormalizeString(name_to_register);

  auto it = MtMap.find(name_to_register);
  if (it != MtMap.end())
  {
    if ((int)weight >= (int)(it->second.second))
    {
      it->second.first = mode;
      it->second.second = weight;
    }
  }
  else
  {
    MtMap.emplace(name_to_register, std::make_pair(mode, weight));
  }
}

bool ScriptEnvironment::FilterHasMtMode(const Function* filter) const
{
  if (filter->name == nullptr) { // no named function
    return false;
  }
  const auto &end = MtMap.end();
  return (end != MtMap.find(NormalizeString(filter->canon_name)))
    || (end != MtMap.find(NormalizeString(filter->name)));
}

MtMode ScriptEnvironment::GetFilterMTMode(const Function* filter, bool* is_forced) const
{
  assert(NULL != filter);
  if (filter->name == nullptr) { // no named function
    *is_forced = false;
    return DefaultMtMode;
  }

  assert(NULL != filter->name);
  assert(NULL != filter->canon_name);

  auto it = MtMap.find(NormalizeString(filter->canon_name));
  if (it != MtMap.end())
  {
    *is_forced = it->second.second == MtWeight::MT_WEIGHT_2_USERFORCE;
    return it->second.first;
  }

  it = MtMap.find(NormalizeString(filter->name));
  if (it != MtMap.end())
  {
    *is_forced = it->second.second == MtWeight::MT_WEIGHT_2_USERFORCE;
    return it->second.first;
  }

  *is_forced = false;
  return DefaultMtMode;
}

/* This function adds information about builtin functions into global variables.
 * External utilities (like AvsPmod) can parse these variables and use them
 * to learn about supported functions and their syntax.
 */
void ScriptEnvironment::ExportBuiltinFilters()
{
  std::string FunctionList;
  FunctionList.reserve(512);
  const size_t NumFunctionArrays = sizeof(builtin_functions) / sizeof(builtin_functions[0]);
  for (size_t i = 0; i < NumFunctionArrays; ++i)
  {
    for (const AVSFunction* f = builtin_functions[i]; !f->empty(); ++f)
    {
      // This builds the $InternalFunctions$ variable, which is a list of space-delimited
      // function names. Utilities can learn the names of the builtin function from this.
      FunctionList.append(f->name);
      FunctionList.push_back(' ');

      // For each supported function, a global variable is added with <param_var_name> as the name,
      // and the list of parameters to that function as the value.
      std::string param_var_name;
      param_var_name.reserve(128);
      param_var_name.append("$Plugin!");
      param_var_name.append(f->name);
      param_var_name.append("!Param$");
      threadEnv->SetGlobalVar(threadEnv->SaveString(param_var_name.c_str(), (int)param_var_name.size()), AVSValue(f->param_types));
    }
  }

  // Save $InternalFunctions$
  threadEnv->SetGlobalVar("$InternalFunctions$", AVSValue(threadEnv->SaveString(FunctionList.c_str(), (int)FunctionList.size())));
}

size_t  ScriptEnvironment::GetProperty(AvsEnvProperty prop)
{
  switch (prop)
  {
  case AEP_NUM_DEVICES:
    return Devices->GetNumDevices();
  case AEP_FRAME_ALIGN:
    return frame_align;
	case AEP_PLANE_ALIGN:
		return plane_align;
  case AEP_FILTERCHAIN_THREADS:
    return nMaxFilterInstances;
  case AEP_PHYSICAL_CPUS:
    return GetNumPhysicalCPUs();
  case AEP_LOGICAL_CPUS:
    return std::thread::hardware_concurrency();
  case AEP_THREAD_ID:
    return 0;
  case AEP_THREADPOOL_THREADS:
    return thread_pool->NumThreads();
  case AEP_VERSION:
    return AVS_SEQREV;
  default:
    this->ThrowError("Invalid property request.");
    return std::numeric_limits<size_t>::max();
  }

  assert(0);
}

bool ScriptEnvironment::LoadPlugin(const char* filePath, bool throwOnError, AVSValue *result)
{
  // Autoload needed to ensure that manual LoadPlugin() calls always override autoloaded plugins.
  // For that, autoloading must happen before any LoadPlugin(), so we force an
  // autoload operation before any LoadPlugin().
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);

  this->AutoloadPlugins();
  return plugin_manager->LoadPlugin(filePath, throwOnError, result);
}

void ScriptEnvironment::AddAutoloadDir(const char* dirPath, bool toFront)
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->AddAutoloadDir(dirPath, toFront);
}

void ScriptEnvironment::ClearAutoloadDirs()
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->ClearAutoloadDirs();
}

void ScriptEnvironment::AutoloadPlugins()
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->AutoloadPlugins();
}

int ScriptEnvironment::SetMemoryMax(int mem) {

  Device* cpuDevice = Devices->GetCPUDevice();
  if (mem > 0)  /* If mem is zero, we should just return current setting */
    cpuDevice->memory_max = ConstrainMemoryRequest(mem * 1048576ull);

  return (int)(cpuDevice->memory_max / 1048576ull);
}

int ScriptEnvironment::SetWorkingDir(const char * newdir) {
  return SetCurrentDirectory(newdir) ? 0 : 1;
}

void ScriptEnvironment::CheckVersion(int version) {
  if (version > AVISYNTH_INTERFACE_VERSION)
    ThrowError("Plugin was designed for a later version of Avisynth (%d)", version);
}

int ScriptEnvironment::GetCPUFlags() { return ::GetCPUFlags(); }

void ScriptEnvironment::AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data) {
  this->AddFunction(name, params, apply, user_data, NULL);
}

void ScriptEnvironment::AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data, const char *exportVar) {
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->AddFunction(name, params, apply, user_data, exportVar);
}

VideoFrame* ScriptEnvironment::AllocateFrame(size_t vfb_size, size_t margin, Device* device)
{
  if (vfb_size > (size_t)std::numeric_limits<int>::max())
  {
    throw AvisynthError(threadEnv->Sprintf("Requested buffer size of %zu is too large", vfb_size));
  }

  VFBStorage* vfb = NULL;
  try
  {
    vfb = new VFBStorage((int)vfb_size, margin, device);
  }
  catch(const std::bad_alloc&)
  {
    return NULL;
  }

  VideoFrame *newFrame = NULL;
  try
  {
    newFrame = new VideoFrame(vfb, new AVSMap(), 0, 0, 0, 0);
  }
  catch(const std::bad_alloc&)
  {
    delete vfb;
    return NULL;
  }

  device->memory_used += vfb_size;
  vfb->Attach(threadEnv->GetCurrentGraphNode());

  // automatically inserts keys if they not exist!
  // no locking here, calling method have done it already
  FrameRegistry2[vfb_size][vfb].push_back(DebugTimestampedFrame(newFrame, newFrame->avsmap));

  //_RPT1(0, "ScriptEnvironment::AllocateFrame %Iu frame=%p vfb=%p %I64d\n", vfb_size, newFrame, newFrame->vfb, memory_used); // P.F.

  return newFrame;
}

#ifdef _DEBUG
void ScriptEnvironment::ListFrameRegistry(size_t min_size, size_t max_size, bool someframes)
{
  int linearsearchcount;
  //#define FULL_LIST_OF_VFBs
  //#define LIST_ALSO_SOME_FRAMES
  linearsearchcount = 0;
  int size1 = 0;
  int size2 = 0;
  int size3 = 0;
  int inner_frame_count = 0;
  int inner_frame_count_for_frame_refcount_nonzero = 0;
  _RPT3(0, "******** %p <= FrameRegistry2 Address. Buffer list for size between %7Iu and %7Iu\n", &FrameRegistry2, min_size, max_size);
  _RPT1(0, ">> IterateLevel #1: Different vfb sizes: FrameRegistry2.size=%Iu \n", FrameRegistry2.size());
  size_t total_vfb_size = 0;
  // list to debugview: all frames up-to vfb_size size
  for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(min_size), end_it = FrameRegistry2.upper_bound(max_size);
    it != end_it;
    ++it)
  {
    size1++;
    _RPT3(0, ">>>> IterateLevel #2 [%3d]: Vfb count for size %7Iu is %7Iu\n", size1, it->first, it->second.size());
    for (FrameBufferRegistryType::iterator it2 = it->second.begin(), end_it2 = it->second.end();
      it2 != end_it2;
      ++it2)
    {
      size2++;
      VideoFrameBuffer *vfb = it2->first;
      total_vfb_size += vfb->GetDataSize();
      size_t inner_frame_count_size = it2->second.size();
      char buf[128];
      _snprintf(buf, 127, ">>>> IterateLevel #3 %5Iu frames in [%3d,%5d] --> vfb=%p vfb_refcount=%3d seqNum=%d\n", inner_frame_count_size, size1, size2, vfb, vfb->refcount, vfb->GetSequenceNumber());
      _RPT0(0, buf); // P.F. iterate the frame list of this vfb
      int inner_frame_count = 0;
      int inner_frame_count_for_frame_refcount_nonzero = 0;
      for (VideoFrameArrayType::iterator it3 = it2->second.begin(), end_it3 = it2->second.end();
        it3 != end_it3;
        ++it3)
      {
        size3++;
        inner_frame_count++;
#ifdef _DEBUG
        VideoFrame *frame = it3->frame;
        std::chrono::time_point<std::chrono::high_resolution_clock> frame_entry_timestamp = it3->timestamp;
#else
        VideoFrame *frame = *it3;
#endif
        if (0 != frame->refcount)
          inner_frame_count_for_frame_refcount_nonzero++;
        if (someframes)
        {
          std::chrono::time_point<std::chrono::high_resolution_clock> t_end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed_seconds = t_end - frame_entry_timestamp;
          if (inner_frame_count <= 2) // list only the first 2. There can be even many thousand of frames!
          {
            // log only if frame creation timestamp is too old!
            // e.g. 100 secs, it must be a stuck frame (but can also be a valid static frame from ColorBars)
            // if (elapsed_seconds.count() > 100.0f && frame->refcount > 0)
            if (frame->refcount > 0)
            {
              _RPT5(0, "  >> Frame#%6d: vfb=%p frame=%p frame_refcount=%3d timestamp=%f ago\n", inner_frame_count, vfb, frame, frame->refcount, elapsed_seconds);
            }
          }
          else if (inner_frame_count == inner_frame_count_size - 1)
          {
            // log the last one
            if (frame->refcount > 0)
            {
              _RPT4(0, "  ...Frame#%6d: vfb=%p frame=%p frame_refcount=%3d \n", inner_frame_count, vfb, frame, frame->refcount);
            }
            _RPT2(0, "  == TOTAL of %d frames. Number of nonzero refcount=%d \n", inner_frame_count, inner_frame_count_for_frame_refcount_nonzero);
          }
          if (0 == vfb->refcount && 0 != frame->refcount)
          {
            _RPT3(0, "  ########## VFB=0 FRAME!=0 ####### VFB: %p Frame:%p frame_refcount=%3d \n", vfb, frame, frame->refcount);
          }
        }
      }
    }
  }
  _RPT4(0, ">> >> >> array sizes %d %d %d Total VFB size=%Iu\n", size1, size2, size3, total_vfb_size);
  _RPT0(0, "  ----------------------------\n");
}
#endif

VideoFrame* ScriptEnvironment::GetFrameFromRegistry(size_t vfb_size, Device* device)
{
#ifdef _DEBUG
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end; // std::chrono::time_point<std::chrono::system_clock> t_start, t_end;
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // FrameRegistry2 is like: map<key1, map<key2, vector<VideoFrame *>> >
  // typedef std::vector<VideoFrame*> VideoFrameArrayType;
  // typedef std::map<VideoFrameBuffer *, VideoFrameArrayType> FrameBufferRegistryType;
  // typedef std::map<size_t, FrameBufferRegistryType> FrameRegistryType2;
  // [vfb_size = 10000][vfb = 0x111111111] [frame = 0x129837192(,timestamp=xxx)]
  //                                       [frame = 0x012312122(,timestamp=xxx)]
  //                                       [frame = 0x232323232(,timestamp=xxx)]
  //                   [vfb = 0x222222222] [frame = 0x333333333(,timestamp=xxx)]
  //                                       [frame = 0x444444444(,timestamp=xxx)]
  // Which is better?
  // - found exact vfb_size or
  // - allow reusing existing vfb's with size up to size_to_find*1.5 THIS ONE!
  // - allow to occupy any buffer that is bigger than the requested size
  //for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(vfb_size), end_it = FrameRegistry2.upper_bound(vfb_size); // exact! no-go. special service clips can fragment it
  //for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(vfb_size), end_it = FrameRegistry2.end(); // vfb_size or bigger, so a 100K size would claim a 1.5M space.
  for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(vfb_size), end_it = FrameRegistry2.upper_bound(vfb_size * 3 / 2); // vfb_size or at most 1.5* bigger
    it != end_it;
    ++it)
  {
    for (FrameBufferRegistryType::iterator it2 = it->second.begin(), end_it2 = it->second.end();
      it2 != end_it2;
      ++it2)
    {
      VFBStorage *vfb = static_cast<VFBStorage*>(it2->first); // same for all map content, the key is vfb pointer
      if (device == vfb->device && 0 == vfb->refcount) // vfb device and refcount check
      {
        size_t videoFrameListSize = it2->second.size();
        VideoFrame *frame_found;
        AVSMap *map_found;
        bool found = false;
        for (VideoFrameArrayType::iterator it3 = it2->second.begin(), end_it3 = it2->second.end();
          it3 != end_it3;
          /*++it3 not here, because of the delete*/)
        {
          VideoFrame *frame = it3->frame;

          // sanity check if its refcount is zero
          // because when a vfb is free (refcount==0) then all its parent frames should also be free
          assert(0 == frame->refcount);
          assert(0 == frame->avsmap->data.size());

          if (!found)
          {
            InterlockedIncrement(&(frame->vfb->refcount)); // same as &(vfb->refcount)
            vfb->free_count = 0; // reset free count
            vfb->Attach(threadEnv->GetCurrentGraphNode());
#ifdef _DEBUG
            char buf[256];
            t_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = t_end - t_start;
            _snprintf(buf, 255, "ScriptEnvironment::GetNewFrame NEW METHOD EXACT hit! VideoFrameListSize=%7Iu GotSize=%7Iu FrReg.Size=%6Iu vfb=%p frame=%p SeekTime:%f\n", videoFrameListSize, vfb_size, FrameRegistry2.size(), vfb, frame, elapsed_seconds.count());
            _RPT0(0, buf);
#ifdef SIZETMOD
            _RPT5(0, "                                          frame %p RowSize=%d Height=%d Pitch=%d Offset=%7Iu\n", frame, frame->GetRowSize(), frame->GetHeight(), frame->GetPitch(), frame->GetOffset()); // P.F.
#else
            _RPT5(0, "                                          frame %p RowSize=%d Height=%d Pitch=%d Offset=%d\n", frame, frame->GetRowSize(), frame->GetHeight(), frame->GetPitch(), frame->GetOffset()); // P.F.
#endif
#endif
                                                                                                                                                                                                             // only 1 frame in list -> no delete
            if (videoFrameListSize <= 1)
            {
              _RPT1(0, "ScriptEnvironment::GetNewFrame returning frame. VideoFrameListSize was 1\n", videoFrameListSize); // P.F.
#ifdef _DEBUG
              it3->timestamp = std::chrono::high_resolution_clock::now(); // refresh timestamp!
#endif
              return frame; // return immediately
              break;
            }
            // more than X: just registered the frame found, and erase all other frames from list plus delete frame objects also
            frame_found = frame;
            map_found = it3->avsmap;
            found = true;
            ++it3;
          }
          else {
            // if the first frame to this vfb was already found, then we free all others and delete it from the list
            // Benefit: no 4-5k frame list count per a single vfb.
            //_RPT4(0, "ScriptEnvironment::GetNewFrame Delete one frame %p RowSize=%d Height=%d Pitch=%d Offset=%d\n", frame, frame->GetRowSize(), frame->GetHeight(), frame->GetPitch(), frame->GetOffset()); // P.F.
            delete frame;
            delete it3->avsmap;
            ++it3;
          }
        } // for it3
        if (found)
        {
          _RPT1(0, "ScriptEnvironment::GetNewFrame returning frame_found. clearing frames. List count: it2->second.size(): %7zu \n", it2->second.size());
          it2->second.clear();
          it2->second.reserve(16); // initial capacity set to 16, avoid reallocation when 1st, 2nd, etc.. elements pushed later (possible speedup)
          it2->second.push_back(DebugTimestampedFrame(frame_found, map_found)); // keep only the first
          return frame_found;
        }
      }
    } // for it2
  } // for it
  _RPT3(0, "ScriptEnvironment::GetNewFrame, no free entry in FrameRegistry. Requested vfb size=%Iu memused=%I64d memmax=%I64d\n", vfb_size, device->memory_used.load(), device->memory_max);

#ifdef _DEBUG
  //ListFrameRegistry(vfb_size, vfb_size, true); // for chasing stuck frames
  //ListFrameRegistry(0, vfb_size, true); // for chasing stuck frames
#endif

  return NULL;
}

VideoFrame* ScriptEnvironment::GetNewFrame(size_t vfb_size, size_t margin, Device* device)
{
  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex);
#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT1(0, "ScScriptEnvironment::GetNewFrame memory mutex lock: %p\n", (void *)&memory_mutex);
#endif

  // prevent fragmentation of vfb buffer list many different small-sized vfb's
  if (vfb_size < 64) vfb_size = 64;
  else if (vfb_size < 256) vfb_size = 256;
  else if (vfb_size < 512) vfb_size = 512;
  else if (vfb_size < 1024) vfb_size = 1024;
  else if (vfb_size < 2048) vfb_size = 2048;
  else if (vfb_size < 4096) vfb_size = 4096;

  /* -----------------------------------------------------------
  *   Try to return an unused but already allocated instance
  * -----------------------------------------------------------
  */
  VideoFrame* frame = GetFrameFromRegistry(vfb_size, device);
  if (frame != NULL)
    return frame;

  /* -----------------------------------------------------------
   *   No unused instance was found, try to allocate a new one
   * -----------------------------------------------------------
   */
   // We reserve 15% for unaccounted stuff
  if (device->memory_used + vfb_size < device->memory_max * 0.85f) {
    frame = AllocateFrame(vfb_size, margin, device);
  }
  if ( frame != NULL)
    return frame;

#ifdef _DEBUG
  // #define LIST_CACHES
  // list all cache_entries
#ifdef LIST_CACHES
  int cache_counter = 0;
  const CacheRegistryType::iterator end_cit_0 = CacheRegistry.end();
  for (
    CacheRegistryType::iterator cit = CacheRegistry.begin();
    (cit != end_cit_0);
    ++cit
    )
  {
    cache_counter++;
    Cache* cache = *cit;
    int cache_size = cache->SetCacheHints(CACHE_GET_SIZE, 0);
    _RPT4(0, "  cache#%d cache_ptr=%p cache_size=%d \n", cache_counter, (void *)cache, cache_size); // let's see what's in the cache
  }
#endif
#endif

  /* -----------------------------------------------------------
  * Couldn't allocate, shrink cache and get more unused frames
  * -----------------------------------------------------------
  */
  ShrinkCache(device);

  /* -----------------------------------------------------------
  *   Try to return an unused frame again
  * -----------------------------------------------------------
  */
  frame = GetFrameFromRegistry(vfb_size, device);
  if (frame != NULL)
    return frame;

  /* -----------------------------------------------------------
  *   Try to allocate again
  * -----------------------------------------------------------
  */
  frame = AllocateFrame(vfb_size, margin, device);
  if (frame != NULL)
    return frame;

  OneTimeLogTicket ticket(LOGTICKET_W1100);
  LogMsgOnce(ticket, LOGLEVEL_WARNING, "Memory reallocation occurs. This will probably degrade performance. You can try increasing the limit using SetMemoryMax().");

  /* -----------------------------------------------------------
   * No frame found, free all the unused frames!!!
   * -----------------------------------------------------------
   */
  _RPT1(0, "Allocate failed. GC start memory_used=%I64d\n", device->memory_used.load());
  // unfortunately if we reach here, only 0 or 1 vfbs or frames can be freed, from lower vfb sizes
  // usually it's not enough
  // yet it is true that it's meaningful only to free up smaller vfb sizes here
  for (FrameRegistryType2::iterator it = FrameRegistry2.begin(), end_it = FrameRegistry2.upper_bound(vfb_size);
    it != end_it;
    ++it)
  {
    for (FrameBufferRegistryType::iterator it2 = (it->second).begin(), end_it2 = (it->second).end();
      it2 != end_it2;
      /*++it2: not here: may delete iterator position */)
    {
      VFBStorage *vfb = static_cast<VFBStorage*>(it2->first);
      if (device == vfb->device && 0 == vfb->refcount) // vfb refcount check
      {
        vfb->device->memory_used -= vfb->GetDataSize(); // frame->vfb->GetDataSize();
        delete vfb;
        const VideoFrameArrayType::iterator end_it3 = it2->second.end(); // const
        for (VideoFrameArrayType::iterator it3 = it2->second.begin();
          it3 != end_it3;
          ++it3)
        {
          VideoFrame *currentframe = it3->frame;
          assert(0 == currentframe->refcount);
          delete currentframe;
          delete it3->avsmap;
        }
        // delete array belonging to this vfb in one step
        it2->second.clear(); // clear frame list
        it2 = (it->second).erase(it2); // clear current vfb
      }
      else ++it2;
    }
  }
  _RPT1(0, "End of garbage collection A memused=%I64d\n", device->memory_used.load()); // P.F.
#if 0
  static int counter = 0;
  char buf[200]; sprintf(buf, "Re allocation %d\r\n", counter++);
  OutputDebugStringA(buf);
#endif
  /* -----------------------------------------------------------
   *   Try to allocate again
   * -----------------------------------------------------------
   */
  frame = AllocateFrame(vfb_size, margin, device);
  if ( frame != NULL)
    return frame;


  /* -----------------------------------------------------------
   *   Oh boy...
   * -----------------------------------------------------------
   */

  // See if we could benefit from 64-bit Avisynth
  if (sizeof(void*) == 4 && device == Devices->GetCPUDevice())
  {
    // Get system memory information
    MEMORYSTATUSEX memstatus;
    memstatus.dwLength = sizeof(memstatus);
    GlobalMemoryStatusEx(&memstatus);

    BOOL using_wow64;
    if (IsWow64Process(GetCurrentProcess(), &using_wow64)     // if running 32-bits on a 64-bit OS
      && (memstatus.ullAvailPhys > 1024ull * 1024 * 1024))    // if at least 1GB of system memory is still free
    {
      OneTimeLogTicket ticket(LOGTICKET_W1007);
      LogMsgOnce(ticket, LOGLEVEL_INFO, "We have run out of memory, but your system still has some free RAM left. You might benefit from a 64-bit build of Avisynth+.");
    }
  }

  ThrowError("Could not allocate video frame. Out of memory. memory_max = %I64d, memory_used = %I64d Request=%Iu", device->memory_max, device->memory_used.load(), vfb_size);
  return NULL;
}

void ScriptEnvironment::ShrinkCache(Device* device)
{
  /* -----------------------------------------------------------
  *   Shrink cache to keep memory limit
  * -----------------------------------------------------------
  */
  int shrinkcount = 0;

  const CacheRegistryType::iterator end_cit = CacheRegistry.end();
  for (
    CacheRegistryType::iterator cit = CacheRegistry.begin();
    cit != end_cit;
    ++cit
    )
  {
    // Oh darn. We'd need more memory than we are allowed to use.
    // Let's reduce the amount of caching.

    // We try to shrink least recently used caches first.

    Cache* cache = *cit;
    if (cache->GetDevice() != device) {
      continue;
    }
    int cache_size = cache->SetCacheHints(CACHE_GET_SIZE, 0);
    if (cache_size != 0)
    {
      _RPT2(0, "ScriptEnvironment::EnsureMemoryLimit shrink cache. cache=%p new size=%d\n", (void *)cache, cache_size - 1);
      cache->SetCacheHints(CACHE_SET_MAX_CAPACITY, cache_size - 1);
      shrinkcount++;
    } // if
  } // for cit

  if (shrinkcount != 0)
  {
    OneTimeLogTicket ticket(LOGTICKET_W1003);
    LogMsgOnce(ticket, LOGLEVEL_WARNING, "Caches have been shrunk due to low memory limit. This will probably degrade performance. You can try increasing the limit using SetMemoryMax().");
  }

  /* -----------------------------------------------------------
  * Count up free_count and free if it exceeds the threshold
  * -----------------------------------------------------------
  */
  // Free up in one pass in FrameRegistry2
  if (shrinkcount)
  {
    _RPT1(0, "EnsureMemoryLimit GC start: memused=%I64d\n", device->memory_used.load());
    int freed_vfb_count = 0;
    int freed_frame_count = 0;
    int unfreed_frame_count = 0;
    const FrameRegistryType2::iterator end_it = FrameRegistry2.end(); // const iterator. maybe need simial in the end of NewFrameBuffer
    for (FrameRegistryType2::iterator it = FrameRegistry2.begin();
      it != end_it;
      ++it)
    {
      for (FrameBufferRegistryType::iterator it2 = (it->second).begin(), end_it2 = (it->second).end();
        it2 != end_it2;
        /*++it2: not here: may delete iterator position */)
      {
        VFBStorage *vfb = static_cast<VFBStorage*>(it2->first);
        // vfb device and refcount check and free count exceeds the threshold
        if (device == vfb->device && 0 == vfb->refcount && vfb->free_count++ >= device->free_thresh)
        {
#if 0
          static int counter = 0;
          char buf[200]; sprintf(buf, "Free frame !!! %d\r\n", counter++);
          OutputDebugStringA(buf);
#endif
          device->memory_used -= vfb->GetDataSize();
          VFBStorage *_vfb = vfb;
          delete vfb;
          ++freed_vfb_count;
          const VideoFrameArrayType::iterator end_it3 = it2->second.end();
          for (VideoFrameArrayType::iterator it3 = it2->second.begin();
            it3 != end_it3;
            ++it3)
          {
            VideoFrame *frame = it3->frame;
            assert(0 == frame->refcount);
            if (0 == frame->refcount)
            {
              delete frame;
              delete it3->avsmap;
              ++freed_frame_count;
            }
            else {
              // there should not be such case: vfb.refcount=0 and frame.refcount!=0
              ++unfreed_frame_count;
              _RPT3(0, "  ?????? frame refcount error!!! _vfb=%p frame=%p framerefcount=%d \n", _vfb, frame, frame->refcount); // P.F.
            }
          }
          // delete array belonging to this vfb in one step
          it2->second.clear(); // clear frame list
          it2 = (it->second).erase(it2); // clear vfb entry
        }
        else ++it2;
      }
    }
    _RPT4(0, "End of garbage collection B: freed_vfb=%d frame=%d unfreed=%d memused=%I64d\n", freed_vfb_count, freed_frame_count, unfreed_frame_count, device->memory_used.load()); // P.F.
  }
}

// no alpha
PVideoFrame ScriptEnvironment::NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, Device* device)
{
  return NewPlanarVideoFrame(row_size, height, row_sizeUV, heightUV, align, U_first, false, device); // no alpha
}

// with alpha support
PVideoFrame ScriptEnvironment::NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, bool alpha, Device* device)
{
  if (align < 0)
  {
    align = -align;
    OneTimeLogTicket ticket(LOGTICKET_W1009);
    LogMsgOnce(ticket, LOGLEVEL_WARNING, "A filter is using forced frame alignment, a feature that is deprecated and disabled. The filter will likely behave erroneously.");
  }
  align = max(align, frame_align);

  int pitchUV;
  const int pitchY = AlignNumber(row_size, align);
  if (!PlanarChromaAlignmentState && (row_size == row_sizeUV*2) && (height == heightUV*2)) { // Meet old 2.5 series API expectations for YV12
    // Legacy alignment - pack Y as specified, pack UV half that
    pitchUV = (pitchY+1)>>1;  // UV plane, width = 1/2 byte per pixel - don't align UV planes seperately.
  }
  else {
    // Align planes separately
    pitchUV = AlignNumber(row_sizeUV, align);
  }

	size_t sizeY = AlignNumber(pitchY * height, plane_align);
	size_t sizeUV = AlignNumber(pitchUV * heightUV, plane_align);
  size_t size = sizeY + 2 * sizeUV + (alpha ? sizeY : 0);

  VideoFrame *res = GetNewFrame(size, align - 1, device);

  int  offsetU, offsetV, offsetA;
  const int offsetY = (int)(AlignPointer(res->vfb->GetWritePtr(), align) - res->vfb->GetWritePtr()); // first line offset for proper alignment
  if (U_first) {
    offsetU = offsetY + sizeY;
    offsetV = offsetY + sizeY + sizeUV;
    offsetA = alpha ? offsetV + sizeUV : 0;
  } else {
    offsetV = offsetY + sizeY;
    offsetU = offsetY + sizeY + sizeUV;
    offsetA = alpha ? offsetU + sizeUV : 0;
  }

  res->offset = offsetY;
  res->pitch = pitchY;
  res->row_size = row_size;
  res->height = height;
  res->offsetU = offsetU;
  res->offsetV = offsetV;
  res->pitchUV = pitchUV;
  res->row_sizeUV = row_sizeUV;
  res->heightUV = heightUV;
  // alpha support
  res->offsetA = offsetA;
  res->row_sizeA = alpha ? row_size : 0;
  res->pitchA = alpha ? pitchY : 0;


  return PVideoFrame(res);
}


PVideoFrame ScriptEnvironment::NewVideoFrameOnDevice(int row_size, int height, int align, Device* device)
{
  if (align < 0)
  {
    align = -align;
    OneTimeLogTicket ticket(LOGTICKET_W1009);
    this->LogMsgOnce(ticket, LOGLEVEL_WARNING, "A filter is using forced frame alignment, a feature that is deprecated and disabled. The filter will likely behave erroneously.");
  }
  align = max(align, frame_align);

  const int pitch = AlignNumber(row_size, align);
  size_t size = pitch * height;

  VideoFrame *res = GetNewFrame(size, align - 1, device);

  const int offset = (int)(AlignPointer(res->vfb->GetWritePtr(), align) - res->vfb->GetWritePtr()); // first line offset for proper alignment

  res->offset = offset;
  res->pitch = pitch;
  res->row_size = row_size;
  res->height = height;
  res->offsetU = offset;
  res->offsetV = offset;
  res->pitchUV = 0;
  res->row_sizeUV = 0;
  res->heightUV = 0;
  // alpha support
  res->offsetA = 0;
  res->row_sizeA = 0;
  res->pitchA = 0;


  return PVideoFrame(res);
}

PVideoFrame ScriptEnvironment::NewVideoFrame(const VideoInfo& vi, const PDevice& device) {
  return NewVideoFrameOnDevice(vi, frame_align, (Device*)(void*)device);
}

PVideoFrame ScriptEnvironment::NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device) {
  // todo: high bit-depth: we have too many types now. Do we need really check?
  // Check requested pixel_type:
  switch (vi.pixel_type) {
  case VideoInfo::CS_BGR24:
  case VideoInfo::CS_BGR32:
  case VideoInfo::CS_YUY2:
  case VideoInfo::CS_Y8:
  case VideoInfo::CS_YV12:
  case VideoInfo::CS_YV16:
  case VideoInfo::CS_YV24:
  case VideoInfo::CS_YV411:
  case VideoInfo::CS_I420:
    // AVS16 do not reject when a filter requests it
        // planar YUV 10-32 bit
  case VideoInfo::CS_YUV420P10:
  case VideoInfo::CS_YUV422P10:
  case VideoInfo::CS_YUV444P10:
  case VideoInfo::CS_Y10:
  case VideoInfo::CS_YUV420P12:
  case VideoInfo::CS_YUV422P12:
  case VideoInfo::CS_YUV444P12:
  case VideoInfo::CS_Y12:
  case VideoInfo::CS_YUV420P14:
  case VideoInfo::CS_YUV422P14:
  case VideoInfo::CS_YUV444P14:
  case VideoInfo::CS_Y14:
  case VideoInfo::CS_YUV420P16:
  case VideoInfo::CS_YUV422P16:
  case VideoInfo::CS_YUV444P16:
  case VideoInfo::CS_Y16:
  case VideoInfo::CS_YUV420PS:
  case VideoInfo::CS_YUV422PS:
  case VideoInfo::CS_YUV444PS:
  case VideoInfo::CS_Y32:
    // 16 bit/sample packed RGB
  case VideoInfo::CS_BGR48:
  case VideoInfo::CS_BGR64:
    // planar RGB
  case VideoInfo::CS_RGBP:
  case VideoInfo::CS_RGBP10:
  case VideoInfo::CS_RGBP12:
  case VideoInfo::CS_RGBP14:
  case VideoInfo::CS_RGBP16:
  case VideoInfo::CS_RGBPS:
    // planar RGBA
  case VideoInfo::CS_RGBAP:
  case VideoInfo::CS_RGBAP10:
  case VideoInfo::CS_RGBAP12:
  case VideoInfo::CS_RGBAP14:
  case VideoInfo::CS_RGBAP16:
  case VideoInfo::CS_RGBAPS:
    // planar YUVA 8-32 bit
  case VideoInfo::CS_YUVA420:
  case VideoInfo::CS_YUVA422:
  case VideoInfo::CS_YUVA444:
  case VideoInfo::CS_YUVA420P10:
  case VideoInfo::CS_YUVA422P10:
  case VideoInfo::CS_YUVA444P10:
  case VideoInfo::CS_YUVA420P12:
  case VideoInfo::CS_YUVA422P12:
  case VideoInfo::CS_YUVA444P12:
  case VideoInfo::CS_YUVA420P14:
  case VideoInfo::CS_YUVA422P14:
  case VideoInfo::CS_YUVA444P14:
  case VideoInfo::CS_YUVA420P16:
  case VideoInfo::CS_YUVA422P16:
  case VideoInfo::CS_YUVA444P16:
  case VideoInfo::CS_YUVA420PS:
  case VideoInfo::CS_YUVA422PS:
  case VideoInfo::CS_YUVA444PS:
    break;
  default:
    ThrowError("Filter Error: Filter attempted to create VideoFrame with invalid pixel_type.");
  }

  PVideoFrame retval;

  if (vi.IsPlanar() && (vi.NumComponents() > 1)) {
    if (vi.IsYUV() || vi.IsYUVA()) {
      // Planar requires different math ;)
      const int xmod = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
      const int xmask = xmod - 1;
      if (vi.width & xmask)
        ThrowError("Filter Error: Attempted to request a planar frame that wasn't mod%d in width!", xmod);

      const int ymod = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
      const int ymask = ymod - 1;
      if (vi.height & ymask)
        ThrowError("Filter Error: Attempted to request a planar frame that wasn't mod%d in height!", ymod);

      const int heightUV = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

      retval = NewPlanarVideoFrame(vi.RowSize(PLANAR_Y), vi.height, vi.RowSize(PLANAR_U), heightUV, align, !vi.IsVPlaneFirst(), vi.IsYUVA(), device);
    } else {
      // plane order: G,B,R
      retval = NewPlanarVideoFrame(vi.RowSize(PLANAR_G), vi.height, vi.RowSize(PLANAR_G), vi.height, align, !vi.IsVPlaneFirst(), vi.IsPlanarRGBA(), device);
    }
  }
  else {
    if ((vi.width&1)&&(vi.IsYUY2()))
      ThrowError("Filter Error: Attempted to request an YUY2 frame that wasn't mod2 in width.");

    retval= NewVideoFrameOnDevice(vi.RowSize(), vi.height, align, device);
  }

  return retval;
}

bool ScriptEnvironment::MakeWritable(PVideoFrame* pvf) {
  const PVideoFrame& vf = *pvf;

  // If the frame is already writable, do nothing.
  if (vf->IsWritable())
    return false;

  // Otherwise, allocate a new frame (using NewVideoFrame) and
  // copy the data into it.  Then modify the passed PVideoFrame
  // to point to the new buffer.
  Device* device = vf->GetFrameBuffer()->device;
  PVideoFrame dst;

  if (device->device_type == DEV_TYPE_CUDA) {
    // copy whole frame
    dst = GetOnDeviceFrame(vf, device);
    CopyCUDAFrame(dst, vf, threadEnv.get());
  }
  else {
    const int row_size = vf->GetRowSize();
    const int height = vf->GetHeight();

    bool alpha = 0 != vf->GetPitch(PLANAR_A);
    if (vf->GetPitch(PLANAR_U)) {  // we have no videoinfo, so we assume that it is Planar if it has a U plane.
      const int row_sizeUV = vf->GetRowSize(PLANAR_U); // for Planar RGB this returns row_sizeUV which is the same for all planes
      const int heightUV = vf->GetHeight(PLANAR_U);
      dst = NewPlanarVideoFrame(row_size, height, row_sizeUV, heightUV, frame_align, false, alpha, device);  // Always V first on internal images
    }
    else {
      dst = NewVideoFrameOnDevice(row_size, height, frame_align, device);
    }

    BitBlt(dst->GetWritePtr(), dst->GetPitch(), vf->GetReadPtr(), vf->GetPitch(), row_size, height);
    // Blit More planes (pitch, rowsize and height should be 0, if none is present)
    BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), vf->GetReadPtr(PLANAR_V),
      vf->GetPitch(PLANAR_V), vf->GetRowSize(PLANAR_V), vf->GetHeight(PLANAR_V));
    BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), vf->GetReadPtr(PLANAR_U),
      vf->GetPitch(PLANAR_U), vf->GetRowSize(PLANAR_U), vf->GetHeight(PLANAR_U));
    if (alpha)
      BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), vf->GetReadPtr(PLANAR_A),
        vf->GetPitch(PLANAR_A), vf->GetRowSize(PLANAR_A), vf->GetHeight(PLANAR_A));
  }

  // Copy properties
  dst->avsmap->data = vf->avsmap->data;

  *pvf = dst;
  return true;
}

bool ScriptEnvironment::MakePropertyWritable(PVideoFrame* pvf) {
	const PVideoFrame& vf = *pvf;

	// If the frame is already writable, do nothing.
	if (vf->IsPropertyWritable())
		return false;

	// Otherwise, allocate a new frame (using Subframe)
	PVideoFrame dst;
	if (vf->GetPitch(PLANAR_A)) {
		// planar + alpha
		dst = vf->Subframe(0, vf->GetPitch(), vf->GetRowSize(), vf->GetHeight(), 0, 0, vf->GetPitch(PLANAR_U), 0);
	}
	else if (vf->GetPitch(PLANAR_U)) {
		// planar
		dst = vf->Subframe(0, vf->GetPitch(), vf->GetRowSize(), vf->GetHeight(), 0, 0, vf->GetPitch(PLANAR_U));
	}
	else {
		// single plane
		dst = vf->Subframe(0, vf->GetPitch(), vf->GetRowSize(), vf->GetHeight());
	}

	// Copy properties
	dst->avsmap->data = vf->avsmap->data;

	*pvf = dst;
	return true;
}


void ScriptEnvironment::AtExit(IScriptEnvironment::ShutdownFunc function, void* user_data) {
  at_exit.Add(function, user_data);
}

PVideoFrame ScriptEnvironment::Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) {

  if (src->GetFrameBuffer()->device->device_type == DEV_TYPE_CPU)
    if ((new_pitch | rel_offset) & (frame_align - 1))
      ThrowError("Filter Error: Filter attempted to break alignment of VideoFrame.");

  VideoFrame* subframe;
  subframe = src->Subframe(rel_offset, new_pitch, new_row_size, new_height);
  subframe->avsmap->data = src->avsmap->data;

  size_t vfb_size = src->GetFrameBuffer()->GetDataSize();

  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex); // vector needs locking!
#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT1(0, "ScScriptEnvironment::SubFrame memory mutext lock: %p\n", (void *)&memory_mutex);
#endif
  // automatically inserts if not exists!
  assert(NULL != subframe);
  FrameRegistry2[vfb_size][src->GetFrameBuffer()].push_back(DebugTimestampedFrame(subframe, subframe->avsmap)); // insert with timestamp!

  return subframe;
}

//tsp June 2005 new function compliments the above function
PVideoFrame ScriptEnvironment::SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
                                                        int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) {
  if(src->GetFrameBuffer()->device->device_type == DEV_TYPE_CPU)
    if ((rel_offset | new_pitch | rel_offsetU | rel_offsetV | new_pitchUV) & (frame_align - 1))
      ThrowError("Filter Error: Filter attempted to break alignment of VideoFrame.");

  VideoFrame *subframe = src->Subframe(rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV);
  subframe->avsmap->data = src->avsmap->data;

  size_t vfb_size = src->GetFrameBuffer()->GetDataSize();

  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex); // vector needs locking!
#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT1(0, "ScScriptEnvironment::SubFramePlanar memory mutext lock: %p\n", (void *)&memory_mutex);
#endif
  // automatically inserts if not exists!
  assert(subframe != NULL);
  FrameRegistry2[vfb_size][src->GetFrameBuffer()].push_back(DebugTimestampedFrame(subframe, subframe->avsmap)); // insert with timestamp!

  return subframe;
}

// alpha aware version
PVideoFrame ScriptEnvironment::SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
  int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA) {
  if (src->GetFrameBuffer()->device->device_type == DEV_TYPE_CPU)
    if ((rel_offset | new_pitch | rel_offsetU | rel_offsetV | new_pitchUV | rel_offsetA) & (frame_align - 1))
      ThrowError("Filter Error: Filter attempted to break alignment of VideoFrame.");
  VideoFrame* subframe;
  subframe = src->Subframe(rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV, rel_offsetA);
  subframe->avsmap->data = src->avsmap->data;

  size_t vfb_size = src->GetFrameBuffer()->GetDataSize();

  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex); // vector needs locking!
                                                       // automatically inserts if not exists!
#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT1(0, "ScScriptEnvironment::SubFramePlanar(2) memory mutext lock: %p\n", (void *)&memory_mutex);
#endif
  assert(subframe != NULL);
  FrameRegistry2[vfb_size][src->GetFrameBuffer()].push_back(DebugTimestampedFrame(subframe, subframe->avsmap)); // insert with timestamp!

  return subframe;
}

void* ScriptEnvironment::ManageCache(int key, void* data) {
// An extensible interface for providing system or user access to the
// ScriptEnvironment class without extending the IScriptEnvironment
// definition.

#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT3(0, "ScriptEnvironment::ManageCache %d memory mutex try to lock: %p thread %d\n", key, (void *)&memory_mutex, GetCurrentThreadId());
  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex);
  _RPT2(0, "ScriptEnvironment::ManageCache memory mutex lock ok: %p thread %d\n", (void *)&memory_mutex, GetCurrentThreadId());
#else
  std::lock_guard<std::recursive_mutex> env_lock(memory_mutex);
#endif
  switch ((MANAGE_CACHE_KEYS)key)
  {
    // Called by Cache instances upon creation
  case MC_RegisterCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);
    if (FrontCache != NULL)
      CacheRegistry.push_back(FrontCache);
    FrontCache = cache;
    break;
  }
  // Called by Cache instances upon destruction
  case MC_UnRegisterCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);
    if (FrontCache == cache)
      FrontCache = NULL;
    else
      CacheRegistry.remove(cache);
    break;
  }
  // Called by Cache instances when they want to expand their limit
  case MC_NodAndExpandCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);

    // Nod
    if (cache != FrontCache)
    {
      CacheRegistry.move_to_back(cache);
    }

    // Given that we are within our memory limits,
    // try to expand the limit of those caches that
    // need it.
    // We try to expand most recently used caches first.

    int cache_cap = cache->SetCacheHints(CACHE_GET_CAPACITY, 0);
    int cache_reqcap = cache->SetCacheHints(CACHE_GET_REQUESTED_CAP, 0);
    if (cache_reqcap <= cache_cap)
      return 0;

    Device* device = cache->GetDevice();
    if ((device->memory_used > device->memory_max) || (device->memory_max - device->memory_used < device->memory_max*0.1f))
    {
      // If we don't have enough free reserves, take away a cache slot from
      // a cache instance that hasn't been used since long.

      for (Cache* old_cache : CacheRegistry)
      {
        if (old_cache->GetDevice() != device) {
          continue;
        }
        int osize = cache->SetCacheHints(CACHE_GET_SIZE, 0);
        if (osize != 0)
        {
          old_cache->SetCacheHints(CACHE_SET_MAX_CAPACITY, osize - 1);
          break;
        }
      } // for cit
    }
#ifdef _DEBUG
    _RPT2(0, "ScriptEnvironment::ManageCache increase capacity to %d cache_id=%s\n", cache_cap + 1, cache->FuncName.c_str());
#endif
    cache->SetCacheHints(CACHE_SET_MAX_CAPACITY, cache_cap + 1);

    break;
  }
  // Called by Cache instances when they are accessed
  case MC_NodCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);
    if (cache == FrontCache) {
#ifdef DEBUG_GSCRIPTCLIP_MT
      env_lock.unlock();
#endif
      return 0;
    }

    CacheRegistry.move_to_back(cache);
    break;
  } // case
  case MC_RegisterMTGuard:
  {
    MTGuard* guard = reinterpret_cast<MTGuard*>(data);

    // If we already have a prefetcher, enable MT on the guard
    if (ThreadPoolRegistry.size() > 0)
    {
      guard->EnableMT(nMaxFilterInstances);
    }

    MTGuardRegistry.push_back(guard);

    break;
  }
  case MC_UnRegisterMTGuard:
  {
    MTGuard* guard = reinterpret_cast<MTGuard*>(data);
    for (auto& item : MTGuardRegistry)
    {
      if (item == guard)
      {
        item = NULL;
        break;
      }
    }
    break;
  }
	case MC_RegisterGraphNode:
	{
		FilterGraphNode* node = reinterpret_cast<FilterGraphNode*>(data);
		GraphNodeRegistry.push_back(node);
		break;
	}
	case MC_UnRegisterGraphNode:
	{
		FilterGraphNode* node = reinterpret_cast<FilterGraphNode*>(data);
		for (auto& item : GraphNodeRegistry)
		{
			if (item == node)
			{
				item = NULL;
				break;
			}
		}
		break;
	}
  } // switch
#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT2(0, "ScriptEnvironment::ManageCache memory mutex will unlock on release: %p thread=%d\n", (void *)&memory_mutex, GetCurrentThreadId());
#endif
  return 0;
}


bool ScriptEnvironment::PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key){
  bool oldPlanarChromaAlignmentState = PlanarChromaAlignmentState;

  switch (key)
  {
  case IScriptEnvironment::PlanarChromaAlignmentOff:
  {
    PlanarChromaAlignmentState = false;
    break;
  }
  case IScriptEnvironment::PlanarChromaAlignmentOn:
  {
    PlanarChromaAlignmentState = true;
    break;
  }
  default:
    break;
  }
  return oldPlanarChromaAlignmentState;
}

/* A helper for Invoke.
   Copy a nested array of 'src' into a flat array 'dst'.
   Returns the number of elements that have been written to 'dst'.
   If 'dst' is NULL, will still return the number of elements
   that would have been written to 'dst', but will not actually write to 'dst'.
*/
static size_t Flatten(const AVSValue& src, AVSValue* dst, size_t index, int level, const char* const* arg_names = NULL) {
  // level is starting from zero
  if (src.IsArray()
#ifdef NEW_AVSVALUE
    && level == 0
#endif
    ) { // flatten for the first arg level
    const int array_size = src.ArraySize();
    for (int i=0; i<array_size; ++i) {
      if (!arg_names || arg_names[i] == 0)
        index = Flatten(src[i], dst, index, level+1);
    }
  } else {
    if (dst != NULL)
      dst[index] = src;
    ++index;
  }
  return index;
}

const Function* ScriptEnvironment::Lookup(const char* search_name, const AVSValue* args, size_t num_args,
                    bool &pstrict, size_t args_names_count, const char* const* arg_names, IScriptEnvironment2* ctx)
{
  AVSValue avsv;
  if (ctx->GetVar(search_name, &avsv) && avsv.IsFunction()) {
    auto& funcv = avsv.AsFunction();
    const char* name = funcv->GetLegacyName();
    const Function* func = funcv->GetDefinition();
    if (name != nullptr) {
      // wrapped function
      search_name = name;
    }
    else if (AVSFunction::TypeMatch(func->param_types, args, num_args, false, threadEnv.get()) &&
      AVSFunction::ArgNameMatch(func->param_types, args_names_count, arg_names))
    {
      pstrict = AVSFunction::TypeMatch(func->param_types, args, num_args, true, threadEnv.get());
      return func;
    }
  }

  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);

  const Function *result = NULL;

  size_t oanc;
  do {
    for (int strict = 1; strict >= 0; --strict) {
      pstrict = strict&1;

      // first, look in loaded plugins
      result = plugin_manager->Lookup(search_name, args, num_args, pstrict, args_names_count, arg_names);
      if (result)
        return result;

      // then, look for a built-in function
      for (int i = 0; i < sizeof(builtin_functions)/sizeof(builtin_functions[0]); ++i)
        for (const AVSFunction* j = builtin_functions[i]; !j->empty(); ++j)
        {
          if (streqi(j->name, search_name) &&
            AVSFunction::TypeMatch(j->param_types, args, num_args, pstrict, ctx) &&
            AVSFunction::ArgNameMatch(j->param_types, args_names_count, arg_names))
            return j;
        }
    }
    // Try again without arg name matching
    oanc = args_names_count;
    args_names_count = 0;
  } while (oanc);

  // If we got here it means the function has not been found.
  // If we haven't done so yet, load the plugins in the autoload folders
  // and try again.
  if (!plugin_manager->HasAutoloadExecuted())
  {
    plugin_manager->AutoloadPlugins();
    return Lookup(search_name, args, num_args, pstrict, args_names_count, arg_names, ctx);
  }

  return NULL;
}

bool ScriptEnvironment::CheckArguments(const Function* func, const AVSValue* args, size_t num_args,
  bool &pstrict, size_t args_names_count, const char* const* arg_names)
{
  if (AVSFunction::TypeMatch(func->param_types, args, num_args, false, threadEnv.get()) &&
    AVSFunction::ArgNameMatch(func->param_types, args_names_count, arg_names))
  {
    pstrict = AVSFunction::TypeMatch(func->param_types, args, num_args, true, threadEnv.get());
    return true;
  }
  return false;
}

bool ScriptEnvironment::Invoke_(AVSValue *result, const AVSValue& implicit_last,
  const char* name, const Function *f, const AVSValue& args, const char* const* arg_names,
  InternalEnvironment* env_thread, bool is_runtime)
{

  const int args_names_count = (arg_names && args.IsArray()) ? args.ArraySize() : 0;

  if (name == nullptr) {
    // for debug printing
    name = "<anonymous function>";
  }

  // get how many args we will need to store
  size_t args2_count = Flatten(args, NULL, 0, 0, arg_names);
  if (args2_count > ScriptParser::max_args)
    ThrowError("Too many arguments passed to function (max. is %d)", ScriptParser::max_args);

  // flatten unnamed args
  std::vector<AVSValue> args2(args2_count + 1, AVSValue());
  args2[0] = implicit_last;
  Flatten(args, args2.data() + 1, 0, 0, arg_names);

  bool strict = false;
  int argbase = 1;
  if (f != nullptr) {
    // check arguments
    if (!this->CheckArguments(f, args2.data() + 1, args2_count, strict, args_names_count, arg_names)) {
      if (!implicit_last.Defined() ||
        !this->CheckArguments(f, args2.data(), args2_count + 1, strict, args_names_count, arg_names))
        return false;
      argbase = 0;
      args2_count += 1;
    }
  }
  else {
    // find matching function
    f = this->Lookup(name, args2.data() + 1, args2_count, strict, args_names_count, arg_names, env_thread);
    if (!f) {
      if (!implicit_last.Defined())
        return false;
      f = this->Lookup(name, args2.data(), args2_count + 1, strict, args_names_count, arg_names, env_thread);
      if (!f)
        return false;
      argbase = 0;
      args2_count += 1;
    }
  }

  // combine unnamed args into arrays
  size_t src_index = 0, dst_index = 0;
  const char* p = f->param_types;
  const size_t maxarg3 = max(args2_count, strlen(p)); // well it can't be any longer than this.

  std::vector<AVSValue> args3(maxarg3, AVSValue());

  while (*p) {
    if (*p == '[') {
      p = strchr(p + 1, ']');
      if (!p) break;
      p++;
    }
    else if ((p[1] == '*') || (p[1] == '+')) {
      size_t start = src_index;
      while ((src_index < args2_count) && (AVSFunction::SingleTypeMatch(*p, args2[argbase + src_index], strict)))
        src_index++;
      size_t size = src_index - start;
      assert(args2_count >= size);

      // Even if the AVSValue below is an array of zero size, we can't skip adding it to args3,
      // because filters like BlankClip might still be expecting it.
      args3[dst_index++] = AVSValue(size > 0 ? args2.data() + argbase + start : NULL, (int)size); // can't delete args2 early because of this

      p += 2;
    }
    else {
      if (src_index < args2_count)
        args3[dst_index] = args2[argbase + src_index];
      src_index++;
      dst_index++;
      p++;
    }
  }
  if (src_index < args2_count)
    ThrowError("Too many arguments to function %s", name);

  const int args3_count = (int)dst_index;

  // copy named args
  for (int i = 0; i<args_names_count; ++i) {
    if (arg_names[i]) {
      size_t named_arg_index = 0;
      for (const char* p = f->param_types; *p; ++p) {
        if (*p == '*' || *p == '+') {
          continue;   // without incrementing named_arg_index
        }
        else if (*p == '[') {
          p += 1;
          const char* q = strchr(p, ']');
          if (!q) break;
          if (strlen(arg_names[i]) == size_t(q - p) && !_strnicmp(arg_names[i], p, q - p)) {
            // we have a match
            if (args3[named_arg_index].Defined()) {
              // so named args give can't have .+ specifier
              ThrowError("Script error: the named argument \"%s\" was passed more than once to %s", arg_names[i], name);
            }
#ifndef NEW_AVSVALUE
            //PF 161028 AVS+ arrays as named arguments
            else if (args[i].IsArray()) {
              ThrowError("Script error: can't pass an array as a named argument");
            }
#endif
            else if (args[i].Defined() && !AVSFunction::SingleTypeMatch(q[1], args[i], false)) {
              ThrowError("Script error: the named argument \"%s\" to %s had the wrong type", arg_names[i], name);
            }
            else {
              args3[named_arg_index] = args[i];
              goto success;
            }
          }
          else {
            p = q + 1;
          }
        }
        named_arg_index++;
      }
      // failure
      ThrowError("Script error: %s does not have a named argument \"%s\"", name, arg_names[i]);
    success:;
    }
  }

  // Trim array size to the actual number of arguments
  args3.resize(args3_count);
  std::vector<AVSValue>(args3).swap(args3);

  if(is_runtime) {
    // Invoked by a thread or GetFrame
    AVSValue funcArgs(args3.data(), (int)args3.size());
    *result = f->apply(funcArgs, f->user_data, env_thread);
    return true;
  }

#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT3(0, "ScriptEnvironment::Invoke %s try memory lock %p thread %d\n", name, (void *)&memory_mutex, GetCurrentThreadId());
  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex);
  _RPT3(0, "ScriptEnvironment::Invoke %s memory mutex lock: %p thread %d\n", name, (void *)&memory_mutex, GetCurrentThreadId());
#else
  std::lock_guard<std::recursive_mutex> env_lock(memory_mutex);
#endif

  ScopedCounter suppressThreadCount_(threadEnv->GetSuppressThreadCount());

  // chainedCtor is true if we are being constructed inside/by the
  // constructor of another filter. In that case we want MT protections
  // applied not here, but by the Invoke() call of that filter.
  const bool chainedCtor = invoke_stack.size() > 0;

  MtModeEvaluator mthelper;
#ifdef USE_MT_GUARDEXIT
  std::vector<MTGuardExit*> GuardExits;
#endif

  bool foundClipArgument = false;
  for (int i = argbase; i < (int)args2.size(); ++i)
  {
    auto& argx = args2[i];
#ifndef NEW_AVSVALUE
    assert(!argx.IsArray()); // todo: we can have arrays 161106
#endif
    // todo PF 161112 new arrays: recursive look into arrays whether they contain clips
    if (argx.IsClip())
    {
      foundClipArgument = true;

      const PClip &clip = argx.AsClip();
      IClip *clip_raw = (IClip*)((void*)clip);
      ClipDataStore *data = this->ClipData(clip_raw);

      if (!data->CreatedByInvoke)
      {
#ifdef _DEBUG
        _RPT3(0, "ScriptEnvironment::Invoke.AddChainedFilter %s thread %d this->DefaultMtMode=%d\n", name, GetCurrentThreadId(), (int)this->DefaultMtMode);
#endif
        mthelper.AddChainedFilter(clip, this->DefaultMtMode);
      }

#ifdef USE_MT_GUARDEXIT
      // Wrap this input parameter into a guard exit, which is used when
      // the new clip created later below is MT_SERIALIZED.
      MTGuardExit *ge = new MTGuardExit(argx.AsClip(), name);
      GuardExits.push_back(ge);
      argx = ge;
#endif
    }
  }
  bool isSourceFilter = !foundClipArgument;

  // ... and we're finally ready to make the call
  std::unique_ptr<const FilterConstructor> funcCtor = std::make_unique<const FilterConstructor>(threadEnv.get(), f, &args2, &args3);
  _RPT1(0, "ScriptEnvironment::Invoke after funcCtor make unique %s\r\n", name);

  bool is_mtmode_forced;
  bool filterHasSpecialMT = this->GetFilterMTMode(f, &is_mtmode_forced) == MT_SPECIAL_MT;

  if (filterHasSpecialMT) // e.g. MP_Pipeline
  {
    *result = funcCtor->InstantiateFilter();
#ifdef _DEBUG
    _RPT1(0, "ScriptEnvironment::Invoke done funcCtor->InstantiateFilter %s\r\n", name); // P.F.
#endif
  }
  else if (funcCtor->IsScriptFunction())
  {
    // Eval, EvalOop, Import and user defined script functions
    // Warn user if he set an MT-mode for a script function
    if (this->FilterHasMtMode(f))
    {
      OneTimeLogTicket ticket(LOGTICKET_W1010, f);
      LogMsgOnce(ticket, LOGLEVEL_WARNING, "An MT-mode is set for %s() but it is a script function. You can only set the MT-mode for binary filters, for scripted functions it will be ignored.", f->name);
    }

#ifdef DEBUG_GSCRIPTCLIP_MT
#ifdef _DEBUG
    _RPT1(0, "ScriptEnvironment::Invoke.IsScriptFunction before funcCtor->InstantiateFilter %s\r\n", name); // P.F.
#endif
    /*
    if(funcCtor->IsRuntimeScriptFunction())
      env_lock.unlock();
    */
    /*
      if this is here, we get heap corruption (var table pop context?)
      if this is not here, and the filter is runtime filter, such as YPlaneMax, then inside
      YPlaneMax the child->GetFrame will go into deadlock, as the memory mutex is still held,
      but the GetFrame goes through the cache, and the ManageCache (Nop) call can't get this
      very same memory mutex since our Invoke (see above) still holds it.
      Because there the ManageCache may use another (core?) thread id
      and cannot obtain the lock :(((
      YPlaneMax (MinMax) calls child->GetFrame, that can Invoke srestore_inside1 again but from a different thread.
      */
#endif
    *result = funcCtor->InstantiateFilter();
#ifdef _DEBUG
    _RPT1(0, "ScriptEnvironment::Invoke done funcCtor->InstantiateFilter %s\r\n", name); // P.F.
#endif
  }
  else
  {
#ifdef DEBUG_GSCRIPTCLIP_MT
    _RPT3(0, "ScriptEnvironment::Invoke.WasNotScriptFunction %s memory mutex lock: %p thread %d\n", name, (void *)&memory_mutex, GetCurrentThreadId());
#endif
#ifdef _DEBUG
    Cache *PrevFrontCache = FrontCache;
#endif

    AVSValue fret;

    invoke_stack.push(&mthelper);
    try
    {
      fret = funcCtor->InstantiateFilter();
      invoke_stack.pop();
    }
    catch (...)
    {
      invoke_stack.pop();
      throw;
    }

    // Determine MT-mode, as if this instance had not called Invoke()
    // in its constructor. Note that this is not necessary the final
    // MT-mode.
    // PF 161012 hack(?) don't call if prefetch. If effective mt mode is MT_MULTI, then
    // Prefetch create gets called again
    // Prefetch is activated above in: fret = funcCtor->InstantiateFilter();
    if (fret.IsClip() && (f->name == nullptr || strcmp(f->name, "Prefetch")))
    {
      const PClip &clip = fret.AsClip();

      bool is_mtmode_forced;
      this->GetFilterMTMode(f, &is_mtmode_forced);
      MtMode mtmode = MtModeEvaluator::GetMtMode(clip, f, threadEnv.get());

      if (chainedCtor)
      {
#ifdef DEBUG_GSCRIPTCLIP_MT
        _RPT3(0, "ScriptEnvironment::Invoke.chainedCtor %s memory mutex lock: %p thread %d\n", name, (void *)&memory_mutex, GetCurrentThreadId());
#endif
        // Propagate information about our children's MT-safety
          // to our parent.
        invoke_stack.top()->Accumulate(mthelper);

        // Add our own MT-mode's information to the parent.
        invoke_stack.top()->Accumulate(mtmode);

        *result = fret;
      }
      else
      {
        if (!is_mtmode_forced) {
          mtmode = mthelper.GetFinalMode(mtmode);
        }

        // Special handling for source filters
        if (isSourceFilter
          && MtModeEvaluator::UsesDefaultMtMode(clip, f, threadEnv.get())
          && (MT_SERIALIZED != mtmode))
        {
          mtmode = MT_SERIALIZED;
          OneTimeLogTicket ticket(LOGTICKET_W1001, f);
          LogMsgOnce(ticket, LOGLEVEL_INFO, "%s() does not have any MT-mode specification. Because it is a source filter, it will use MT_SERIALIZED instead of the default MT mode.", f->canon_name);
        }


        *result = MTGuard::Create(mtmode, clip, std::move(funcCtor), threadEnv.get());

#ifdef USE_MT_GUARDEXIT
        // 170531: concept introduced in r2069 is not working
        // Mutex of serialized filters are unlocked and allow to call
        // such filters as MT_NICE_FILTER in a reentrant way
        // Kept for reference, but put in USE_MT_GUARDEXIT define.

        // Activate the guard exists. This allows us to exit the critical
        // section encompassing the filter when execution leaves its routines
        // to call other filters.
        if (MT_SERIALIZED == mtmode)
        {
          for (auto &ge : GuardExits)
          {
            _RPT3(0, "ScriptEnvironment::Invoke.ActivateGuard %s thread %d\n", name, GetCurrentThreadId());
            ge->Activate(guard);
          }
        }
#endif

        IClip *clip_raw = (IClip*)((void*)clip);
        ClipDataStore *data = this->ClipData(clip_raw);
        data->CreatedByInvoke = true;
      } // if (chainedCtor)

      // Nekopanda: moved here from above.
      // some filters invoke complex filters in its constructor, and they need cache.
      *result = CacheGuard::Create(*result, NULL, threadEnv.get());

      // Check that the filter returns zero for unknown queries in SetCacheHints().
      // This is actually something we rely upon.
      if ((clip->GetVersion() >= 5) && (0 != clip->SetCacheHints(CACHE_USER_CONSTANTS, 0)))
      {
        OneTimeLogTicket ticket(LOGTICKET_W1002, f);
        LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() violates semantic contracts and may cause undefined behavior. Please inform the author of the plugin.", f->canon_name);
      }

      // Warn user if the MT-mode of this filter is unknown
      if (MtModeEvaluator::UsesDefaultMtMode(clip, f, threadEnv.get()) && !isSourceFilter)
      {
        OneTimeLogTicket ticket(LOGTICKET_W1004, f);
        LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() has no MT-mode set and will use the default MT-mode. This might be dangerous.", f->canon_name);
      }

      // Warn user if he forced an MT-mode that differs from the one specified by the filter itself
      if (is_mtmode_forced
        && MtModeEvaluator::ClipSpecifiesMtMode(clip)
        && MtModeEvaluator::GetInstanceMode(clip) != mtmode)
      {
        OneTimeLogTicket ticket(LOGTICKET_W1005, f);
        LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() specifies an MT-mode for itself, but a script forced a different one. Either the plugin or the script is erronous.", f->canon_name);
      }

      // Inform user if a script unnecessarily specifies an MT-mode for this filter
      if (!is_mtmode_forced
        && this->FilterHasMtMode(f)
        && MtModeEvaluator::ClipSpecifiesMtMode(clip))
      {
        OneTimeLogTicket ticket(LOGTICKET_W1006, f);
        LogMsgOnce(ticket, LOGLEVEL_INFO, "Ignoring unnecessary MT-mode specification for %s() by script.", f->canon_name);
      }

    } // if (fret.IsClip())
    else
    {
      *result = fret;
    }

    // static device check
    // this is not enough to check all dependencies but much helpful to users
    if ((*result).IsClip()) {
      auto last = (argbase == 0) ? implicit_last : AVSValue();
      CheckChildDeviceTypes((*result).AsClip(), name, last, args, arg_names, threadEnv.get());
    }

    // filter graph
    if (graphAnalysisEnable && (*result).IsClip()) {
      auto last = (argbase == 0) ? implicit_last : AVSValue();
      *result = new FilterGraphNode((*result).AsClip(), f->name, last, args, arg_names, threadEnv.get());
    }

    // args2 and args3 are not valid after this point anymore
#ifdef _DEBUG
    if (PrevFrontCache != FrontCache && FrontCache != NULL) // cache registering swaps frontcache to the current
    {
      _RPT2(0, "ScriptEnvironment::Invoke done Cache::Create %s  cache_id=%p\r\n", name, (void *)FrontCache); // P.F.
      FrontCache->FuncName = name; // helps debugging. See also in cache.cpp
    }
    else {
      _RPT1(0, "ScriptEnvironment::Invoke done Cache::Create %s\r\n", name); // P.F.
    }
#endif
  }
#ifdef DEBUG_GSCRIPTCLIP_MT
  _RPT3(0, "ScriptEnvironment::Invoke %s memory mutex unlock soon: %p thread %d\n", name, (void *)&memory_mutex, GetCurrentThreadId());
#endif

  return true;
}


bool ScriptEnvironment::FunctionExists(const char* name)
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);

  // Look among variable table
  AVSValue result;
  if (threadEnv->GetVar(name, &result)) {
    if (result.IsFunction()) {
      return true;
    }
  }

  // Look among internal functions
  if (InternalFunctionExists(name))
    return true;

  // Look among plugin functions
  if (plugin_manager->FunctionExists(name))
    return true;

  // Uhh... maybe if we load the plugins we'll have the function
  if (!plugin_manager->HasAutoloadExecuted())
  {
    plugin_manager->AutoloadPlugins();
    return this->FunctionExists(name);
  }

  return false;
}

bool ScriptEnvironment::InternalFunctionExists(const char* name)
{
  for (int i = 0; i < sizeof(builtin_functions)/sizeof(builtin_functions[0]); ++i)
    for (const AVSFunction* j = builtin_functions[i]; !j->empty(); ++j)
      if (streqi(j->name, name))
        return true;

  return false;
}

void ScriptEnvironment::BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height) {
  if (height<0)
    ThrowError("Filter Error: Attempting to blit an image with negative height.");
  if (row_size<0)
    ThrowError("Filter Error: Attempting to blit an image with negative row size.");
  ::BitBlt(dstp, dst_pitch, srcp, src_pitch, row_size, height);
}

void ScriptEnvironment::ThrowError(const char* fmt, ...)
{
  va_list val;
  va_start(val, fmt);
  threadEnv->VThrowError(fmt, val);
  va_end(val);
}

void ScriptEnvironment::VThrowError(const char* fmt, va_list va)
{
  threadEnv->VThrowError(fmt, va);
}

PVideoFrame ScriptEnvironment::SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA)
{
  return SubframePlanar(src, rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV, rel_offsetA);
}

PDevice ScriptEnvironment::GetDevice(AvsDeviceType device_type, int device_index) const
{
  return Devices->GetDevice(device_type, device_index);
}

int ScriptEnvironment::SetMemoryMax(AvsDeviceType type, int index, int mem)
{
  return Devices->GetDevice(type, index)->SetMemoryMax(mem);
}

PVideoFrame ScriptEnvironment::GetOnDeviceFrame(const PVideoFrame& src, Device* device)
{
#ifdef SIZETMOT
  typedef size_t offset_t;
  typedef ptrdiff_t diff_t;
#else
  typedef int offset_t;
  typedef int diff_t;
#endif

  size_t srchead = GetFrameHead(src);

  // make space for alignment
  size_t size = GetFrameTail(src) - srchead;

  VideoFrame *res = GetNewFrame(size, frame_align - 1, device);

  const diff_t offset = (diff_t)(AlignPointer(res->vfb->GetWritePtr(), frame_align) - res->vfb->GetWritePtr()); // first line offset for proper alignment
  const diff_t diff = offset - (diff_t)srchead;

  res->offset = src->offset + diff;
  res->pitch = src->pitch;
  res->row_size = src->row_size;
  res->height = src->height;
  res->offsetU = src->pitchUV ? (src->offsetU + diff) : res->offset;
  res->offsetV = src->pitchUV ? (src->offsetV + diff) : res->offset;
  res->pitchUV = src->pitchUV;
  res->row_sizeUV = src->row_sizeUV;
  res->heightUV = src->heightUV;
  res->offsetA = src->pitchA ? (src->offsetA + diff) : 0;
  res->pitchA = src->pitchA;
  res->row_sizeA = src->row_sizeA;
  res->avsmap->data = src->avsmap->data;

  return PVideoFrame(res);
}

void ScriptEnvironment::CopyFrameProps(PVideoFrame src, PVideoFrame dst) const
{
  dst->avsmap->data = src->avsmap->data;
}

ThreadPool* ScriptEnvironment::NewThreadPool(size_t nThreads)
{
  ThreadPool* pool = new ThreadPool(nThreads, nTotalThreads, threadEnv.get());
  ThreadPoolRegistry.emplace_back(pool);

  nTotalThreads += nThreads;

  if (nMaxFilterInstances < nThreads + 1) {
    // make 2^n
    nMaxFilterInstances = 1;
    while (nThreads + 1 > (nMaxFilterInstances <<= 1));
  }

  // Since this method basically enables MT operation,
  // upgrade all MTGuards to MT-mode.
  for (MTGuard* guard : MTGuardRegistry)
  {
    if (guard != NULL)
      guard->EnableMT(nMaxFilterInstances);
  }

  return pool;
}

void ScriptEnvironment::SetDeviceOpt(DeviceOpt opt, int val)
{
  Devices->SetDeviceOpt(opt, val, threadEnv.get());
}

void ScriptEnvironment::UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar)
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->UpdateFunctionExports(funcName, funcParams, exportVar);
}

extern void ApplyMessage(PVideoFrame* frame, const VideoInfo& vi,
  const char* message, int size, int textcolor, int halocolor, int bgcolor,
  IScriptEnvironment* env);


const AVS_Linkage* ScriptEnvironment::GetAVSLinkage() {
  extern const AVS_Linkage* const AVS_linkage; // In interface.cpp

  return AVS_linkage;
}


void ScriptEnvironment::ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor)
{
  if ((*frame)->GetDevice()->device_type == DEV_TYPE_CUDA) {
    // if frame is CUDA frame, copy to CPU and apply
    PVideoFrame copy = GetOnDeviceFrame(*frame, Devices->GetCPUDevice());
    CopyCUDAFrame(copy, *frame, threadEnv.get(), true);
    ::ApplyMessage(&copy, vi, message, size, textcolor, halocolor, bgcolor, threadEnv.get());
    CopyCUDAFrame(*frame, copy, threadEnv.get(), true);
  }
  else {
    ::ApplyMessage(frame, vi, message, size, textcolor, halocolor, bgcolor, threadEnv.get());
  }
}


void ScriptEnvironment::DeleteScriptEnvironment() {
  // Provide a method to delete this ScriptEnvironment in
  // the same malloc context in which it was created below.
  delete this;
}


AVSC_API(IScriptEnvironment*, CreateScriptEnvironment)(int version) {
  return CreateScriptEnvironment2(version);
}

AVSC_API(IScriptEnvironment2*, CreateScriptEnvironment2)(int version)
{
  /* Some plugins use OpenMP. But OMP threads do not exit immediately
  * after all work is exhausted, and keep spinning for a small amount
  * of time waiting for new jobs. If we unload the OMP DLL (indirectly
  * by unloading its plugin that started it) while its threads are
  * running, the sky comes crashing down. This results in crashes
  * from OMP plugins if the IScriptEnvironment is destructed shortly
  * after a GetFrame() call.
  *
  * OMP_WAIT_POLICY=passive changes the behavior of OMP thread pools
  * to shut down immediately instead of continuing to spin.
  * This solves our problem at the cost of some performance.
  */
  _putenv("OMP_WAIT_POLICY=passive");

  if (version <= AVISYNTH_INTERFACE_VERSION)
    return (new ScriptEnvironment())->GetMainThreadEnv();
  else
    return NULL;
}