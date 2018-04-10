#ifndef _AVS_FILTER_GRAPH_H
#define _AVS_FILTER_GRAPH_H

#include "internal.h"
#include <vector>
#include <memory>

class FilterGraph;

class FilterGraphNode : public IClip
{
  PClip child;

  std::string name;
  AVSValue args;
  std::vector<std::unique_ptr<AVSValue[]>> arrays;
  std::vector<std::string> argnames;

  friend FilterGraph;
public:
  FilterGraphNode(PClip child, const char* name, const AVSValue& last, const AVSValue& args, const char* const* arg_names);

  virtual int __stdcall GetVersion() { return child->GetVersion(); }
  virtual PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  { return child->GetFrame(n, env); }
  virtual bool __stdcall GetParity(int n) { return child->GetParity(n); }
  virtual void __stdcall GetAudio(void* buf, __int64 start, __int64 count, IScriptEnvironment* env)
  { return child->GetAudio(buf, start, count, env); }
  virtual int __stdcall SetCacheHints(int cachehints, int frame_range)
  { return child->SetCacheHints(cachehints, frame_range); }
  virtual const VideoInfo& __stdcall GetVideoInfo() { return child->GetVideoInfo(); }
};

#endif  // _AVS_FILTER_GRAPH_H
