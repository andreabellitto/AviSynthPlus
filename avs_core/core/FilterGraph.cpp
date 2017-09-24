#include "FilterGraph.h"
#include "InternalEnvironment.h"

#include <map>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

static AVSValue DeepCopyValue(std::vector<std::unique_ptr<AVSValue[]>>& arrays, const AVSValue& src) {
  if (src.IsArray()) {
    AVSValue* copy = new AVSValue[src.ArraySize()];
    for (int i = 0; i < src.ArraySize(); ++i) {
      copy[i] = DeepCopyValue(arrays, src[i]);
    }
    arrays.emplace_back(std::unique_ptr<AVSValue[]>(copy));
    return AVSValue(copy, src.ArraySize());
  }
  return src;
}

FilterGraphNode::FilterGraphNode(PClip child, const char* name, const AVSValue& args_, const char* const* argnames_)
  : child(child)
  , name(name)
{
  args = DeepCopyValue(arrays, args_.IsArray() ? args_ : AVSValue(args_, 1));

  argnames.resize(args.ArraySize());
  if (argnames_) {
    for (int i = 0; i < args.ArraySize(); ++i) {
      if (argnames_[i]) {
        argnames[i] = argnames_[i];
      }
    }
  }
}

class FilterGraph
{
protected:
  struct ClipInfo {
    int number;
    std::string name;
    std::string args;
    std::vector<void*> refNodes;

    ClipInfo() { }
    ClipInfo(int number) : number(number) { }
  };

  std::map<void*, ClipInfo> clipMap;

  int DoClip(IClip* pclip) {
    if (clipMap.find(pclip) == clipMap.end()) {
      clipMap.insert(std::make_pair(pclip, (int)clipMap.size()));
      FilterGraphNode* node = dynamic_cast<FilterGraphNode*>(pclip);
      if (node != nullptr) {
        ClipInfo& info = clipMap[node];
        info.name = node->name;
        info.args = DoArray(node, info, node->argnames.data(), node->args);
      }
      OutClip(clipMap[node]);
    }
    return clipMap[pclip].number;
  }

  std::string DoArray(FilterGraphNode* node, ClipInfo& info, std::string* argnames, const AVSValue& arr) {
    std::stringstream ss;
    ss << "(";
    for (int i = 0; i < arr.ArraySize(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      if (argnames && argnames[i].size() > 0) {
        ss << argnames[i] << "=";
      }
      const AVSValue& v = arr[i];
      if (v.IsClip()) {
        IClip* pclip = (IClip*)(void*)v.AsClip();
        int clipnum = DoClip(pclip);
        ss << "clip" << (clipnum + 1);
        info.refNodes.push_back(pclip);
      }
      else if (v.IsArray()) {
        ss << OutArray(DoArray(node, info, nullptr, v));
      }
      else if (v.IsBool()) {
        ss << (v.AsBool() ? "True" : "False");
      }
      else if (v.IsInt()) {
        ss << v.AsInt();
      }
      else if (v.IsFloat()) {
        ss << std::setprecision(8) << v.AsFloat();
      }
      else if (v.IsString()) {
        ss << "\"" << v.AsString() << "\"";
      }
      else {
        ss << "<error>";
      }
    }
    ss << ")";
    return ss.str();
  }

  virtual void OutClip(const ClipInfo& info) = 0;
  virtual std::string OutArray(const std::string& args) = 0;

public:
  int Construct(FilterGraphNode* root) {
    clipMap.clear();
    return DoClip(root);
  }
};

class AvsScriptFilterGraph : private FilterGraph
{
  std::stringstream ss;

protected:
  virtual void OutClip(const ClipInfo& info) {
    int num = info.number + 1;
    if (info.name.size() == 0) {
      ss << "clip" << num << ": Failed to get information" << std::endl;
    }
    else {
      ss << "clip" << num << " = " << info.name << info.args << std::endl;
    }
  }

  int nextArrayNumber = 0;
  virtual std::string OutArray(const std::string& args) {
    std::string name = std::string("array") + std::to_string(++nextArrayNumber);
    ss << name;
    ss << " = ArrayCreate" << args << std::endl;
    return name;
  }
public:

  void Construct(FilterGraphNode* root) {
    int last = FilterGraph::Construct(root);
    ss << "return clip" << (last + 1) << std::endl;
  }

  std::string GetOutput() {
    return ss.str();
  }
};

static void ReplaceAll(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

class DotFilterGraph : private FilterGraph
{
  bool enableArgs;
  std::stringstream ss;

protected:
  virtual void OutClip(const ClipInfo& info) {
    int num = info.number + 1;
    ss << "clip" << num;
    if (info.name.size() == 0) {
      ss << " [label = \"...\"];" << std::endl;
    }
    else {
      if (enableArgs) {
        std::string label = info.name + info.args;
        ReplaceAll(label, "\\", "\\\\");
        ReplaceAll(label, "\"", "\\\"");
        ss << " [label = \"" << label << "\"];" << std::endl;
      }
      else {
        ss << " [label = \"" << info.name << "\"];" << std::endl;
      }
    }
    for (void* pclip : info.refNodes) {
      int refnum = clipMap[pclip].number + 1;
      ss << "clip" << refnum << " -> " << "clip" << num << ";" << std::endl;
    }
  }

  int nextArrayNumber = 0;
  virtual std::string OutArray(const std::string& args) {
    std::string name = std::string("array") + std::to_string(++nextArrayNumber);
    //ss << name;
    //ss << " = ArrayCreate" << args << std::endl;
    return name;
  }
public:

  void Construct(FilterGraphNode* root, bool enableArgs) {
    this->enableArgs = enableArgs;
    ss << "digraph avs_filter_graph {" << std::endl;
    ss << "node [ shape = box ];" << std::endl;
    int last = FilterGraph::Construct(root);
    ss << "GOAL;" << std::endl;
    ss << "clip" << (last + 1) << " -> GOAL" << std::endl;
    ss << "}" << std::endl;
  }

  std::string GetOutput() {
    return ss.str();
  }
};

static AVSValue DumpFilterGraph(AVSValue args, void* user_data, IScriptEnvironment* env) {
  PClip clip = args[0].AsClip();
  FilterGraphNode* root = dynamic_cast<FilterGraphNode*>((IClip*)(void*)clip);
  if (root == nullptr) {
    env->ThrowError("clip is not a FilterChainNode. Ensure that you enabled the chain analysis by SetChainAnalysis(true).");
  }

  std::string ret;

  int mode = args[2].AsInt(0);

  if (mode == 0) {
    AvsScriptFilterGraph graph;
    graph.Construct(root);
    ret = graph.GetOutput();
  }
  else if (mode == 1 || mode == 2) {
    DotFilterGraph graph;
    graph.Construct(root, mode == 1);
    ret = graph.GetOutput();
  }
  else {
    env->ThrowError("Unknown mode (%d)", mode);
  }

  const char* path = args[1].AsString("");
  FILE* fp = fopen(path, "w");
  if (fp == nullptr) {
    env->ThrowError("Could not open output file ...");
  }
  fwrite(ret.data(), ret.size(), 1, fp);
  fclose(fp);

  return clip;
}

static AVSValue __cdecl SetGraphAnalysis(AVSValue args, void* user_data, IScriptEnvironment* env) {
  static_cast<InternalEnvironment*>(env)->SetGraphAnalysis(args[0].AsBool());
  return AVSValue();
}

extern const AVSFunction FilterGraph_filters[] = {
  { "SetGraphAnalysis", BUILTIN_FUNC_PREFIX, "b", SetGraphAnalysis, nullptr },
  { "DumpFilterGraph", BUILTIN_FUNC_PREFIX, "c[outfile]s[mode]i", DumpFilterGraph, nullptr },
  { 0 }
};
