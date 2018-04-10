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

FilterGraphNode::FilterGraphNode(PClip child, const char* name,
  const AVSValue& last_, const AVSValue& args_, const char* const* argnames_)
  : child(child)
  , name(name)
{
  if (last_.Defined()) {
    std::vector<AVSValue> argstmp;
    argstmp.push_back(last_);
    if (argnames_) {
      argnames.push_back(std::string());
    }
    for (int i = 0; i < args_.ArraySize(); ++i) {
      argstmp.push_back(args_[i]);
    }
    args = DeepCopyValue(arrays, AVSValue(argstmp.data(), argstmp.size()));
  }
  else {
    args = DeepCopyValue(arrays, args_.IsArray() ? args_ : AVSValue(args_, 1));
  }

  if (argnames_) {
    for (int i = 0; i < args_.ArraySize(); ++i) {
      argnames.push_back(argnames_[i] ? std::string(argnames_[i]) : std::string());
    }
  }
}

class FilterGraph
{
protected:
  IScriptEnvironment * env;

  struct NodeInfo {
    bool isFunction;
    int number;
    std::string name;
    std::string args;
    std::vector<void*> refNodes;

		int cacheSize;
		int cacheCapacity;

    NodeInfo() { }
    NodeInfo(int number) : number(number) { }
  };

  std::map<void*, NodeInfo> nodeMap;

  int DoClip(IClip* pclip) {
    if (nodeMap.find(pclip) == nodeMap.end()) {
      nodeMap.insert(std::make_pair(pclip, (int)nodeMap.size()));
      FilterGraphNode* node = dynamic_cast<FilterGraphNode*>(pclip);
      if (node != nullptr) {
        NodeInfo& info = nodeMap[node];
        info.isFunction = false;
        info.name = node->name;
        info.args = "(" + DoArray(info, nullptr, node->argnames.data(), node->args) + ")";
				info.cacheSize = node->SetCacheHints(CACHE_GET_SIZE, 0);
				info.cacheCapacity = node->SetCacheHints(CACHE_GET_CAPACITY, 0);
      }
      OutClip(nodeMap[node]);
    }
    return nodeMap[pclip].number;
  }

  int DoFunc(IFunction* pfunc) {
    if (nodeMap.find(pfunc) == nodeMap.end()) {
      nodeMap.insert(std::make_pair(pfunc, (int)nodeMap.size()));
      NodeInfo& info = nodeMap[pfunc];
      info.isFunction = true;
      auto captures = pfunc->GetCaptures();
      info.name = pfunc->ToString(env);
      info.args = "[" + DoArray(info, captures.var_names, nullptr, AVSValue(captures.var_data, captures.count)) + "]";
      info.cacheSize = 0;
      info.cacheCapacity = 0;
      OutFunc(info);
    }
    return nodeMap[pfunc].number;
  }

  std::string DoArray(NodeInfo& info, const char** argnames_c, std::string* argnames_s, const AVSValue& arr) {
    std::stringstream ss;
    int breakpos = 0;
    int maxlen = 60;

    for (int i = 0; i < arr.ArraySize(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      if (argnames_c && argnames_c[i]) {
        ss << argnames_c[i] << "=";
      }
      if (argnames_s && argnames_s[i].size() > 0) {
        ss << argnames_s[i] << "=";
      }
      const AVSValue& v = arr[i];
      if (!v.Defined()) {
        ss << "default";
      }
      else if (v.IsClip()) {
        IClip* pclip = (IClip*)(void*)v.AsClip();
        int clipnum = DoClip(pclip);
        ss << "clip" << (clipnum + 1);
        info.refNodes.push_back(pclip);
      }
      else if (v.IsFunction()) {
        IFunction* pfunc = (IFunction*)(void*)v.AsFunction();
        int funcnum = DoFunc(pfunc);
        ss << "func" << (funcnum + 1);
        info.refNodes.push_back(pfunc);
      }
      else if (v.IsArray()) {
        ss << OutArray("(" + DoArray(info, nullptr, nullptr, v) + ")");
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
      if ((int)ss.tellp() - breakpos > maxlen) {
        ss << "\n";
        breakpos = (int)ss.tellp();
      }
    }
    return ss.str();
  }

  virtual void OutClip(const NodeInfo& info) = 0;
  virtual void OutFunc(const NodeInfo& info) = 0;
  virtual std::string OutArray(const std::string& args) = 0;

public:
  int Construct(FilterGraphNode* root, IScriptEnvironment* env_)
  {
    env = env_;
    nodeMap.clear();
    return DoClip(root);
  }
};

static void ReplaceAll(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

class AvsScriptFilterGraph : private FilterGraph
{
  std::stringstream ss;

protected:
  virtual void OutClip(const NodeInfo& info) {
    int num = info.number + 1;
    if (info.name.size() == 0) {
      ss << "clip" << num << ": Failed to get information" << std::endl;
    }
    else {
      auto args = info.args;
      ReplaceAll(args, "\n", "");
      ss << "clip" << num << " = " << info.name << args << std::endl;
    }
  }
  virtual void OutFunc(const NodeInfo& info) {
    int num = info.number + 1;
    auto args = info.args;
    ReplaceAll(args, "\n", "");
    ss << "func" << num << " = function" << args << "(){ " << info.name << " }" << std::endl;
  }

  int nextArrayNumber = 0;
  virtual std::string OutArray(const std::string& args) {
    std::string name = std::string("array") + std::to_string(++nextArrayNumber);
    ss << name;
    ss << " = ArrayCreate" << args << std::endl;
    return name;
  }
public:

  void Construct(FilterGraphNode* root, IScriptEnvironment* env) {
    int last = FilterGraph::Construct(root, env);
    ss << "return clip" << (last + 1) << std::endl;
  }

  std::string GetOutput() {
    return ss.str();
  }
};

class DotFilterGraph : private FilterGraph
{
  bool enableArgs;
  std::stringstream ss;

protected:
  virtual void OutClip(const NodeInfo& info) {
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
        ReplaceAll(label, "\n", "\\n");
        ss << " [label = \"" << label << "\"];" << std::endl;
      }
      else {
				if (info.cacheCapacity != 0) {
					ss << " [label = \"" << info.name << "(" << info.cacheSize << " of " << info.cacheCapacity << ")" << "\"];" << std::endl;
				}
				else {
					ss << " [label = \"" << info.name << "\"];" << std::endl;
				}
      }
    }
    for (void* pclip : info.refNodes) {
      auto& node = nodeMap[pclip];
      int refnum = node.number + 1;
      if (node.isFunction) {
        ss << "func" << refnum << " -> " << "clip" << num << ";" << std::endl;
      }
      else {
        ss << "clip" << refnum << " -> " << "clip" << num << ";" << std::endl;
      }
    }
  }
  virtual void OutFunc(const NodeInfo& info) {
    int num = info.number + 1;
    ss << "func" << num;
    if (enableArgs) {
      std::string label = info.name + "\n" + info.args;
      ReplaceAll(label, "\\", "\\\\");
      ReplaceAll(label, "\"", "\\\"");
      ReplaceAll(label, "\n", "\\n");
      ss << " [label = \"" << label << "\"];" << std::endl;
    }
    else {
      ss << " [label = \"" << info.name << "\"];" << std::endl;
    }
    for (void* pclip : info.refNodes) {
      auto& node = nodeMap[pclip];
      int refnum = node.number + 1;
      if (node.isFunction) {
        ss << "func" << refnum << " -> " << "func" << num << ";" << std::endl;
      }
      else {
        ss << "clip" << refnum << " -> " << "func" << num << ";" << std::endl;
      }
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

  void Construct(FilterGraphNode* root, bool enableArgs, IScriptEnvironment* env) {
    this->enableArgs = enableArgs;
    ss << "digraph avs_filter_graph {" << std::endl;
    ss << "node [ shape = box ];" << std::endl;
    int last = FilterGraph::Construct(root, env);
    ss << "GOAL;" << std::endl;
    ss << "clip" << (last + 1) << " -> GOAL" << std::endl;
    ss << "}" << std::endl;
  }

  std::string GetOutput() {
    return ss.str();
  }
};

static void DoDumpGraph(PClip clip, int mode, const char* path, IScriptEnvironment* env)
{
	FilterGraphNode* root = dynamic_cast<FilterGraphNode*>((IClip*)(void*)clip);

	std::string ret;

	if (mode == 0) {
		AvsScriptFilterGraph graph;
		graph.Construct(root, env);
		ret = graph.GetOutput();
	}
	else if (mode == 1 || mode == 2) {
		DotFilterGraph graph;
		graph.Construct(root, mode == 1, env);
		ret = graph.GetOutput();
	}
	else {
		env->ThrowError("Unknown mode (%d)", mode);
	}

	FILE* fp = fopen(path, "w");
	if (fp == nullptr) {
		env->ThrowError("Could not open output file ...");
	}
	fwrite(ret.data(), ret.size(), 1, fp);
	fclose(fp);
}

class DelayedDump : public GenericVideoFilter
{
	std::string outpath;
	int mode;
	int nframes;
	bool fired;
public:
	DelayedDump(PClip clip, const std::string& outpath, int mode, int nframes)
		: GenericVideoFilter(clip)
		, outpath(outpath)
		, mode(mode)
		, nframes(nframes)
		, fired(false)
	{ }

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		if (n == nframes && fired == false) {
			fired = true;
			DoDumpGraph(child, mode, outpath.c_str(), env);
		}
		return child->GetFrame(n, env);
	}
};

static AVSValue DumpFilterGraph(AVSValue args, void* user_data, IScriptEnvironment* env) {
  PClip clip = args[0].AsClip();
  FilterGraphNode* root = dynamic_cast<FilterGraphNode*>((IClip*)(void*)clip);
  if (root == nullptr) {
    env->ThrowError("clip is not a FilterChainNode. Ensure that you enabled the chain analysis by SetChainAnalysis(true).");
  }

	int mode = args[2].AsInt(0);
	const char* path = args[1].AsString("");
	int nframes = args[3].AsInt(-1);

	if (nframes >= 0) {
		return new DelayedDump(clip, path, mode, nframes);
	}

	DoDumpGraph(clip, mode, path, env);

  return clip;
}

static AVSValue __cdecl SetGraphAnalysis(AVSValue args, void* user_data, IScriptEnvironment* env) {
  static_cast<InternalEnvironment*>(env)->SetGraphAnalysis(args[0].AsBool());
  return AVSValue();
}

extern const AVSFunction FilterGraph_filters[] = {
  { "SetGraphAnalysis", BUILTIN_FUNC_PREFIX, "b", SetGraphAnalysis, nullptr },
  { "DumpFilterGraph", BUILTIN_FUNC_PREFIX, "c[outfile]s[mode]i[nframes]i", DumpFilterGraph, nullptr },
  { 0 }
};
