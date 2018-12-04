#pragma once

#include <Vortex2D/Vortex2D.h>

extern Vortex2D::Renderer::Device* gDevice;

class Reduce
{
public:
  Reduce(const Vortex2D::Renderer::Device& device,
         int size,
         int localSize);

  void Upload(const std::vector<float>& input);
  float Download();
  void Submit();
  uint64_t GetElapsedNs();

private:
  Vortex2D::Renderer::Timer mTimer;
  Vortex2D::Renderer::CommandBuffer mUploadCmd, mDownloadCmd, mReduceCmd;
  Vortex2D::Renderer::Work mReduceWork; // reduce shader
  std::vector<Vortex2D::Renderer::Buffer<float>> mBuffers; // buffers for input, intermediate results and output
  Vortex2D::Renderer::Buffer<float> mLocalInput, mLocalOutput; // buffers for copying input/output to device
  std::vector<Vortex2D::Renderer::Work::Bound> mReduce; // bound reduce shaders for each level
};

void CheckReduce();
