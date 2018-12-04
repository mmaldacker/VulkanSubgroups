#pragma once

#include <Vortex2D/Vortex2D.h>

extern Vortex2D::Renderer::Device* gDevice;

class Scan
{
public:
  Scan(const Vortex2D::Renderer::Device& device,
       int size,
       int localSize);

  void Upload(const std::vector<float>& input);
  std::vector<float> Download();
  void Submit();
  uint64_t GetElapsedNs();

private:
  Vortex2D::Renderer::Timer mTimer;
  Vortex2D::Renderer::CommandBuffer mUploadCmd, mDownloadCmd, mScanCmd;
  Vortex2D::Renderer::Work mScanWork;
  Vortex2D::Renderer::Work mAddWork;

  std::vector<Vortex2D::Renderer::Buffer<float>> mBuffers; // buffers for intermediate results
  Vortex2D::Renderer::Buffer<float> mLocalInput, mLocalOutput; // buffers for copying input/output to device
  std::vector<Vortex2D::Renderer::Work::Bound> mScan; // bound scan shaders for each level
  std::vector<Vortex2D::Renderer::Work::Bound> mAdd; // bound add shaders for each level
};

void CheckScan();
