#pragma once

#include <Vortex2D/Vortex2D.h>

extern Vortex2D::Renderer::Device* gDevice;

class Scan
{
public:
  Scan(const Vortex2D::Renderer::Device& device,
       int size,
       int localSize);

  std::vector<float> operator()(const std::vector<float>& input);

private:
  Vortex2D::Renderer::CommandBuffer mScanCmd;
  Vortex2D::Renderer::Work mScanWork;
  Vortex2D::Renderer::Work mAddWork;

  std::vector<Vortex2D::Renderer::Buffer<float>> mBuffers; // buffers for intermediate results
  Vortex2D::Renderer::Buffer<float> mLocalInput, mLocalOutput; // buffers for copying input/output to device
  std::vector<Vortex2D::Renderer::Work::Bound> mScan; // bound scan shaders for each level
  std::vector<Vortex2D::Renderer::Work::Bound> mAdd; // bound add shaders for each level
};

void CheckScan();
