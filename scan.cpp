#include "scan.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <numeric>
#include <execution>
#include "vulkansubgroups_spirv.h"

namespace
{
int NextSize(int size, int localSize)
{
  return (size + localSize) / localSize;
}
}

Scan::Scan(const Vortex2D::Renderer::Device& device,
           int size,
           int localSize)
  : mScanCmd(device)
  , mScanWork(device, {size, localSize}, Vortex2D::SPIRV::Scan_comp)
  , mAddWork(device, {NextSize(size, localSize), localSize}, Vortex2D::SPIRV::Add_comp)
  , mLocalInput(device, size, VMA_MEMORY_USAGE_CPU_TO_GPU)
  , mLocalOutput(device, size, VMA_MEMORY_USAGE_GPU_TO_CPU)
{
  int n = size;
  while (n > 1)
  {
    mBuffers.emplace_back(device, n, VMA_MEMORY_USAGE_GPU_ONLY);
    n = NextSize(n, localSize);
  }

  mBuffers.emplace_back(device, 1, VMA_MEMORY_USAGE_GPU_ONLY);

  n = size;
  for (std::size_t i = 0; i < mBuffers.size() - 1; i++)
  {
    mScan.emplace_back(mScanWork.Bind({n, localSize}, {mBuffers[i], mBuffers[i], mBuffers[i + 1]}));
    mAdd.emplace_back(mAddWork.Bind({n, localSize}, {mBuffers[i + 1], mBuffers[i]}));
    n = NextSize(n, localSize);
  }

  mScanCmd.Record([&](vk::CommandBuffer commandBuffer)
  {
    mBuffers[0].CopyFrom(commandBuffer, mLocalInput);
    int n = size;

    mScan[0].PushConstant(commandBuffer, n);
    mScan[0].Record(commandBuffer);
    Vortex2D::Renderer::BufferBarrier(mBuffers[0].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
    Vortex2D::Renderer::BufferBarrier(mBuffers[1].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

    for (std::size_t i = 1; i < mScan.size(); i++)
    {
      mScan[i].PushConstant(commandBuffer, NextSize(n, localSize));
      mScan[i].Record(commandBuffer);
      Vortex2D::Renderer::BufferBarrier(mBuffers[i - 1].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
      Vortex2D::Renderer::BufferBarrier(mBuffers[i].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

      mAdd[i - 1].PushConstant(commandBuffer, n);
      mAdd[i - 1].Record(commandBuffer);
      Vortex2D::Renderer::BufferBarrier(mBuffers[i - 1].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

      n = NextSize(n, localSize);
    }

    mLocalOutput.CopyFrom(commandBuffer, mBuffers[0]);
  });
}

std::vector<float> Scan::operator()(const std::vector<float>& input)
{
  Vortex2D::Renderer::CopyFrom(mLocalInput, input);

  mScanCmd.Submit();
  mScanCmd.Wait();

  std::vector<float> output(mLocalOutput.Size() / sizeof(float), 0.0f);
  Vortex2D::Renderer::CopyTo(mLocalOutput, output);
  return output;
}

void CheckScan()
{
  int size = 300;
  Scan scan(*gDevice, size, 256);

  std::vector<float> inputData(size, 1.0f);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  auto output = scan(inputData);

  std::vector<float> expectedOutput(size);
  std::inclusive_scan(std::execution::seq, inputData.begin(), inputData.end(), expectedOutput.begin());

  for (std::size_t i = 0; i < size; i++)
  {
    if (output[i] != expectedOutput[i])
    {
      std::cout << "Diference at " << i << " values " << output[i] << " != " << expectedOutput[i] << std::endl;
    }
  }

  std::cout << "Scan complete" << std::endl;
}

static void Scan_GPU_Subgroup(benchmark::State& state)
{
  auto size = state.range(0);

  Scan scan(*gDevice, size, 512);

  std::vector<float> inputData(size, 1.0f);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(scan(inputData));
  }
}

static void Scan_GPU_SharedMemory(benchmark::State& state)
{
  auto size = state.range(0);

  Vortex2D::Fluid::PrefixScan scan(*gDevice, {size, 1});

  Vortex2D::Renderer::Buffer<float> localInput(*gDevice, size, VMA_MEMORY_USAGE_CPU_TO_GPU);
  Vortex2D::Renderer::Buffer<float> input(*gDevice, size, VMA_MEMORY_USAGE_GPU_ONLY);

  Vortex2D::Renderer::Buffer<float> localOutput(*gDevice, size, VMA_MEMORY_USAGE_CPU_ONLY);
  Vortex2D::Renderer::Buffer<float> output(*gDevice, size, VMA_MEMORY_USAGE_GPU_TO_CPU);

  Vortex2D::Renderer::Buffer<Vortex2D::Renderer::DispatchParams> dispatchParams(*gDevice);

  auto boundScan = scan.Bind(input, output, dispatchParams);

  Vortex2D::Renderer::CommandBuffer cmd(*gDevice);
  cmd.Record([&](vk::CommandBuffer commandBuffer)
  {
    input.CopyFrom(commandBuffer, localInput);
    boundScan.Record(commandBuffer);
    localOutput.CopyFrom(commandBuffer, output);
  });

  std::vector<float> outputData(size);

  for (auto _ : state)
  {
    cmd.Submit();
    cmd.Wait();

    Vortex2D::Renderer::CopyTo(localOutput, outputData);
  }
}

static void Scan_CPU_Seq(benchmark::State& state)
{
  auto size = state.range(0);

  std::vector<float> inputData(size, 1.0f);
  std::vector<float> outputData(size);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(std::inclusive_scan(std::execution::seq, inputData.begin(), inputData.end(), outputData.begin()));
  }
}

static void Scan_CPU_Par(benchmark::State& state)
{
  auto size = state.range(0);

  std::vector<float> inputData(size, 1.0f);
  std::vector<float> outputData(size);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(std::inclusive_scan(std::execution::par, inputData.begin(), inputData.end(), outputData.begin()));
  }
}


BENCHMARK(Scan_GPU_Subgroup)->Range(8, 8<<20);
BENCHMARK(Scan_GPU_SharedMemory)->Range(8, 8<<20);
BENCHMARK(Scan_CPU_Seq)->Range(8, 8<<20);
BENCHMARK(Scan_CPU_Par)->Range(8, 8<<20);
