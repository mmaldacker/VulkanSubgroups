#include "reduce.h"
#include <benchmark/benchmark.h>
#include <numeric>
#include <execution>
#include <iostream>
#include "vulkansubgroups_spirv.h"

namespace
{
int NextSize(int size, int localSize)
{
  return (size + localSize) / localSize;
}
}

Reduce::Reduce(const Vortex2D::Renderer::Device& device,
               int size,
               int localSize)
  : mTimer(device)
  , mUploadCmd(device)
  , mDownloadCmd(device)
  , mReduceCmd(device)
  , mReduceWork(device, {size, localSize}, Vortex2D::SPIRV::Reduce_comp)
  , mLocalInput(device, size, VMA_MEMORY_USAGE_CPU_TO_GPU)
  , mLocalOutput(device, 1, VMA_MEMORY_USAGE_GPU_TO_CPU)
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
    mReduce.emplace_back(mReduceWork.Bind({n, localSize}, {mBuffers[i], mBuffers[i + 1]}));
    n = NextSize(n, localSize);
  }

  mUploadCmd.Record([&](vk::CommandBuffer commandBuffer)
  {
    mBuffers.front().CopyFrom(commandBuffer, mLocalInput);
  });

  mDownloadCmd.Record([&](vk::CommandBuffer commandBuffer)
  {
    mLocalOutput.CopyFrom(commandBuffer, mBuffers.back());
  });

  mReduceCmd.Record([&](vk::CommandBuffer commandBuffer)
  {
    mTimer.Start(commandBuffer);

    int n = size;
    for (std::size_t i = 0; i < mReduce.size(); i++)
    {
      mReduce[i].PushConstant(commandBuffer, n);
      mReduce[i].Record(commandBuffer);

      Vortex2D::Renderer::BufferBarrier(mBuffers[i + 1].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

      n = NextSize(n, localSize);
    }

    mTimer.Stop(commandBuffer);
  });
}

void Reduce::Upload(const std::vector<float>& input)
{
  Vortex2D::Renderer::CopyFrom(mLocalInput, input);
  mUploadCmd.Submit();
}

float Reduce::Download()
{
  mDownloadCmd.Submit();
  mDownloadCmd.Wait();

  float total = 0.0;
  Vortex2D::Renderer::CopyTo(mLocalOutput, total);
  return total;
}

void Reduce::Submit()
{
  mReduceCmd.Submit();
  mReduceCmd.Wait();
}

uint64_t Reduce::GetElapsedNs()
{
  return mTimer.GetElapsedNs();
}

static void Reduce_CPU_Seq(benchmark::State& state)
{
  auto size = state.range(0);

  std::vector<float> inputData(size, 1.0f);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(std::reduce(std::execution::seq, inputData.begin(), inputData.end()));
  }
}

static void Reduce_CPU_Par(benchmark::State& state)
{
  auto size = state.range(0);

  std::vector<float> inputData(size, 1.0f);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(std::reduce(std::execution::par, inputData.begin(), inputData.end()));
  }
}

static void Reduce_GPU_Subgroup(benchmark::State& state)
{
  auto size = state.range(0);

  Reduce reduce(*gDevice, size, 512);

  std::vector<float> inputData(size, 1.0f);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  for (auto _ : state)
  {
    reduce.Submit();
    state.SetIterationTime(reduce.GetElapsedNs() / 1000000000.0);
  }
}

static void Reduce_GPU_SharedMemory(benchmark::State& state)
{
  auto size = state.range(0);

  Vortex2D::Renderer::Timer timer(*gDevice);
  Vortex2D::Fluid::ReduceSum reduce(*gDevice, {size, 1});

  Vortex2D::Renderer::Buffer<float> input(*gDevice, size, VMA_MEMORY_USAGE_GPU_ONLY);
  Vortex2D::Renderer::Buffer<float> output(*gDevice, 1, VMA_MEMORY_USAGE_GPU_ONLY);

  auto boundReduce = reduce.Bind(input, output);

  Vortex2D::Renderer::CommandBuffer cmd(*gDevice);
  cmd.Record([&](vk::CommandBuffer commandBuffer)
  {
    timer.Start(commandBuffer);
    boundReduce.Record(commandBuffer);
    timer.Stop(commandBuffer);
  });

  for (auto _ : state)
  {
    cmd.Submit();
    cmd.Wait();
    state.SetIterationTime(timer.GetElapsedNs() / 1000000000.0);
  }
}

BENCHMARK(Reduce_GPU_Subgroup)->Range(8, 8<<20)->UseManualTime();
BENCHMARK(Reduce_GPU_SharedMemory)->Range(8, 8<<20)->UseManualTime();
BENCHMARK(Reduce_CPU_Seq)->Range(8, 8<<20);
BENCHMARK(Reduce_CPU_Par)->Range(8, 8<<20);

void CheckReduce()
{
  int size = 300;
  Reduce reduce(*gDevice, size, 256);

  std::vector<float> inputData(size, 1.0f);
  std::iota(inputData.begin(), inputData.end(), 1.0f);

  reduce.Upload(inputData);
  reduce.Submit();
  float total = reduce.Download();

  std::cout << "Total " << total << std::endl;
  std::cout << "Expected total " << 0.5f * size * (size + 1) << std::endl;
}
