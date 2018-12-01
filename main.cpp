#include <Vortex2D/Vortex2D.h>
#include <Vortex2D/Engine/LinearSolver/Reduce.h>
#include <benchmark/benchmark.h>
#include <iostream>
#include <numeric>
#include <execution>
#include <cassert>

#include "vulkansubgroups_spirv.h"

class Reduce
{
public:
    Reduce(const Vortex2D::Renderer::Device& device,
           int size,
           int localSize)
        : mReduceCmd(device)
        , mReduceWork(device, {size, localSize}, Vortex2D::SPIRV::Reduce_comp)
        , mLocalInput(device, size, VMA_MEMORY_USAGE_CPU_TO_GPU)
        , mLocalOutput(device, 1, VMA_MEMORY_USAGE_GPU_TO_CPU)
    {
        int n = size;
        while (n > 1)
        {
            mBuffers.emplace_back(device, n, VMA_MEMORY_USAGE_GPU_ONLY);
            n = (n + localSize) / localSize;
        }

        mBuffers.emplace_back(device, 1, VMA_MEMORY_USAGE_GPU_ONLY);

        for (std::size_t i = 0; i < mBuffers.size() - 1; i++)
        {
            mReduce.emplace_back(mReduceWork.Bind({mBuffers[i], mBuffers[i + 1]}));
        }

        mReduceCmd.Record([&](vk::CommandBuffer commandBuffer)
        {
            mBuffers.front().CopyFrom(commandBuffer, mLocalInput);
            int n = size;
            for (std::size_t i = 0; i < mReduce.size(); i++)
            {
                mReduce[i].PushConstant(commandBuffer, n);
                mReduce[i].Record(commandBuffer);
                Vortex2D::Renderer::BufferBarrier(mBuffers[i + 1].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
                n = (n + localSize) / localSize;
            }
            mLocalOutput.CopyFrom(commandBuffer, mBuffers.back());
        });
    }

    float operator()(const std::vector<float>& input)
    {
        Vortex2D::Renderer::CopyFrom(mLocalInput, input);

        mReduceCmd.Submit();
        mReduceCmd.Wait();

        float total = 0.0;
        Vortex2D::Renderer::CopyTo(mLocalOutput, total);
        return total;
    }

private:
    Vortex2D::Renderer::CommandBuffer mReduceCmd;
    Vortex2D::Renderer::Work mReduceWork; // reduce shader
    std::vector<Vortex2D::Renderer::Buffer<float>> mBuffers; // buffers for input, intermediate results and output
    Vortex2D::Renderer::Buffer<float> mLocalInput, mLocalOutput; // buffers for copying input/output to device
    std::vector<Vortex2D::Renderer::Work::Bound> mReduce; // bound reduce shaders for each level
};

Vortex2D::Renderer::Device* gDevice = nullptr;

static void Reduce_CPU_Seq(benchmark::State& state)
{
    int size = state.range(0);

    std::vector<float> inputData(size, 1.0f);
    std::iota(inputData.begin(), inputData.end(), 1.0f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::reduce(std::execution::seq, inputData.begin(), inputData.end()));
    }
}

static void Reduce_CPU_Par(benchmark::State& state)
{
    int size = state.range(0);

    std::vector<float> inputData(size, 1.0f);
    std::iota(inputData.begin(), inputData.end(), 1.0f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::reduce(std::execution::par, inputData.begin(), inputData.end()));
    }
}

static void Reduce_GPU_Subgroup(benchmark::State& state)
{
    int size = state.range(0);

    Reduce reduce(*gDevice, size, 512);

    std::vector<float> inputData(size, 1.0f);
    std::iota(inputData.begin(), inputData.end(), 1.0f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(reduce(inputData));
    }
}

static void Reduce_GPU_SharedMemory(benchmark::State& state)
{
    int size = state.range(0);

    Vortex2D::Fluid::ReduceSum reduce(*gDevice, {size, 1});

    Vortex2D::Renderer::Buffer<float> localInput(*gDevice, size, VMA_MEMORY_USAGE_CPU_TO_GPU);
    Vortex2D::Renderer::Buffer<float> input(*gDevice, size, VMA_MEMORY_USAGE_GPU_ONLY);

    Vortex2D::Renderer::Buffer<float> localOutput(*gDevice, 1, VMA_MEMORY_USAGE_CPU_ONLY);
    Vortex2D::Renderer::Buffer<float> output(*gDevice, 1, VMA_MEMORY_USAGE_GPU_TO_CPU);

    auto boundReduce = reduce.Bind(input, output);

    Vortex2D::Renderer::CommandBuffer cmd(*gDevice);
    cmd.Record([&](vk::CommandBuffer commandBuffer)
    {
        input.CopyFrom(commandBuffer, localInput);
        boundReduce.Record(commandBuffer);
        localOutput.CopyFrom(commandBuffer, output);
    });

    for (auto _ : state)
    {
        cmd.Submit();
        cmd.Wait();

        float total = 0.0;
        Vortex2D::Renderer::CopyTo(localOutput, total);
    }
}

BENCHMARK(Reduce_GPU_Subgroup)->Range(8, 8<<20);
BENCHMARK(Reduce_GPU_SharedMemory)->Range(8, 8<<20);
BENCHMARK(Reduce_CPU_Seq)->Range(8, 8<<20);
BENCHMARK(Reduce_CPU_Par)->Range(8, 8<<20);

void CheckReduce()
{
    int size = 300;
    Reduce reduce(*gDevice, size, 256);

    std::vector<float> inputData(size, 1.0f);
    std::iota(inputData.begin(), inputData.end(), 1.0f);

    float total = reduce(inputData);

    std::cout << "Total " << total << std::endl;
    std::cout << "Expected total " << 0.5f * size * (size + 1) << std::endl;
}

class Scan
{
public:
    Scan(const Vortex2D::Renderer::Device& device,
    int size,
    int localSize)
        : mScanCmd(device)
        , mScanWork(device, {size, localSize}, Vortex2D::SPIRV::Scan_comp)
        , mLocalInput(device, size, VMA_MEMORY_USAGE_CPU_TO_GPU)
        , mLocalOutput(device, 1, VMA_MEMORY_USAGE_GPU_TO_CPU)
    {
        int n = size;
        while (n > 1)
        {
            mBuffers.emplace_back(device, n, VMA_MEMORY_USAGE_GPU_ONLY);
            n = (n + localSize) / localSize;
        }

        mBuffers.emplace_back(device, 1, VMA_MEMORY_USAGE_GPU_ONLY);

        for (std::size_t i = 0; i < mBuffers.size() - 1; i++)
        {
            mScan.emplace_back(mScanWork.Bind({mBuffers[i], mBuffers[i + 1]}));
        }

        mScanCmd.Record([&](vk::CommandBuffer commandBuffer)
        {
            mBuffers.front().CopyFrom(commandBuffer, mLocalInput);
            int n = size;
            for (std::size_t i = 0; i < mScan.size(); i++)
            {
                mScan[i].PushConstant(commandBuffer, n);
                mScan[i].Record(commandBuffer);
                Vortex2D::Renderer::BufferBarrier(mBuffers[i + 1].Handle(), commandBuffer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
                n = (n + localSize) / localSize;
            }
            mLocalOutput.CopyFrom(commandBuffer, mBuffers.back());
        });
    }

    float operator()(const std::vector<float>& input)
    {
        Vortex2D::Renderer::CopyFrom(mLocalInput, input);

        mScanCmd.Submit();
        mScanCmd.Wait();

        float total = 0.0;
        Vortex2D::Renderer::CopyTo(mLocalOutput, total);
        return total;
    }

private:
    Vortex2D::Renderer::CommandBuffer mScanCmd;
    Vortex2D::Renderer::Work mScanWork;

    std::vector<Vortex2D::Renderer::Buffer<float>> mBuffers; // buffers for input, intermediate results and output
    Vortex2D::Renderer::Buffer<float> mLocalInput, mLocalOutput; // buffers for copying input/output to device
    std::vector<Vortex2D::Renderer::Work::Bound> mScan; // bound scan shaders for each level
};

void CheckScan()
{
    int size = 300;
    Scan scan(*gDevice, size, 256);

    std::vector<float> inputData(size, 1.0f);
    std::iota(inputData.begin(), inputData.end(), 1.0f);

    float total = scan(inputData);

    std::cout << "Total " << total << std::endl;
    //std::cout << "Expected total " << std::inclusive_scan(std::execution::seq, inputData.begin(), inputData.end()) << std::endl;
}

int main(int argc, char** argv)
{
    Vortex2D::Renderer::Instance instance("VulkanSubgroups", {}, true);
    auto physicalDevice = instance.GetPhysicalDevice();

    auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
    auto subgroupProperties = properties.get<vk::PhysicalDeviceSubgroupProperties>();

    std::cout << "Subgroup size: " << subgroupProperties.subgroupSize << std::endl;
    std::cout << "Subgroup supported operations: " << vk::to_string(subgroupProperties.supportedOperations) << std::endl;

    Vortex2D::Renderer::Device device(physicalDevice);
    gDevice = &device;

    CheckReduce();
    CheckScan();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
