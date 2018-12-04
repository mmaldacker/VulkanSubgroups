#include "reduce.h"
#include "scan.h"

#include <benchmark/benchmark.h>
#include <iostream>

Vortex2D::Renderer::Device* gDevice = nullptr;

int main(int argc, char** argv)
{
  Vortex2D::Renderer::Instance instance("VulkanSubgroups", {}, false);
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
