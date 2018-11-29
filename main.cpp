#include <Vortex2D/Vortex2D.h>
#include <iostream>
#include <numeric>

#include "vulkansubgroups_spirv.h"

int main()
{
    Vortex2D::Renderer::Instance instance("VulkanSubgroups", {}, true);
    auto physicalDevice = instance.GetPhysicalDevice();

    auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
    auto subgroupProperties = properties.get<vk::PhysicalDeviceSubgroupProperties>();

    std::cout << "Subgroup size: " << subgroupProperties.subgroupSize << std::endl;
    std::cout << "Subgroup supported operations: " << vk::to_string(subgroupProperties.supportedOperations) << std::endl;

    Vortex2D::Renderer::Device device(physicalDevice);

    int size = 200;
    Vortex2D::Renderer::ComputeSize computeSize(size);
    Vortex2D::Renderer::Work reduceWork(device,
                                        computeSize,
                                        Vortex2D::SPIRV::Reduce_comp,
                                        Vortex2D::Renderer::SpecConst(Vortex2D::Renderer::SpecConstValue(2, 32)));

    Vortex2D::Renderer::Buffer<float> input(device, size, VMA_MEMORY_USAGE_CPU_ONLY);
    Vortex2D::Renderer::Buffer<float> output(device, 1, VMA_MEMORY_USAGE_CPU_ONLY);

    auto reduceBound = reduceWork.Bind({input, output});

    std::vector<float> inputData(size, 1.0f);
    //std::iota(inputData.begin(), inputData.end(), 1.0f);

    CopyFrom(input, inputData);

    ExecuteCommand(device, [&](vk::CommandBuffer commandBuffer)
    {
        reduceBound.PushConstant(commandBuffer, size);
        reduceBound.Record(commandBuffer);
    });

    float total;
    CopyTo(output, total);

    std::cout << "Total " << total << std::endl;
    std::cout << "Expected total " << 0.5f * size * (size + 1) << std::endl;

    Vortex2D::Renderer::Work scanWork(device,
                                      computeSize,
                                      Vortex2D::SPIRV::Scan_comp,
                                      Vortex2D::Renderer::SpecConst(Vortex2D::Renderer::SpecConstValue(2, 32)));

    Vortex2D::Renderer::Buffer<float> partial_sums(device, 1, VMA_MEMORY_USAGE_CPU_ONLY);
    Vortex2D::Renderer::Buffer<float> scan_output(device, size, VMA_MEMORY_USAGE_CPU_ONLY);

    auto scanBound = scanWork.Bind({input, scan_output, partial_sums});

    ExecuteCommand(device, [&](vk::CommandBuffer commandBuffer)
    {
        scanBound.PushConstant(commandBuffer, size);
        scanBound.Record(commandBuffer);
    });

    std::vector<float> scan_output_data(size, 0.0f);
    CopyTo(scan_output, scan_output_data);

    std::cout << "Result" << std::endl;
    for (auto value: scan_output_data) std::cout << value << ", " << std::endl;;

}
