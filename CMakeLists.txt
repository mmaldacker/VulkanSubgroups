cmake_minimum_required(VERSION 3.11)

project(VulkanSubgroups)

set(CMAKE_CXX_STANDARD 17)

# cmake file to download and include dependencies
include(FetchContent)

FetchContent_Declare(vortex2d
                     GIT_REPOSITORY      https://github.com/mmaldacker/Vortex2D.git
                     GIT_TAG             master)
FetchContent_GetProperties(vortex2d)
if(NOT vortex2d_POPULATED)
  FetchContent_Populate(vortex2d)
  add_subdirectory(${vortex2d_SOURCE_DIR} ${vortex2d_BINARY_DIR})
endif()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark
                     GIT_REPOSITORY      https://github.com/google/benchmark.git
                     GIT_TAG             v1.4.1)
if(NOT benchmark_POPULATED)
  FetchContent_Populate(benchmark)
  add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})
endif()

compile_shader(SOURCES
                 ${CMAKE_CURRENT_SOURCE_DIR}/Reduce.comp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Scan.comp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Add.comp
               OUTPUT "vulkansubgroups_spirv"
               VERSION 1.1)

add_executable(VulkanSubgroups
    main.cpp
    reduce.h
    reduce.cpp
    scan.h
    scan.cpp
    Reduce.comp
    Scan.comp
    Add.comp
    ${CMAKE_CURRENT_BINARY_DIR}/vulkansubgroups_spirv.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/vulkansubgroups_spirv.h
)

target_include_directories(VulkanSubgroups PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
    vortex2d_copy_dll(VulkanSubgroups)
endif()

target_link_libraries(VulkanSubgroups vortex2d benchmark)
