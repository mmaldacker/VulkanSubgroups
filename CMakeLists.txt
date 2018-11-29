cmake_minimum_required(VERSION 3.2)

project(VulkanSubgroups)

set(CMAKE_CXX_STANDARD 14)

# cmake file to download and include dependencies
include(cmake/DownloadProject.cmake)

download_project(PROJ                Vortex2D
                 GIT_REPOSITORY      https://github.com/mmaldacker/Vortex2D.git
                 GIT_TAG             master
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE})
add_subdirectory(${Vortex2D_SOURCE_DIR} ${Vortex2D_BINARY_DIR})

compile_shader(SOURCES
                 ${CMAKE_CURRENT_SOURCE_DIR}/Reduce.comp
                 ${CMAKE_CURRENT_SOURCE_DIR}/Scan.comp
               OUTPUT "vulkansubgroups_spirv"
               VERSION 1.1)

add_executable(VulkanSubgroups
    main.cpp
    Reduce.comp
    Scan.comp
    ${CMAKE_CURRENT_BINARY_DIR}/vulkansubgroups_spirv.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/vulkansubgroups_spirv.h
)

target_include_directories(VulkanSubgroups PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
    vortex2d_copy_dll(VulkanSubgroups)
endif()

target_link_libraries(VulkanSubgroups vortex2d)
