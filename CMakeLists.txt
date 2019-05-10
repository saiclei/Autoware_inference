cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(OFFLINE_TEST_PP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(PCL 1.8 REQUIRED COMPONENTS common io)

macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

set_ifndef(TRT_LIB_DIR /home/saiclei/Research/Softwares/TensorRT-5.0.2.6/targets/x86_64-linux-gnu/lib)
set_ifndef(TRT_INC_DIR /home/saiclei/Research/Softwares/TensorRT-5.0.2.6/include)

# set flags for TensorRT availability
find_library(_TRT_INC_DIR NvInfer.h NvHINTS ${TRT_INC_DIR} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR ${_TRT_INC_DIR})
find_library(NVINFER NAMES nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
find_library(NVPARSERS NAMES nvparsers HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
find_library(NVONNXPARSERS NAMES nvonnxparser HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
if(NVINFER AND NVPARSERS AND NVONNXPARSERS)
   message("TensorRT is available!")
   message("NVINFER: ${NVINFER}")
   message("NVPARSERS: ${NVPARSERS}")
   message("NVONNXPARSERS: ${NVONNXPARSERS}")
   set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()


# set flags for CUDA availability
find_package(CUDA)
if (CUDA_FOUND)
    message("CUDA is availabel!")
    message("CUDA Libs: ${CUDA_LIBRARIES}")
    message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
    set(CUDA_AVAIL ON)
else()
    message("CUDA NOT FOUND")
    set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)


# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
PATH_SUFFIXES lib lib64 bin
DOC "CUDNN library." )
if(CUDNN_LIBRARY)
   message("CUDNN is available!")
   message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
   set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()


include_directories(${PCL_INCLUDE_DIRS}
                    ${TRT_INC_DIR}
                    /mnt/raid1/Research/Autoware_inference/
                    /usr/local/cuda/include)

set(SOURCE_FILES
    src/offline_test.cpp 
    src/point_pillars.cpp
    src/preprocess_points.cpp)


set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
cuda_add_library(gpu_point_pillars_lib 
         src/preprocess_points_cuda.cu
         src/anchor_mask_cuda.cu
         src/scatter_cuda.cu
         src/postprocess_cuda.cu
         src/nms_cuda.cu
         #         OPTIONS -arch sm_70
        )

target_link_libraries(gpu_point_pillars_lib
   ${CUDA_LIBRARIES}
  )


link_directories(${PCL_LIBRARY_DIRS}
                 /usr/local/cuda/lib64
                 /home/saiclei/Research/Software/TensorRT-5.0.2.6/lib)

add_definitions(${PCL_DEFINITIONS})


add_executable(offline_test_pp ${SOURCE_FILES})

target_link_libraries(offline_test_pp ${PCL_LIBRARIES} 
                                      ${CUDA_LIBRARIES} 
                                      ${CUDA_CUBLAS_LIBRARIES}
                                      ${CUDA_curand_LIBRARY}
                                      ${CUDNN_LIBRARY}
                                      ${NVINFER}
                                      ${NVPARSERS}
                                      ${NVONNXPARSERS}
                                      gpu_point_pillars_lib)


