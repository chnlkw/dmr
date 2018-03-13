
project(libengine)

#cuda_utils
add_library(cuda_utils cuda/cuda_utils.cpp)
target_include_directories(cuda_utils
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        PUBLIC ${CUDA_INCLUDE_DIRS})

#logging
add_library(logging logging/easylogging++.cc)
target_include_directories(logging
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/logging)

#engine
aux_source_directory(src SRC_FILES)
add_library(engine ${SRC_FILES})
target_link_libraries(engine cuda_utils logging ${CUDA_LIBRARIES})

target_include_directories(engine
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


