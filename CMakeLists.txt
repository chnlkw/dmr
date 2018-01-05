cmake_minimum_required(VERSION 3.5)
project(dmr)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/include)

add_executable(dmr main.cpp)
