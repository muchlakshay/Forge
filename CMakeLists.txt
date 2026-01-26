cmake_minimum_required(VERSION 3.31)
project(Forge)

set(CMAKE_CXX_STANDARD 20)

add_executable(main
        tests/main.cpp
)
target_include_directories(main PUBLIC C:/Users/MSI/Documents/CPPLibs/eigen-3.4.0 core/tensor)