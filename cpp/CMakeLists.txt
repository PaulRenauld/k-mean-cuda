cmake_minimum_required(VERSION 3.13)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(cpp
        cxxopts.hpp
        kmean.cpp
        kmean_computer.h
        Makefile
        Point.cpp
        Point.h seq_computer.cpp seq_computer.h silhouette_finder.cpp silhouette_finder.h data_structure.h kmean_computer.cpp)
