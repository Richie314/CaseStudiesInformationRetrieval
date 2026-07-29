cmake_minimum_required(VERSION 3.16)
project(information_retrieval LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(external/gef)

add_executable(information_retrieval src/main.cpp)
target_link_libraries(information_retrieval PRIVATE gef::gef)

target_compile_features(information_retrieval PRIVATE cxx_std_20)
