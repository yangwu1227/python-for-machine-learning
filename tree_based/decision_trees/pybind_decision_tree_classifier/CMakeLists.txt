cmake_minimum_required(VERSION 3.15)
project(decision-tree-classifier VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard for the entire project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Add subdirectory with the module's CMakeLists.txt
add_subdirectory(src/decision_tree_classifier/src)
