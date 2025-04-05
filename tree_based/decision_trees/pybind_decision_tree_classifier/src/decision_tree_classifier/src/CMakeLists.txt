# Name the library _dtree so the resulting file is "_dtree.so"
add_library(_dtree MODULE decision_tree_classifier.cpp)

# Find and link Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(_dtree PRIVATE pybind11::module Eigen3::Eigen)

# Enable OpenMP for parallel prediction if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_dtree PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enable stronger optimizations
target_compile_options(_dtree PRIVATE 
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG -ffast-math>
)

# Add row-major memory layout for better compatibility with NumPy
target_compile_definitions(_dtree PRIVATE EIGEN_DEFAULT_TO_ROW_MAJOR)

set_target_properties(_dtree PROPERTIES
    PREFIX ""          
    SUFFIX ".so"       # Produce _dtree.so on macOS/Linux
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS _dtree DESTINATION decision_tree_classifier)
