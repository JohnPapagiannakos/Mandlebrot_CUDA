enable_language(CUDA)

add_library(include
            cudaoperators.cu
            cudaoperators.hpp
)

target_compile_features(include PUBLIC cxx_std_11)

set_target_properties(include
                      PROPERTIES 
                      CUDA_SEPARABLE_COMPILATION ON 
                      RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)
target_link_libraries(include)