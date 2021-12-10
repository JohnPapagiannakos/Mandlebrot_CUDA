enable_language("CUDA")

add_executable(test_julia_cpu test_julia_cpu.cpp)

SET_TARGET_PROPERTIES(test_julia_cpu
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

target_link_libraries(test_julia include ${TEST_HEADER_FILES})
