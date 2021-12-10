enable_language("CUDA")

add_executable(test_julia test_julia.cu)

SET_TARGET_PROPERTIES(test_julia 
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

target_link_libraries(test_julia include ${TEST_HEADER_FILES})
