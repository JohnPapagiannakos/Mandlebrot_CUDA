enable_language("CUDA")

find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

add_executable(test_julia_plot_cuda test_julia_plot.cu)

SET_TARGET_PROPERTIES(test_julia_plot_cuda 
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

target_link_libraries(test_julia_plot_cuda 
                        include 
                        ${TEST_HEADER_FILES}
                        OpenGL::GL
                        OpenGL::GLU
                        GLUT::GLUT
                        GLEW::GLEW
                        m)

add_dependencies(test_julia_plot_cuda OpenGL::OpenGL)
include_directories(${GLUT_INCLUDE_DIRS} )
target_include_directories(test_julia_plot_cuda PUBLIC ${INCLUDE_DIR})
