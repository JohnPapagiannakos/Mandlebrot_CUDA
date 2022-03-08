# enable_language("CUDA")

find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

add_executable(test_compute_and_plot test_compute_and_plot.cpp)

SET_TARGET_PROPERTIES(test_compute_and_plot
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

target_link_libraries(test_compute_and_plot
                        include 
                        ${TEST_HEADER_FILES}
                        OpenGL::GL
                        OpenGL::GLU
                        GLUT::GLUT
                        GLEW::GLEW
                        m)

add_dependencies(test_compute_and_plot OpenGL::OpenGL)
include_directories(${GLUT_INCLUDE_DIRS} )
target_include_directories(test_compute_and_plot PUBLIC ${INCLUDE_DIR})

