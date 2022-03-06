# enable_language("CUDA")

find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

add_executable(test_julia_plot test_julia_plot.cpp)

SET_TARGET_PROPERTIES(test_julia_plot
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

target_link_libraries(test_julia_plot
                        include 
                        ${TEST_HEADER_FILES}
                        OpenGL::GL
                        OpenGL::GLU
                        GLUT::GLUT
                        GLEW::GLEW
                        m)

add_dependencies(test_julia_plot OpenGL::OpenGL)
include_directories(${GLUT_INCLUDE_DIRS} )
target_include_directories(test_julia_plot PUBLIC ${INCLUDE_DIR})

