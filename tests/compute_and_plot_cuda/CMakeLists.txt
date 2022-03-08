enable_language("CUDA")

find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

set(TEST_SRC_FILE_NAMES test_julia_plot
                        test_julia_plot_iterative 
                        )

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

foreach(TEST IN LISTS TEST_SRC_FILE_NAMES)
    add_executable(${TEST} ${TEST}.cu)

    SET_TARGET_PROPERTIES(${TEST} 
                        PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
    )

    target_link_libraries(${TEST} 
                            include 
                            ${TEST_HEADER_FILES}
                            OpenGL::GL
                            OpenGL::GLU
                            GLUT::GLUT
                            GLEW::GLEW
                            m)

    add_dependencies(${TEST} OpenGL::OpenGL)
    include_directories(${GLUT_INCLUDE_DIRS} )
    target_include_directories(${TEST} PUBLIC ${INCLUDE_DIR})
endforeach()