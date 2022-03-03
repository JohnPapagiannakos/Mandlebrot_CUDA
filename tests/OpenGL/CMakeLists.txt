# CMake entry point
cmake_minimum_required (VERSION 3.10)
project (OGL)

find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

# set(CMAKE_CXX_FLAGS              "${CMAKE_CXX_FLAGS} -std=c++17")
# set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -fopenmp -funroll-loops -march=native -mtune=native ")
# set(CMAKE_CXX_FLAGS_DEBUG        "-g")
# set(BIN_DIR                      "${CMAKE_CURRENT_LIST_DIR}/bin")
# set(INCLUDE_DIR                  "${CMAKE_CURRENT_LIST_DIR}/include")

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

add_executable(plotgrid plotgrid.cpp
)

add_dependencies(plotgrid OpenGL::OpenGL)
include_directories(${GLUT_INCLUDE_DIRS} )
target_include_directories(plotgrid PUBLIC ${INCLUDE_DIR})
target_link_libraries(plotgrid  
					OpenGL::GL
					OpenGL::GLU
					GLUT::GLUT
					GLEW::GLEW
					m)

SET_TARGET_PROPERTIES(
        plotgrid
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
    )	
