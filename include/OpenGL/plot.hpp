#ifndef PLOT_HPP
#define PLOT_HPP

#include <GL/glew.h> // <- NOTE: Include this lib first.
#include <GL/glut.h>
// #include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <array>
#include <cassert>

#include "OGLConstants.hpp"

/* Let a 24-bit number N = B23 B22 ... B01 B00
   We can assume that this number can be represented in RGB 24bit color model as follows:

    <Red>  := B23 B22 ... B17 (8bit)
   <Green> := B15 B14 ... B08 (8bit)
   <Blue>  := B07 B06 ... B00 (8bit)

   Each color information can be extracted through typeXbyte (X:R,G,B).

*/

template <typename T>
inline T type2Rbyte(const T &value)
{
    return value >> 16;
}

template <typename T>
inline T type2Gbyte(const T &value)
{
    return (value >> 8) & 255;
}

template <typename T>
inline T type2Bbyte(const T &value)
{
    return value & 255;
}

template <typename T>
class figure
{
    private:
        std::array<size_t, 2> _resolution;

        const std::array<size_t, 2> _maxResolution = {1920, 1080};

        int windowID = -1;

        void initFigure(void)
        {
            int argc = 1;
            char *argv[1] = {(char *)"unused"};
            glutInit(&argc, argv);
            
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
            glutInitWindowSize(_resolution[0], _resolution[1]);
            
            GLenum error_code = glewInit();
            if (error_code != 0)
            {
                std::cerr << "Error initiallizing glew : " << glewGetErrorString(error_code) << std::endl;
            }

            assert((_resolution[0] <= _maxResolution[0] && _resolution[1] <= _maxResolution[1]));
        }

        void plot(const std::vector<T> &_data)
        {
            size_t prod_dims = _data.size();
            assert(_resolution[0] * _resolution[1] == prod_dims);

            // Normalize data in [0, 1]
            T max = *max_element(_data.begin(), _data.end());

            std::vector<T> tmpdata = _data;

            for (size_t i = 0; i < prod_dims; i++)
            {
                tmpdata[i] /= max;
            }

            // Multiply by 2 ^ 24
            for (size_t i = 0; i < prod_dims; i++)
            {
                tmpdata[i] *= MAXINT_24BIT;
            }

            GLubyte data[_resolution[0] * _resolution[1] * 3];
            for (size_t col = 0; col < _resolution[1]; ++col)
            {
                size_t lin_idx = (col * _resolution[0]);
                for (size_t row = 0; row < _resolution[0]; ++row)
                {
                    unsigned int test = static_cast<unsigned int>(tmpdata[row + col * _resolution[0]]);
                    // [0] : R, [1] : G, [2] : B
                    size_t lin_idx3D = lin_idx * 3;
                    data[lin_idx3D + 0] = (type2Rbyte<unsigned int>(test));
                    data[lin_idx3D + 1] = (type2Gbyte<unsigned int>(test));
                    data[lin_idx3D + 2] = (type2Bbyte<unsigned int>(test));
                    lin_idx++;
                }
            }

            glDrawPixels(_resolution[0], _resolution[1], GL_RGB, GL_UNSIGNED_BYTE, data);

            glutSwapBuffers();

            // glutMainLoop();
        }

    public:
        figure() : _resolution({1920, 1080})
        {
            initFigure();
        }

        figure(std::array<size_t, 2> Resolution) : _resolution(Resolution)
        {
            initFigure();
        }

        void plotRGB(const std::vector<double> &_DATA)
        {
            // initFigure();
            for(auto &d: _resolution)
            {
                std::cout << d << "\t";
            }
            std::cout << std::endl;
            plot(_DATA);
        }

        void newFigure(void)
        {
            newFigure("Figure");
        }

        void newFigure(const char *title)
        {
            windowID = glutCreateWindow(title);
            assert(windowID != -1);
            // glutSetWindow(windowID);

            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            return;
        }

        void closeFigure(void)
        {
            // glutLeaveMainLoop(); // needs freeglut.h
            if (windowID > 0)
            { 
                glutDestroyWindow(windowID);
                // glutDestroyWindow(glutGetWindow());
            }
            return;
        }

        void showFigure(void)
        {
            glutMainLoop();
        }
};




#endif