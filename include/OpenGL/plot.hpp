#ifndef PLOT_HPP
#define PLOT_HPP

#include <GL/glew.h> // <- NOTE: Include this lib first.
#include <GL/glut.h>
// #include <cstdlib>
#include <iostream>
#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "OGLConstants.hpp"

bool FIGLOCK = false;

/* Let a 24-bit number N = B23 B22 ... B01 B00
    We can assume that this number can be represented in RGB 24bit color model as follows:

    <Red>  := B23 B22 ... B17 (8bit)
    <Green> := B15 B14 ... B08 (8bit)
    <Blue>  := B07 B06 ... B00 (8bit)

    Each color information can be extracted through typeXbyte (X:R,G,B).

*/

/*
    Extract Red color from a 24bit number (24-bit RGB model).
    */
inline GLubyte uint2Rbyte(const unsigned int &value)
{
    return value >> 16;
}

/*
    Extract Green color from a 24bit number (24-bit RGB model).
 */
inline GLubyte uint2Gbyte(const unsigned int &value)
{
    return (value >> 8) & 255;
}

/*
    Extract Blue color from a 24bit number (24-bit RGB model).
 */
inline GLubyte uint2Bbyte(const unsigned int &value)
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
            FIGLOCK = true;
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
            glutInitWindowSize(_resolution[0], _resolution[1]);
            // for (auto &d : _resolution)
            // {
            //     std::cout << d << "\t";
            // }
            // std::cout << std::endl;

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

            std::vector<T> tmpdata = _data;
            
            T max = *max_element(_data.begin(), _data.end());
            T min = *min_element(_data.begin(), _data.end());

            // Normalize data in [0, 1]
            if (min > 0 || std::isinf(min))
            {
                for (size_t i = 0; i < prod_dims; i++)
                {
                    tmpdata[i] /= max;
                }
            }
            else if (min < 0 && max > 0)
            {
                T diff = max - min;
                for (size_t i = 0; i < prod_dims; i++)
                {
                    tmpdata[i] = (_data[i] - min) / diff;
                }
            }
            else if (max < 0)
            {
                for (size_t i = 0; i < prod_dims; i++)
                {
                    tmpdata[i] = _data[i] / max;
                }
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
                    data[lin_idx3D + 0] = uint2Rbyte(test);
                    data[lin_idx3D + 1] = uint2Gbyte(test);
                    data[lin_idx3D + 2] = uint2Bbyte(test);
                    lin_idx++;
                }
            }

            glDrawPixels(_resolution[0], _resolution[1], GL_RGB, GL_UNSIGNED_BYTE, data);

            glutSwapBuffers();

            // glutMainLoop();
        }

        /*
            clim argument specifies the data values that map to the first and last elements of the colormap. 
            Specify clims as a two-element vector of the form {cmin, cmax}, where values less than or equal 
            to cmin map to the first color in the colormap and values greater than or equal to cmax map to 
            the last color in the colormap. Specify clims after name-value pair arguments.
         */
        void plot(const std::vector<T> &_data, const std::array<T,2> &clim)
        {
            size_t prod_dims = _data.size();

            std::vector<T> tmpdata = _data;
            
            if (clim[0] >= clim[1])
            {
                std::cerr << "Error! clim[0] >= clim[1]" << std::endl;
                return;
            }

            /*
                             { clim[0], if _data[i] < clim[0]
                tmpdata[i] = { _data  , if _data[i] >= clim[0] and  _data[i] <= clim[1]
                             { clim[1], otherwise
            */
            for (size_t i=0; i<prod_dims; i++)
            {
                tmpdata[i] = ((_data[i] < clim[0]) * clim[0]) + (((_data[i] >= clim[0]) * (_data[i] <= clim[1])) * _data[i]) + ((_data[i] > clim[1]) * clim[1]);
            }

            plot(tmpdata);
        }

    public:
        figure() : _resolution({1920, 1080})
        {
            if(!FIGLOCK)
            {
                initFigure();
            }
        }

        figure(std::array<size_t, 2> Resolution) : _resolution(Resolution)
        {
            if (!FIGLOCK)
            {
                initFigure();
            }
        }

        void plotRGB(const std::vector<double> &_DATA)
        {
            // initFigure();
            plot(_DATA);
        }

        void plotRGB(const std::vector<double> &_DATA, const std::array<T, 2> &clim)
        {
            // initFigure();
            plot(_DATA, clim);
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