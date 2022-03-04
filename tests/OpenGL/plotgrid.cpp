#include <unistd.h>

#include "IO.hpp"

#include "OpenGL/plot.hpp"

int main(int argc, char **argv)
{
   
    std::string filename = "count.bin";

    std::array<size_t,2> resolution = {800, 800};

    figure<double> fig(resolution);
    long int prod_dims = resolution[0] * resolution[1];
    std::vector<double> _data(prod_dims, 1);

    // Read double data from file
    Read_From_File<double>(prod_dims, &_data[0], filename.c_str(), 0);
    fig.newFigure("Fig 1");
    fig.plotRGB(_data);
    sleep(2);
    fig.closeFigure();

    std::vector<double> test(prod_dims, 1);
    for (size_t i = 0; i < prod_dims; i++)
    {
        test[i] = i % resolution[0];
    }
    fig.newFigure("Fig 2");
    fig.plotRGB(test);
    sleep(2);
    fig.closeFigure();

    return 0;
}