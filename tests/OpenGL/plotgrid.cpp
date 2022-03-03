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
    Read_From_File(prod_dims, &_data[0], filename.c_str(), 0);

    fig.plotRGB(_data);

    return 0;
}