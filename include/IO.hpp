#include <iostream>
#include <fstream>
#include <algorithm> // std::random_shuffle
#include <vector>    // std::vector
#include <ctime>     // std::time
#include <cstdlib>   // std::rand, std::srand
#include <string>

void Write_to_File(int nrows, int ncols, double * Mat, const char *file_name)
{
    std::ofstream my_file(file_name, std::ios::out | std::ios::binary | std::ios::trunc);
    if (my_file.is_open())
    {
        my_file.write((char *)Mat, nrows * ncols * sizeof(double));
        my_file.close();
    }
    else
        std::cout << "Unable to open file \n";
}

void Read_From_File(long int nrowsncols, double *Mat, const char *file_name, int skip)
{
    std::ifstream my_file(file_name, std::ios::in | std::ios::binary);
    if (my_file.is_open())
    {
        my_file.ignore(skip * sizeof(double));
        my_file.read((char *)Mat, nrowsncols * sizeof(double));
        my_file.close();
        // cout << "Succesfull  read from file \n";
    }
    else
        std::cout << "Unable to open file \n";
}