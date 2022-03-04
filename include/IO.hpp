#include <iostream>
#include <fstream>
#include <algorithm> // std::random_shuffle
#include <vector>    // std::vector
#include <ctime>     // std::time
#include <cstdlib>   // std::rand, std::srand
#include <string>

template <typename T>
void Write_to_File(size_t nrowsncols, T *Mat, const char *file_name)
{
    std::ofstream my_file(file_name, std::ios::out | std::ios::binary | std::ios::trunc);
    if (my_file.is_open())
    {
        my_file.write((char *)Mat, nrowsncols * sizeof(double));
        my_file.close();
    }
    else
        std::cout << "Unable to open file \n";
}

template <typename T>
void Read_From_File(size_t nrowsncols, T *Mat, const char *file_name, size_t skip)
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