#include<iostream>
#include<random>
#include<limits>
#include<cstdlib>
#include<algorithm>
#include"../include/function.hpp"

int main(int, char **) {

    Eigen::Tensor<float, 2> input(8, 3);
    input.setValues({
        {0.1, 1., -2.},{10., 2., 5.},{5., -5., 0.},{2., 3., 2.}
        ,{100., 1000., -500.},{3., 3., 3.},{-1, 1., -1.},{-11., -0.2, -.1}
    });
    const int batch_size = input.dimension(0);
    const int output_size = input.dimension(1);
    Eigen::array<Eigen::Index, 2> extent = {1, output_size}; // array of size 2 : 1 x output_size
    Eigen::array<int, 1> reshape_dim({output_size}); // array of size 1 : output_size
    for (int i = 0; i < batch_size; i++) {
        Eigen::array<Eigen::Index, 2> offset = {i, 0};  // saves current row index and column 0
        Eigen::Tensor<float, 1> row = input.slice(offset, extent).reshape(reshape_dim); // gets ith row and convert it to rank 1 tensor 
        Eigen::Tensor<float, 1> output = softmax(row); // applies softmax on each element in the row
        std::cout << "softmax([" << row << "]): [" << output << "]\n";
    }
    return 0;
}