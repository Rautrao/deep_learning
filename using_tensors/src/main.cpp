#include<unsupported/Eigen/CXX11/Tensor>
#include<Eigen/Dense>
#include<iostream>

int main(int , char **) {
    // //allocating 3-Rank tensor
    // Eigen::Tensor<int,3> my_tensor(2,3,4);

    // //setting up all coefficients to 42
    // my_tensor.setConstant(42);
    // std::cout<<"my_tensor:\n\n"<<my_tensor<<"\n\n";
    // std::cout<<"tensor size is :"<<my_tensor.size()<<"\n\n";
    
    std::cout<<"-------------Using 4D tensor for convolution-------\n\n";

    // Eigen::Tensor<float,4> X(1,6,6,3);
    // X.setRandom();
    // std::cout<<"input dims:\n\n"<<X<<"\n\n";
    // Eigen::Tensor<float,2> K(3,3);
    // K.setRandom();
    // std::cout<<"Kernel dim: \n\n"<<K<<"\n\n";

    // Eigen::Tensor<float,4> out(1,4,4,3);

    // Eigen::array<int,2> dim({1,2});
    // out = X.convolve(K,dim);

    // std::cout<<"output dims:\n\n"<<out<<"\n\n";

    std::cout<<"-------------Implementing Strides------------\n\n";

    // Eigen::Tensor<int,2> A(4,5);
    // A.setValues({
    //     {1, 2, -1, 0, 1},
    //     {4, 1, 2, -1, 2},
    //     {3, 1, -1, 1, -2},
    //     {-2, 1, 5, 4, 0}
    // });

    // Eigen::array<Eigen::DenseIndex,2>strides({3,2});
    // Eigen::Tensor<int,2> B = A.stride(strides);

    // std::cout<<"A is : \n\n"<<A.dimensions()<<"\n\n"<<A<<"\n\n";
    // std::cout<<"B is : \n\n"<<B.dimensions()<<"\n\n"<<B<<"\n\n";

    std::cout<<"-------------Padding using Eigen Tensor API----------\n\n";

    Eigen::Tensor<int,2> A(2,3);
    A.setValues({{0, 100, 200}, {300, 400, 500}});
    Eigen::array<std::pair<int,int>,2> padding;
    padding[0] = std::make_pair(0,1); // 0 rows above top 1 row below bottom
    padding[1] = std::make_pair(2,3); // 2 rows to left and 3 rows to right
    
    Eigen::Tensor<int,2> B = A.pad(padding);
    std::cout<<"A is \n\n"<<A.dimensions()<<"\n\n"<<A<<"\n\n";
    std::cout<<"B is \n\n"<<B.dimensions()<<"\n\n"<<B<<"\n\n";

    return 0;

}