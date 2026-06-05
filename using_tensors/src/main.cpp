#include<unsupported/Eigen/CXX11/Tensor>
#include<iostream>

int main(int , char **) {
    //allocating 3-Rank tensor
    Eigen::Tensor<int,3> my_tensor(2,3,4);

    //setting up all coefficients to 42
    my_tensor.setConstant(42);
    std::cout<<"my_tensor:\n\n"<<my_tensor<<"\n\n";
    std::cout<<"tensor size is :"<<my_tensor.size()<<"\n\n";
    return 0;
}