#include<unsupported/Eigen/CXX11/Tensor>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>
#include<functional>

// float sigmoid(float z) {
//     float result;
//     if (z>=45.f)  result = 1.f;
//     else if(z<=-45.f) result = 0.f;
//     else result = 1.f/(1.f+exp(-z));    
//     return result;
// }


template<typename T, int Rank>
Eigen::Tensor<T, Rank> sigmoid_activation(const Eigen::Tensor<T, Rank>& Z) {
    return Z.unaryExpr([](T x) {
        if (x >= T(45)) return T(1);
        if (x <= T(-45)) return T(0);
        return T(1) / (T(1) + std::exp(-x));
    });
}

Eigen::Tensor<float,1> calc_layer (const Eigen::Tensor<float, 1> & input ,const Eigen::Tensor<float,2> &weights, const Eigen::Tensor<float,1> & bias) {
    Eigen::array<Eigen::IndexPair<int>, 1> op_dims = {Eigen::IndexPair<int>(0, 0)};
    auto prod = input.contract (weights,op_dims);
    Eigen::Tensor<float,1> Z = prod+bias;
    auto result = sigmoid_activation(Z);
    return result;
}


int main(int ,char **) {
    // Eigen::Tensor<float,2> A(2,3);
    // A.setRandom();
    // std::cout<<"A:\n\n"<<A<<"\n\n"; 
    // Eigen::Tensor<float,2> B = sigmoid_activation(A);
    // std::cout<<"B:\n\n"<<B<<"\n\n";
    
    std::cout<<"-------------------using calc-layer()---------------"<<std::endl;
    Eigen::Tensor<float,1> X(2);
    X.setValues({-1.5,0.4});

    Eigen::Tensor<float,2> W(2,3);
    W.setValues({{1, 2, 3}, {4, 5, 6}});
    
    Eigen::Tensor<float,1> B(3);
    B.setValues({-1,1,2});

    std::cout<<"X:\n\n"<<X<<"\n\n";
    std::cout<<"W:\n\n"<<W<<"\n\n";
    std::cout<<"B:\n\n"<<B<<"\n\n";

    auto R = calc_layer(X,W,B);
    std::cout<<"R:\n\n"<<R<<"\n\n";
    return 0;
}
