#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include<unsupported/Eigen/CXX11/Tensor>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>
#include<functional>

#define SIGMOID_THRESHOLD 45
 
template <typename T>
T sigmoid(T z) {
    T result;
    if (z >= T(SIGMOID_THRESHOLD)) result = 1.;
    else if (z <= - T(SIGMOID_THRESHOLD)) result = 0.;
    else result = T(1.) / (T(1.) + std::exp(-z));
    return result;
}

Eigen::Tensor<float, 1> softmax(const Eigen::Tensor<float, 1> &z) {
    const Eigen::Tensor<float, 0> m = z.maximum();    
    auto normalized = z - z.constant(m(0));
    auto expo = normalized.exp(); 
    const Eigen::Tensor<float, 0> expo_sums = expo.sum();
    Eigen::Tensor<float, 1> result = expo / expo.constant(expo_sums(0));
    return result;
}

#endif //FUNCTION_HPP