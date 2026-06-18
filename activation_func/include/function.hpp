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

Eigen::Tensor<float, 2> softmax_2D(const Eigen::Tensor<float, 2> &z) {
    auto dimensions = z.dimensions(); 
    int batch_size = dimensions.at(0);
    int instances_size = dimensions.at(1);

    // Getting the maximum for each instance.
    // Note that this operation reduces 1 dimension 
    Eigen::array<int, 1> depth_dim({1});
    auto z_max = z.maximum(depth_dim);

    // Getting the max array as an 2-rank tensor
    Eigen::array<int, 2> reshape_dim({batch_size, 1});
    auto max_reshaped = z_max.reshape(reshape_dim); 
    
    // Broadcasting max 
    Eigen::array<int, 2> bcast({1, instances_size});
    auto max_values = max_reshaped.broadcast(bcast);
    
    // Normalizing the input
    auto normalized = z - max_values;
    
    // calculating softmax
    auto expo = normalized.exp();
    auto expo_sums = expo.sum(depth_dim);
    auto sums_reshaped = expo_sums.reshape(reshape_dim);
    auto sums = sums_reshaped.broadcast(bcast);
    Eigen::Tensor<float, 2> result = expo / sums;
    return result;
}
#endif //FUNCTION_HPP