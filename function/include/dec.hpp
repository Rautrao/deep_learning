#ifndef DEC_HPP
#define DEC_HPP

#include<unsupported/Eigen/CXX11/Tensor>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>
#include<functional>

template<typename T,int _RANK>
T mse (const Eigen::Tensor<T,_RANK> &PRED,const Eigen::Tensor<T,_RANK>& TRUE) {
    auto diff = TRUE-PRED;
    auto pow = diff.pow(2);
    T sum = ((Eigen::Tensor<T,0>)(pow.sum()))(0); // sums individual value form the pow : e.g [91] (0) -> 91
    T result = sum/PRED.size();
    return result;
}

template <typename T, int _RANK>
T bce(const Eigen::Tensor<T, _RANK> &PRED, const Eigen::Tensor<T, _RANK> &TRUE) {
    // 1. Create the inverse truth labels (the toggle switch)
    auto COMP_TRUE = TRUE.constant(1.) - TRUE;

    // 2. Create the inverse predictions (what the model thinks the other class is)
    auto COMP_PRED = PRED.constant(1.) - PRED;

    // 3. The "Numerical Stability" Hack (1e-7)
    // We add a tiny value (epsilon) to the prediction before taking the log.
    // If PRED is 0, log(0) is negative infinity, which breaks the program.
    // 1e-7 ensures we never take the log of zero.
    auto epsilon = PRED.constant(1e-7);
    
    auto part1 = TRUE * (PRED + epsilon).log();
    auto part2 = COMP_TRUE * (COMP_PRED + epsilon).log();

    // 4. Combine the parts to get the loss
    auto parts = part1 + part2;

    // 5. Calculate the average
    // .sum() reduces the tensor to a single scalar value
    float sum = ((Eigen::Tensor<T, 0>)(parts.sum()))(0);
    
    // We multiply by -1 because the sum of logs of decimals is negative, 
    // and we want a positive "Loss" score.
    float result = -sum / PRED.size();
    
    return result;
}   


#endif //DEC_HPP

