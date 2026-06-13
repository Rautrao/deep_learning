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

#endif //DEC_HPP

