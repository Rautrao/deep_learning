#include<iostream>
#include<cmath>
#include"../include/dec.hpp"


int main(int,char**) {

    Eigen::Tensor<float,2> TRUE(1,10);
    TRUE.setValues({
        {1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 0.f}
    });
    Eigen::Tensor<float, 2> PRED(1, 10);
    PRED.setValues({
        {.59f, 0.12f, .9f, .77f, 0.f, .95f, 1.f, 0.22f, .55f, 0.123f}
    });

    std::cout << "TRUE: \n\n" << TRUE << "\n\n";
    std::cout << "PRED: \n\n" << PRED << "\n\n";
    std::cout << "BCE: \n\n" << bce(PRED, TRUE) << "\n\n";

    return 0;
}