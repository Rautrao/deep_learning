#include<iostream>
#include<random>
#include<limits>
#include<cstdlib>
#include<algorithm>
#include"../include/dec.hpp"

int main(int, char **) {
    int seed = 1234;
    std::mt19937 rng(seed);
    // generates some dataset Y for some set X
    auto synthetic_generator = [&rng](int size,float range,float a,float b,float noise) {
        Eigen::Tensor<float,2> X(1,size);
        X = X.random()*range;
        Eigen::Tensor<float,2> Y(1,size);
        Y.setConstant(b);
        std::normal_distribution<float> normal_distro(0,noise);
                                                        //interceptor is nothing but : b
        auto random_gen = [&rng, &normal_distro, a] (float interceptor,float x) {
            return interceptor + normal_distro(rng) + x*a;
                    //  y = ax+ b here b = interceptor+noise
        };
        Y = Y.binaryExpr(X,random_gen);
        return std::make_pair(X,Y);
    };
    int size = 30;
    float range=10.f;
    float slope_a=2.f;
    float intercept_b=3.f;
    float noise=2.f;

    auto [X, Y] = synthetic_generator(size, range, slope_a, intercept_b, noise);
    float Cx = ((Eigen::Tensor<float, 0>)(X.mean()))(0);
    float Cy = ((Eigen::Tensor<float, 0>)(Y.mean()))(0);
    
    // here we are passing a 
    auto cost_calc = [&Cx, &Cy](float a, auto X, auto Y) {
        auto S = [&a,&Cx,&Cy] (float x) {
            return a*(x-Cx)+Cy; // y = a*(x-x1)+y1
        };                      
        Eigen::Tensor<float,2> pred = X.unaryExpr(S); // it will generate a pred tensor of rank 2 with predicted values
        return mse(pred,Y);
    };

    float a = -4.f; // initial slope of line
    const float step = 0.01f;  
    float best_a = a; 
    float best_cost = std::numeric_limits<float>::max();
    const int MAX_EPOCHES = 70;
    int epoch = 0;

    while(epoch++ < MAX_EPOCHES) { 
        float cost_init = cost_calc(a,X,Y);  // X and Y are constant data sets
        if (cost_init < best_cost) {
            best_cost = cost_init;
            best_a = a;
        }
        //stopping if cost is too small
        if (cost_init<5.f) break;
        float cost_end = cost_calc(a+step,X,Y); // increasing the slope a by small amount
        float grad = (cost_end-cost_init)/step; 
        std::cout << "epoch:\t" << epoch << "\ta:\t" << a << "\tMSE:\t" << cost_init << "\tgrad:\t" << grad << "\n";

        //stopping if grad is too small
        if (abs(grad)<10e-5) break;
        a = a-step*grad;  
        // a = a - step * grad
        // a = -4 - step * (negative number)  -> a decreases
        // a = -4 - step * (positive number)  -> a increases
    }
    const float best_b = Cy-best_a*Cx;
    std::cout << "Best configuration is (a, b) = (" << best_a << ", " << best_b << ")" << " with cost " << best_cost << "\n";
    return 0;
}