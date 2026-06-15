#include<iostream>
#include<random>
#include<limits>
#include<cstdlib>
#include<algorithm>
#include"../include/dec.hpp"

int main(int, char **) {
    int seed = 1234;
    std::mt19937 rng(seed);
    auto synthetic_generator = [&rng](int size,float range,float a,float b,float noise) {
        Eigen::Tensor<float,2> X(1,size);
        X = X.random()*range;
        Eigen::Tensor<float,2> Y(1,size);
        Y.setConstant(b);
        std::normal_distribution<float> normal_distro(0,noise);
                                                        //interceptor is nothing but : b
        auto random_gen = [&rng, &normal_distro, a] (float interceptor,float x) {
            return interceptor + normal_distro(rng) + x*a;
        };
        Y = Y.binaryExpr(X,random_gen);
        return std::make_pair(X,Y);
    };
    // slope = 2.f , bias = 3.f , here 30 is size of dataset X,Y
    auto [X,Y] = synthetic_generator(30,10.f,2.f,3.f,2.f);
    
    float Cx = ((Eigen::Tensor<float, 0>)(X.mean()))(0);
    float Cy = ((Eigen::Tensor<float, 0>)(Y.mean()))(0);
    float best_cost = std::numeric_limits<float>::max();
    const float step = 0.01;
    float a = -4; //random slope
    float best_a = a; 
    auto S = [&a,&Cx,&Cy](float x) {
        return a*(x-Cx)+(Cy); // y-y1 = a(x-x1) here x1 and y1 are Cx and Cy
    };
    int epochs = 0;

    for (;a<4;a+=step) { //starting with some random slope
        Eigen::Tensor<float,2> pred = X.unaryExpr(S);
        float cost = mse(pred,Y);
        if (cost<best_cost) {
            best_cost = cost;
            best_a = a;
        }
        epochs++;
    }
    // 
    const float best_b = Cy - best_a*Cx; // putting Cy,Cx in the equation : y = ax+b here we are calculating b 
    // we already line passes through (Cy,Cx) and slope is best_a
    std::cout<<"Best Configuration is (a,b)=("<<best_a<<","<<best_b<<")"<<" with cost "<<best_cost<<"\n";
    std::cout<<"Took "<<epochs<<" epochs.\n";
    return 0;
}