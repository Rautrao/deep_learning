#include<cmath>
#include<vector>
#include<stdexcept>
#include<cstdlib>
#include<algorithm>
#include<chrono>
#include<random>
#include "../include/my_functions.hpp"

float sigmoid(float z) {
    float result;
    if (z>=45.f)  result = 1.f;
    else if(z<=-45.f) result = 0.f;
    else result = 1.f/(1.f+exp(-z));    
    return result;
}
// Using EXPECT_EQ and EXPECT_THROW
int find_middle(const std::vector<int> & arr) {
    const size_t N = arr.size();    
    if (N==0) {
        throw std::invalid_argument("cannot find the middle of the empty vector");
    }
    std::vector<int> copy = arr;
    std::sort(copy.begin(),copy.end());
    int result = copy[N/2];
    return result;
}
std::vector<float> glorot_initializer(int fan_in,int fan_out) {
    int size = fan_in*fan_out;
    float std_dev = sqrt(2./(fan_in+fan_out));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(.0,std_dev);
    std::vector<float> result(size);
    std::generate(result.begin(),result.end(),[&generator,&distribution] () {
        return distribution(generator);
    });
    return result;
}