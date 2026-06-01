#include<cmath>
#include<vector>
#include<stdexcept>
#include<cstdlib>
#include<algorithm>

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