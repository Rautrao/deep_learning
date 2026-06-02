#include<vector>

#ifndef MY_FUNCTIONS_HPP
#define MY_FUNCTIONS_HPP

float sigmoid(float z);
int find_middle(const std::vector<int> &);
std::vector<float> glorot_initializer(int fan_in,int fan_out);
#endif