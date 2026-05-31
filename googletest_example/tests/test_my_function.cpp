#include<gtest/gtest.h>
#include"../include/my_functions.hpp"

TEST(SigmoidTest,BaseCase) {
    float actual = sigmoid(-100);
    float expected = 0 ;
    EXPECT_NEAR(actual,expected,1e-7f) <<"Sigmoid lower bound test failed";
    EXPECT_NEAR(sigmoid(100.),1.,1e-7f) <<"Sigmoid upper bound test failed";
    EXPECT_NEAR(sigmoid(.0),.5,1e-7f) <<"Sigmoid ceneter position test failed";
}

