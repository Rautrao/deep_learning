#include<gtest/gtest.h> // this will be automatically downloaded when we run project 
#include"../include/my_functions.hpp"

TEST(SigmoidTest,BaseCase) {
    float actual = sigmoid(-100);
    float expected = 0 ;
    EXPECT_NEAR(actual,expected,1e-7f) <<"Sigmoid lower bound test failed";
    EXPECT_NEAR(sigmoid(100.),1.,1e-7f) <<"Sigmoid upper bound test failed";
    EXPECT_NEAR(sigmoid(.0),.5,1e-7f) <<"Sigmoid ceneter position test failed";
}

TEST(OddCase,BaseCases) {
    std::vector<int> vec {9,4,1,11,20,-1,2};
    int actual = find_middle(vec);
    EXPECT_EQ(4,actual);
}

TEST(EvenCase, BaseCases) {
    std::vector<int> vec {9,4,1,11,20,-1,2};
    int actual = find_middle(vec);
    EXPECT_EQ(4,actual);
}

TEST(CheckEmptyVector,BaseCases) {
    EXPECT_THROW ({
        std::vector<int> vec;//empty vector
        find_middle(vec);
    },std::invalid_argument);

}