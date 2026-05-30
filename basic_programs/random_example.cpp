#include<cmath>
#include<iomanip>
#include<iostream>
#include<map>
#include<random>
#include<string>

std::random_device rd{}; // sets seed for generating random variable

int main(int ,char **) {
	std::mt19937 random_generator {rd()};
	//std::normal_distribution<> distro {2,3}; // for normal distributoin mean=2, variance= 3
	std::uniform_real_distribution<> distro {-10, 13};  //for uniform distribution mean 
	std::map<int,int> histogram;
	
	for (int n=0;n<10'000;++n) {
		int val = std::round(distro(random_generator));
		histogram[val]++;
	}
	for (auto[x,y] : histogram ) {
		std::cout<<std::setw(2)<<x<<""<<std::string(y/20,'*')<<"\n";
	}
	return 0;
}
