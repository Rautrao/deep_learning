#include<cmath>
#include<iomanip>
#include<iostream>
#include<map>
#include<random>
#include<string>

std::random_device rd{};

int main(int,char **) {
	const unsigned int seed = rd(); 
	std::mt19937 random_generator{seed}; // we can control the randomness by setting custom seed 
	
	std::uniform_real_distribution<> distro{-10,13};
	std::uniform_real_distribution<> distro1{-5,8};
	std::uniform_real_distribution<> distro2{3,9};
	std::uniform_real_distribution<> distro3{-8,-1};
	std::uniform_real_distribution<> distro4{-10,10};
	std::map<int ,int> histogram;
	
	for (int i=0;i<10'000;i++) {
		double val = distro(random_generator);
		val += distro1(random_generator);
		val += distro2(random_generator);
		val += distro3(random_generator);
		val += distro4(random_generator);
		val/=5;
		histogram[std::round(val)]++;
	}
	for (auto[x,y] : histogram) {
		std::cout<<std::setw(2)<<x<<""<<std::string(y/20,'*')<<"\n";
	}
	
	return 0;	
}
