#include<iostream>
#include<Eigen/Dense>

int main(int,char **) {
	Eigen::MatrixXd A(2,2);
	A(0,0) = 2.;
	A(1,0) = -2.;
	A(0,1) = 3.;
	A(1,1) = 1.;
	
	Eigen::MatrixXd B(2,3) ;
	B(0,0) = 1.;
	B(1,0) = 1.;
	B(0,1) = 2.;
	B(1,1) = 2.;
	B(0,2) = -1.;
	B(1,2) = 1.;
	
	auto C = A*B ;
	
	std::cout<<"A:\n"<<A<<std::endl;
	std::cout<<"B:\n"<<B<<std::endl;
	std::cout<<"C:\n"<<C<<std::endl;
	auto D = B.cwiseProduct(C) ;
	std::cout<<"coefficient-wise multiplication of B & C is :\n"<<D<<std::endl;
	std::cout<<"The inverse of A is :\n"<<A.inverse()<<std::endl;
	std::cout<<"The determinant of A is :\n"<<A.determinant()<<std::endl;
	std::cout<<"The transpose of B is :\n"<<B.transpose()<<std::endl;
	
	//Unary and binary coefficent-wise operations
	
	std::cout<<"Example of unary operation :\n\n";
	auto func_X_X = [](double x) {return x*x;};
	std::cout<<A.unaryExpr(func_X_X)<<std::endl;
	
	std::cout<<"\nExample of binary operation:\n\n";
	auto func_X_Y = [](double x,double y) {return x*y;};
	std::cout<<B.binaryExpr(C,func_X_Y)<<std::endl;
	
	
	
	
	return 0;
}
