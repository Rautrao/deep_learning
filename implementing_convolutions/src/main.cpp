#include<iostream>
#include<Eigen/Dense>
#include<functional>
auto Conv_2D = [](const Eigen::MatrixXd &X,const Eigen::MatrixXd &K) {
    const int k_rows = K.rows();
    const int k_cols = K.cols();
    const int rows = (X.rows()-k_rows+1);
    const int cols = (X.cols()-k_cols+1);

    Eigen::MatrixXd res = Eigen::MatrixXd(rows,cols);

    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            double sum = X.block(i,j,k_rows,k_cols).cwiseProduct(K).sum();
            res(i,j) = sum;
        }
    }
    return res;
};

auto native_Padded_Conv_2D = [](const Eigen::MatrixXd &X,const Eigen::MatrixXd &K,int padding) {
    const int k_rows = K.rows();
    const int k_cols = K.cols();
    const int rows = (X.rows()-k_rows+1+2*padding);
    const int cols = (X.cols()-k_cols+1+2*padding);
    Eigen::MatrixXd pad = Eigen::MatrixXd::Zero(X.rows()+2*padding, 2*padding+X.cols());
    pad.block(padding,padding,X.rows(),X.cols()) = X;
    Eigen::MatrixXd res = Eigen::MatrixXd(rows,cols);

    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            double sum = pad.block(i,j,k_rows,k_cols).cwiseProduct(K).sum();
            res(i,j) = sum;
        }
    }
    return res;
};
int main(int , char **) {
    Eigen::MatrixXd K(3,3);
    K<< -1,0,1,
        -1,0,1,
        -1,0,1;
    Eigen::MatrixXd X(6,6);
    X<<  3, 1, 0, 2, 5, 6,
         4, 2, 1, 1, 4, 7,
         5, 4, 0, 0, 1, 2,
         1, 2, 2, 1, 3, 4,
         6, 3, 1, 0, 5, 2,
         3, 1, 0, 1, 3, 3;

    auto out = Conv_2D(X,K);
    std::cout<<"Kernel:\n" <<K<<"\n\n";
    std::cout<<"Input:\n" <<X<<"\n\n";
    std::cout<<"Convolution:\n"<<out<<"\n";

    return 0;
}