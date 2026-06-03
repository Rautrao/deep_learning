#include<iostream>
#include<Eigen/Dense>
#include<functional>
#include<tuple>

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
// more efficent code for the above padded version

auto Padded_Conv2D = [](const Eigen::MatrixXd &X, const Eigen::MatrixXd &K,const int padding){
    const int X_rows = X.rows();
    const int X_cols = X.cols();
    const int K_rows = K.rows();
    const int K_cols = K.cols();
    if (X_rows<K_rows) throw std::invalid_argument("The input has less rows than the kernel");
    if (X_cols<K_cols) throw std::invalid_argument("The input has less columns than the kernel");

    const int rows = X_rows - K_rows + 2*padding + 1;
    const int cols = X_cols-K_cols + 2*padding+1;
    
    auto fitting_dims = [&padding] (int pos,int k,int length) {
        int input = pos-padding; //  converting the padded matrix index to original matrix index
        int kernel = 0;         //  starting index of the kernel
        int size = k;           // size of the kernel  
        if (input<0) {           
            kernel= -input;
            size+=input;
            input = 0;
        }
        if (input+size>length) {
            size = length - input;
        }
        return std::make_tuple(input, kernel, size);
    };

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(rows,cols);
    for (int i=0;i<rows;i++) {
        const auto [input_i,kernel_i,size_i] = fitting_dims(i,K_rows,X_rows);
        for (int j=0;size_i>0 && j<cols;++j) {
            const auto [input_j, kernel_j, size_j ] = fitting_dims(j,K_cols,X_cols);
            if(size_j>0) {
                auto input_roi = X.block(input_i,input_j,size_i,size_j) ;
                auto in_kernel = K.block(kernel_i,kernel_j,size_i,size_j);
                res(i,j) = input_roi.cwiseProduct(in_kernel).sum();
            }
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
    auto optimized_out = Padded_Conv2D(X,K,1);
    std::cout<<"------------------\n";
    std::cout<<"Using optimized verion of dot product"<<"\n";
    std::cout<<"Convolution::\n"<<optimized_out<<"\n";

    return 0;
}