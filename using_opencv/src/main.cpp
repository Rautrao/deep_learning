#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<tuple>

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


int main() {
    //demo
    // cv::Mat source = cv::imread("./using_opencv/src/led.jpg",cv::IMREAD_COLOR);
    // cv::Mat rotated, grey;
    // cv::rotate(source,rotated,cv::ROTATE_90_CLOCKWISE);
    // cv::cvtColor(rotated,grey,cv::COLOR_BGR2GRAY);

    // cv::imshow("source",source);
    // cv::imshow("rotated",rotated);
    // cv::imshow("gray",grey);
    // cv::imwrite("leds_rot.jpg",grey);
    // cv::waitKey();

    Eigen::MatrixXd G_x(3,3),G_y(3,3);
    //This code uses two famous kernels called Sobel Gx and Gy
    G_x<< 
    1., 0., -1.,
    2., 0., -2.,
    1., 0., -1.;


    G_y << 
    1.,   2., 1.,
    0.,   0., 0.,
    -1., -2., -1.;

    cv::Mat src = cv::imread("./using_opencv/src/chess.jpg",cv::IMREAD_COLOR); 
    cv::Mat img; 
    cv::cvtColor(src,img,cv::COLOR_BGR2GRAY);
    Eigen::MatrixXd X;
    cv::cv2eigen(img,X);
    const int padding = (G_x.rows()-1)/2;
    auto G_x_conv = Padded_Conv2D(X,G_x,padding);
    auto G_y_conv = Padded_Conv2D(X,G_y,padding);

    cv::Mat temp,G_x_res,G_y_res;
    cv::eigen2cv(G_x_conv,temp);
    temp.convertTo(G_x_res,CV_8UC1);

    cv::eigen2cv(G_y_conv,temp);
    temp.convertTo(G_y_res,CV_8UC1);

    cv::imshow("source",src);
    cv::imshow("gray scale",img);
    cv::imshow("applying Gx",G_x_res);
    cv::imshow("applying Gy",G_y_res);

    cv::waitKey();

    return 0;
}
