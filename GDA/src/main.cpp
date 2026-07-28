#include <iostream>
#include <chrono>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../include/image_io.hpp"

using namespace std::chrono;

auto BatchedConvolution = [](const Eigen::Tensor<float,3> &A, const Eigen::Tensor<float,3> &B)
{

    const int pad_set = 1;
    Eigen::array<std::pair<int, int>, 3> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(pad_set, pad_set);
    padding[2] = std::make_pair(pad_set, pad_set);

    const int batch_size = A.dimension(0);
    const int dim1 = A.dimension(1);
    const int dim2 = A.dimension(2);
    Eigen::Tensor<float,3> output(batch_size, dim1, dim2);

    auto padded = A.pad(padding);
    Eigen::array<Eigen::DenseIndex, 3> conv_dims({0, 1, 2});
    output = padded.convolve(B, conv_dims);
    
    return output;
};

auto backward = [](const Eigen::Tensor<float,3> &X, Eigen::Tensor<float,3> &T, Eigen::Tensor<float,3> &Y)
{
    auto DIFF = Y - T;
    Eigen::Tensor<float,3> batch = BatchedConvolution(X, DIFF);

    Eigen::array<Eigen::DenseIndex, 2> two_dims{{batch.dimension(1), batch.dimension(2)}};
    Eigen::Tensor<float,2> result = batch.reshape(two_dims);
    result = result * result.constant(2.f / X.size());

    return result;
};

auto convolution2D = [](const Eigen::Tensor<float,3> &input, const Eigen::Tensor<float,2> &kernel)
{

    Eigen::array<std::pair<int, int>, 3> padding;
    padding[0] = std::make_pair(0, 0); 
    padding[1] = std::make_pair(1, 1); // padding  = (kernel-1)/2 for same dimensions as input
    padding[2] = std::make_pair(1, 1);

    auto padded = input.pad(padding); 
    Eigen::array<Eigen::DenseIndex, 2> dims({1, 2});
    Eigen::Tensor<float,3> result  = padded.convolve(kernel, dims);
    return result;
};

auto forward = [](const Eigen::Tensor<float,3> &X, const Eigen::Tensor<float,2> &kernel)
{
    return convolution2D(X, kernel);
};

auto MSE = [](const Eigen::Tensor<float,3> &T, const Eigen::Tensor<float,3> &Y)
{

    auto diff = T - Y;
    auto quadratic = diff * diff;
    Eigen::Tensor<float,0> sum = quadratic.sum();
    float result = sum(0) / Y.size();
    return result;
};

int main(int argc, char**) {

    Eigen::Tensor<float,2> Generator_Kernel(3, 3);
    Generator_Kernel.setValues({{1., 0., -1.}, {2., 0., -2.},{1., 0., -1.}});

    auto [X, T] = load_dataset("GDA/images", Generator_Kernel, 160);

    if (argc > 1) {

        const int batch_size = X.dimension(0);
        const int image_size = X.dimension(1);
        const Eigen::array<Eigen::DenseIndex, 3> image_dim = {image_size, image_size, 1};

        for (int i = 0; i < batch_size; ++i) {
            cv::Mat X_output, T_output;

            Eigen::Tensor<float,3> x = X.chip<0>(i).reshape(image_dim); // getting single image and reshaping it to rank 3 tensor
            cv::eigen2cv(x, X_output);
            cv::imshow("x", X_output);

            Eigen::Tensor<float,3> t = T.chip<0>(i).reshape(image_dim);
            Eigen::Tensor<float,0> _max = t.maximum();
            Eigen::Tensor<float,3> normalization = t / t.constant(_max(0));
            cv::eigen2cv(normalization, T_output);
            cv::imshow("t", T_output);

            cv::waitKey();
        }

        cv::destroyAllWindows();
    }

    Eigen::Tensor<float,2> kernel(3, 3);
    kernel = kernel.random();

    const int MAX_EPOCHS = 500;
    const double learning_rate = 0.1;

    int epoch = 0;
    while (epoch < MAX_EPOCHS)
    {

        auto begin = high_resolution_clock::now();
        auto output = forward(X, kernel);

        auto grad = backward(X, T, output);

        double loss = MSE(T, output);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);

        auto update = grad * grad.constant(learning_rate);
        kernel -= update;

        if (epoch % 100 == 0) {
            std::cout << "epoch:\t" << epoch << "\ttook:\t" << duration.count() << " mills\t" << "\tloss:\t" << loss << "\n";
        }

        epoch++;
    }

    std::cout << "\nGenerative kernel is:\n\n" << std::fixed << std::setprecision(2) << Generator_Kernel << "\n\n";
    std::cout << "\nTrained kernel is:\n\n" << std::fixed << std::setprecision(2) << kernel << "\n\n";

    return 0;
}