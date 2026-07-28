#ifndef IMAGE_IO_HPP
#define IMAGE_IO_HPP

#include <filesystem>
#include <stdexcept>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace fs = std::filesystem;

cv::Mat resize_image(const cv::Mat &image, int target_rows, int target_cols)
{
    const int image_rows = image.rows;
    const int image_cols = image.cols;

    int new_rows = 0;
    int new_cols = 0;

    if (image_rows > image_cols) { // image rows is limiting factor so we use it for scaling the image to fit 
        new_rows = target_rows;    // target rows x target cols but its similar to pooling but not exactly 
        new_cols = image_cols * target_rows / image_rows; // image details perhaps, reduces.
    } else {
        new_cols = target_cols;
        new_rows = image_rows * target_cols / image_cols;
    }
    cv::Mat resized;
    resize(image, resized, cv::Size(new_cols, new_rows), cv::INTER_LINEAR); // process similar to pooling for scaling the image
                                                                            // aspect ratio remains same
    cv::Mat result = cv::Mat::zeros(cv::Size(target_cols, target_rows), CV_8UC1);

    resized.copyTo(result(cv::Rect((target_cols - new_cols)/2, (target_rows - new_rows)/2, resized.cols, resized.rows)));
    // coping the resized image in center of the result which is target rows x target cols frame
    return result;
}

auto load_dataset(const std::string &data_folder, const Eigen::Tensor<float,2> &Gen, const int image_size) {

    std::vector<cv::Mat> images;

    for (const auto & entry : fs::directory_iterator(data_folder)) { // fs::directory_iterator lets us loop over the contents of directory
        if (!entry.is_regular_file()) {
            continue;
        }

        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            images.push_back(image);
        }
    }

    if (images.empty()) {
        throw std::runtime_error("No readable images found in " + data_folder);
    }

    Eigen::Tensor<float,3> X(images.size(), image_size, image_size);
    Eigen::Tensor<float,3> T(images.size(), image_size, image_size);

    const Eigen::array<Eigen::DenseIndex, 3> extent = {1, image_size, image_size};

    Eigen::array<std::pair<int, int>, 3> padding;
    padding[0] = std::make_pair(1, 1); // padding_needed_per_side = (kernel_size - 1) / 2 to preserve dimensions 
    padding[1] = std::make_pair(1, 1); // we are using sobel's kernel : 3x3 
    padding[2] = std::make_pair(0, 0); 

    for (Eigen::DenseIndex i = 0; i < static_cast<Eigen::DenseIndex>(images.size()); ++i) {
        const cv::Mat &image = images[static_cast<std::size_t>(i)];
        cv::Mat formatted_image = resize_image(image, image_size, image_size); // as discussed just above
        cv::Mat frame32f;
        formatted_image.convertTo(frame32f, CV_32F); // uchar [0,255] → float [0,255]
        frame32f /= 255.f; // // float [0,255] → float [0.0, 1.0]
        
        Eigen::Tensor<float,3> eigen_frame(image_size, image_size, 1);
        cv::cv2eigen(frame32f, eigen_frame);

        Eigen::Tensor<float,3> convolved(image_size, image_size, 1);
        Eigen::array<int, 2> dims({0, 1});
        convolved = eigen_frame.pad(padding).convolve(Gen, dims);
        // slide the kernel along dimension 0 and dimension 1

        Eigen::array<Eigen::DenseIndex, 3> offset = {i, 0, 0};
        Eigen::array<Eigen::DenseIndex, 3> new_dim({image_size, image_size, 1});
        X.slice(offset, extent) = eigen_frame.reshape(extent);
        T.slice(offset, extent) = convolved.reshape(extent);

    }

    return std::make_tuple(X, T);

};

#endif //IMAGE_IO_HPP
