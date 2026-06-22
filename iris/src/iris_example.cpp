#include<unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <tuple>
#include <sstream>
#include<string>
#include<Eigen/Dense>
#include<functional>
#include "../include/book/book.hpp"

using namespace std;

auto load_iris_dataset = [](std::string file_path, bool shuffle = true, float split_percentage = .8)
{
    std::ifstream file;
    file.open(file_path);
    const int N_REGISTERS = 150;

    if (!file.is_open())
        throw std::invalid_argument("File " + file_path + " not found.");

    std::vector<std::string> lines;
    lines.reserve(N_REGISTERS); // it reserves N_REGISTERS before hand to avoide inefficent dynamic memory reallocation
    std::string line;           // but still lines.size() is zero

    while (getline(file, line))
    {
        lines.push_back(line); // pushing each line form file into lines
    }

    if (shuffle) // shuffel is a boolian variable 
    {
        auto rd = std::random_device {}; // fethes truely unpredictable number from os
        auto rng = std::default_random_engine { rd() }; // actually generates random no using seed rd()
        std::shuffle(lines.begin(), lines.end(), rng); //shuffles the vector
    }

    std::vector<float> data;                         // 4 parametesrs as mentioned in book and 3 encoding
    const int EXPECTED_SIZE = N_REGISTERS * (4 + 3); // 4 attributes + 3 hot-encoding for the 3 classes
    data.reserve(EXPECTED_SIZE);                    // reserves EXPECTED_SIZE of size of space beforehand
    std::string element; 
    const std::string class_setosa = "Iris-setosa";           //100
    const std::string class_versicolor = "Iris-versicolor";   //010
    const std::string class_virginica = "Iris-virginica";     //001 

    for (const auto &_line : lines)
    {
        std::stringstream ss(_line);
        while (getline(ss, element, ','))
        {
            if (class_setosa.compare(element) == 0) // .compare(element) returs 0  if both  string are equal
            {
                data.push_back(1.);
                data.push_back(0.);
                data.push_back(0.);
            }
            else if (class_versicolor.compare(element) == 0)
            {
                data.push_back(0.);
                data.push_back(1.);
                data.push_back(0.);
            }
            else if (class_virginica.compare(element) == 0)
            {
                data.push_back(0.);
                data.push_back(0.);
                data.push_back(1.);
            }
            else
            {
                double value = std::stof(element); 
                data.push_back(value);
            }
        }
    }

    if (data.size() != EXPECTED_SIZE)
    {
        throw std::invalid_argument("Wrong dataset size: " + std::to_string(data.size()));
    }

    Eigen::array<int, 2> dims({1, 0}); // array of size 2
    auto tensor_map = Eigen::TensorMap<Eigen::Tensor<float, 2>> (data.data(), 7, 150).shuffle(dims); 
    // data.data() returns raw pointer to data vector      * by default eigen reades this data by column major order (fills column by column)
    // tensorMap says treat the memory at data pointer as a tersor of dimension 7x150 and shuffel its dimensions to 150x7
    // shuffel takes transpose 
    const int split_at = static_cast<int>(N_REGISTERS * split_percentage);

    Eigen::array<Eigen::Index, 2> training_x_offsets = {0, 0};         
    Eigen::array<Eigen::Index, 2> training_x_extents = {split_at, 4}; // 120 x 4 considering the four parametsrs
    Eigen::Tensor<float, 2> training_X_ds = tensor_map.slice(training_x_offsets, training_x_extents);

    Eigen::array<Eigen::Index, 2> training_y_offsets = {0, 4};         // encoding coding part
    Eigen::array<Eigen::Index, 2> training_y_extents = {split_at, 3};  
    Eigen::Tensor<float, 2> training_Y_ds = tensor_map.slice(training_y_offsets, training_y_extents);

    Eigen::array<Eigen::Index, 2> validation_x_offsets = {split_at, 0};       // from 120,0
    Eigen::array<Eigen::Index, 2> validation_x_extents = {N_REGISTERS - split_at, 4}; // 150-120,4
    Eigen::Tensor<float, 2> validation_X_ds = tensor_map.slice(validation_x_offsets, validation_x_extents); //remaining parameter data

    Eigen::array<Eigen::Index, 2> validation_y_offsets = {split_at, 4};  
    Eigen::array<Eigen::Index, 2> validation_y_extents = {N_REGISTERS - split_at, 3};
    Eigen::Tensor<float, 2> validation_Y_ds = tensor_map.slice(validation_y_offsets, validation_y_extents); // rest of the encoding data

    auto result = std::make_tuple(training_X_ds, training_Y_ds, validation_X_ds, validation_Y_ds);

    return result; // returns four tensors of rank 2
};

float accuracy(Eigen::Tensor<float, 2> &REAL, Eigen::Tensor<float, 2> &PRED)
{

    auto compare = [](int a, int b) {
        return static_cast<float>(a == b);
    };

    // gives index of maximum probablity across rows (1D tensor) for both real pred
    Eigen::Tensor<Eigen::DenseIndex, 1> REAL_MAX = REAL.argmax(1); 
    Eigen::Tensor<Eigen::DenseIndex, 1> PRED_MAX = PRED.argmax(1); 

    auto diff = REAL_MAX.binaryExpr(PRED_MAX, compare);

    Eigen::Tensor<float, 0> mean = diff.mean();

    float result = mean(0) * 100.f;

    return result;
}

int main(int, char **)
{

    srand((unsigned int) time(0));
    
    auto [training_X_ds, training_Y_ds, test_X_ds, test_Y_ds] = load_iris_dataset("../data/iris.csv", true);

    auto init_weights = [](int rows, int cols, float range) {
        Eigen::Tensor<float, 2> result(rows, cols);
        result = result.random() * result.constant(range) - result.constant(range / 2.);
        return result;
    };

    auto initial_weights = init_weights(4, 3, 0.1);

    Softmax<2> activation;
    Dense layer(initial_weights, &activation);

    CategoricalCrossEntropy<2> loss_function;

    const int MAX_EPOCHS = 1000;
    const float learning_rate = 0.1;
    int epoch = 0;
    while (epoch++ < MAX_EPOCHS)
    {

        // obtaining the output for the training dataset
        auto output = layer.forward(training_X_ds);

        // calculating the derivative of the cost with respect to the output
        auto dcost_doutput = loss_function.derivative(training_Y_ds, output);

        // calculating layer gradients
        layer.backward(dcost_doutput);

        // in this step, the internal 
        layer.update_state(learning_rate);

        // evaluating training loss
        float loss = loss_function.evaluate(training_Y_ds, output);

        if (epoch == 1 || epoch % 50 == 0) {

            // evaluating validation loss
            auto val_output = layer.predict(test_X_ds);
            float val_loss = loss_function.evaluate(test_Y_ds, val_output);

            float training_acc = accuracy(training_Y_ds, output);
            float validation_acc = accuracy(test_Y_ds, val_output);

            std::cout << "epoch:\t" <<  epoch 
                << "\tloss:\t" << loss << "\tacc:\t" << training_acc 
                << "\tval_loss:\t" << val_loss << "\tval_acc:\t" << validation_acc << "\n";
        }

    }

    return 0;
}