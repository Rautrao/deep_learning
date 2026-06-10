#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <Eigen/Core>
#include <functional>
#include<random>

template <typename T, int _RANK>
class Dense {
public:
    Dense(Eigen::Tensor<T, 2> weights,
          Eigen::Tensor<T, 1> bias,
          std::function<Eigen::Tensor<T, 2>(const Eigen::Tensor<T, 2>&)> activation)
        : weights(std::move(weights)),
          bias(std::move(bias)),
          activation(activation) {}

    virtual ~Dense() {}

    Eigen::Tensor<T, _RANK> evaluate(const Eigen::Tensor<T, _RANK>& input) {
        const auto input_dims = input.dimensions();
        const auto weight_dims = weights.dimensions();
        
        int instance_size = input_dims[_RANK - 1]; // last dimension size
        int output_size = weight_dims[1]; // output size is no of 

        // perform some runtime checks
        if (instance_size != weight_dims[0]) 
            throw std::invalid_argument("Input size does not match weight matrix shape");

        if (this->bias.dimension(0) != output_size)
            throw std::invalid_argument("Bias size does not match weight matrix shape");
        
        //input.size() returns total no of elements in the tensor : l x b x h
        int instances = input.size() / instance_size;

        // Reshape input to 2D
        Eigen::array<long long, 2> new_dim({instances, instance_size});
        Eigen::Tensor<T, 2> reshaped = input.reshape(new_dim);

        // Matrix multiplication (contract)
        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
        Eigen::Tensor<T, 2> prod = reshaped.contract(weights, contract_dims);

        // broadcasting the bias to match the output dimensions
        Eigen::array<long long, 2> bias_new_dim({1, output_size});
        auto bias_reshaped = this->bias.reshape(bias_new_dim);
        Eigen::array<long long, 2> bias_bcast({instances, 1});
        Eigen::Tensor<T, 2> bias_broadcast = bias_reshaped.broadcast(bias_bcast);

        // Add bias
        auto Z = prod + bias_broadcast;

        // Apply activation
        auto output = activation(Z);

        // reshaping the output to the proper RANK
        auto result_dims = input_dims;
        result_dims[_RANK - 1] = output_size;
        auto result = output.reshape(result_dims);

        return result;
    }
    virtual Eigen::Tensor<T,_RANK> operator() (const Eigen::Tensor<T,_RANK> &input) {
        return this->evaluate(input);
    }
private:
    Eigen::Tensor<T, 2> weights;
    Eigen::Tensor<T, 1> bias;
    std::function<Eigen::Tensor<T, 2>(const Eigen::Tensor<T, 2>&)> activation;
};

template<typename T, int Rank>
Eigen::Tensor<T, Rank> sigmoid_activation(const Eigen::Tensor<T, Rank>& Z) {
    return Z.unaryExpr([](T x) {
                if (x >= T(45)) return T(1);
                if (x <= T(-45)) return T(0);
                return T(1) / (T(1) + std::exp(-x));
           });
}

template <typename T, int _RANK> 
auto flatten(const Eigen::Tensor<T, _RANK> &input) {
    const auto input_dims = input.dimensions();
    const int batch_size = input_dims[0];
    int instance_size = input.size() / batch_size;
    Eigen::array<long long int, 2> new_dim({batch_size, instance_size});
    Eigen::Tensor<T, 2> result = input.reshape(new_dim);
    return result;
}

int main() {
    srand((unsigned int) time(0)); // seeds to some random number

    // Bias initializer: zeros
    auto bias_initializer = [](const int size) {
        Eigen::Tensor<float, 1> result(size);    // 1 Rank tensor
        result.setZero();    // initializing to zero
        return result;
    };

    // Weight initializer: small random values
    auto weight_initializer = [](const int rows, const int cols, float range = 0.05f) {
        Eigen::Tensor<float, 2> _random(rows, cols);   // 2D tensor
        _random.setRandom();  
        Eigen::Tensor<float, 2> result = (_random - _random.constant(0.5f))* _random.constant(range);
        return result;
    };

    // Define layers
    Dense<float, 2> layer1( weight_initializer(4, 6), bias_initializer(6), sigmoid_activation<float, 2> );  
    Dense<float, 2> layer2( weight_initializer(6, 4), bias_initializer(4), sigmoid_activation<float, 2> );
    Dense<float, 2> output_layer( weight_initializer(4, 2), bias_initializer(2), sigmoid_activation<float, 2> );


    // Model pipeline
    auto model = [&](const Eigen::Tensor<float, 2>& X) {
        auto l1 = layer1(X);
        auto l2 = layer2(l1);
        auto result = output_layer(l2);
        return result;
    };

    // Input sample
    Eigen::Tensor<float, 2> input(1, 4);
    input.setValues({{-1.5f, 0.4f, 2.1f, -1.2f}});

    // Run model
    std::cout<<"The input is :\n" <<input<<"\n";
    auto output = model(input);
    std::cout << "The output is:\n" << output << "\n";
    
    
    std::cout<<"--------------flatten---------------\n\n";
    
    
    Eigen::Tensor<float,3> inp(10,3,3);
    inp.setRandom();
    std::cout<<"Input dimensions are: "<<inp.dimensions()<<"\n\n";
    std::cout<<"Input tensor:\n\n"<<inp<<"\n\n";
    auto op = flatten(inp);
    std::cout<<"output dimensions are:\n\n"<<op.dimensions()<<"\n\n";
    std::cout<<"output:\n\n"<<op<<"\n\n";
    return 0;
}