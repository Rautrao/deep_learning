#ifndef ACTIVATION_FUNCTIONS_H_
#define ACTIVATION_FUNCTIONS_H_

template <int NumIndices_>
class ActivationFunction // base class
{
public:
    ActivationFunction(std::string name) : name(std::move(name)) {}
    virtual ~ActivationFunction() {}

    virtual Tensor<NumIndices_> evaluate(const Tensor<NumIndices_> &Z) const = 0;

    virtual Tensor<NumIndices_ + 1> jacobian(const Tensor<NumIndices_> &Z) const = 0;

    virtual Tensor<NumIndices_> operator()(const Tensor<NumIndices_> &Z) const
    {
        return this->evaluate(Z);
    }

    const std::string &get_name() const
    {
        return this->name;
    }

private:
    std::string name;
};

template <int NumIndices_>
class Softmax : public ActivationFunction<NumIndices_>
{
public:
    Softmax() : ActivationFunction<NumIndices_>("book.activations.softmax") {}
    virtual ~Softmax() {} 

    Tensor<NumIndices_> evaluate(const Tensor<NumIndices_> &Z) const {
        auto dimensions = Z.dimensions();

        int instance_length = dimensions.at(NumIndices_ - 1); // for rank 2 tensor : no of columns

        Eigen::array<int, NumIndices_> reshape_dim; // arrays of size NumIndices
        Eigen::array<int, NumIndices_> bcast;     

        for (int i = 0; i < NumIndices_ - 1; ++i) {  
            reshape_dim[i] = dimensions[i];  // reshape dimensions same as dimensions
            bcast[i] = 1;                    // bcast tells eigens how many times to repeat each dimension
        }
        reshape_dim[NumIndices_ - 1] = 1;     // instance size set to one 
        bcast[NumIndices_ - 1] = instance_length;  // bcast tells eigens how many times to repeat each dimension
 
        Eigen::array<int, 1> depth_dim({NumIndices_ - 1});    
        auto z_max = Z.maximum(depth_dim);                 // getting max out of all features 
        auto max_reshaped = z_max.reshape(reshape_dim);    
        // z_max has initial feature dimension as 1 as we need to broadcast the elements  
        auto max_values = max_reshaped.broadcast(bcast);   //  last dimension repeates no of features times

        // avoiding overflow
        auto diff = Z - max_values;  // normalization

        auto expo = diff.exp();
        auto expo_sums = expo.sum(depth_dim);   
        auto sums_reshaped = expo_sums.reshape(reshape_dim);  // sums each row of the normalized tensor
        auto sums = sums_reshaped.broadcast(bcast);           // broadcasting sums 
        Tensor<NumIndices_> result = expo / sums;             // dividing each parameter by sums of its particular feature dimension

        return result;
    }

    Tensor<NumIndices_ + 1> jacobian(const Tensor<NumIndices_> &Z) const
    {

        Tensor<NumIndices_> T = this->evaluate(Z); // T will contain a probablistic disturbuition matrix
 
        const auto dimensions = T.dimensions();
        const int S = dimensions[NumIndices_ - 1]; // extracting last dimension

        Eigen::array<Eigen::Index, NumIndices_ + 1> reshape_dimensions; 
        for (int i = 0; i < NumIndices_ - 1; ++i)
        {
            reshape_dimensions[i] = dimensions[i]; // reshape dimensions same as dimensions as of now except last dimension
        }

        reshape_dimensions[NumIndices_ - 1] = 1; // setting last second dimension to 1 
        reshape_dimensions[NumIndices_] = dimensions[NumIndices_ - 1];    //setting last dimension to no of features 

        const auto T_reshaped = T.reshape(reshape_dimensions); // converted to higer dimension tensor by 1
        // current dimensions in this case : dimension[0]x1xdimension[1]  

        Tensor<2> _2D_diagonal (S, S);  
        _2D_diagonal.setZero();
        for (int i = 0; i < S; ++i) 
            _2D_diagonal(i, i) = 1.f;

        Eigen::array<Eigen::Index, NumIndices_ + 1> diagonal_dimensions;

        for (int i = 0; i < NumIndices_ - 1; ++i)
        {
            diagonal_dimensions[i] = 1; 
        }

        diagonal_dimensions[NumIndices_ - 1] = S;
        diagonal_dimensions[NumIndices_] = S;

        const auto diagonal_reshaped = _2D_diagonal.reshape(diagonal_dimensions);  //1x2x2 in this case

        Eigen::array<Eigen::Index, NumIndices_ + 1> diagonal_bcast;
        for (int i = 0; i < NumIndices_ - 1; ++i)
        {
            diagonal_bcast[i] = dimensions[i]; 
        }

        diagonal_bcast[NumIndices_ - 1] = 1; 
        diagonal_bcast[NumIndices_] = 1; // diagonal_bcast becomes dimensions[0]x1x1 
        // this means clone the first dimension dimensions[0] times and keep the rest dimensions only once
        
        Tensor<NumIndices_ + 1> diagonal = diagonal_reshaped.broadcast(diagonal_bcast); 
        
        Eigen::array<Eigen::Index, NumIndices_ + 1> T_bcast;
        for (int i = 0; i <= NumIndices_; ++i)
        {
            T_bcast[i] = 1;  
        }

        T_bcast[NumIndices_ - 1] = S;  // T_bcast becomes  : 1 x features x 1 (its easy if went through visualization :)

        Tensor<NumIndices_ + 1> T_extended = T_reshaped.broadcast(T_bcast);

        Eigen::array<Eigen::Index, NumIndices_ + 1> transposed_dim;
        for (int i = 0; i <= NumIndices_; ++i)
        {
            transposed_dim[i] = i;
        }

        transposed_dim[NumIndices_] = NumIndices_ - 1; 
        transposed_dim[NumIndices_ - 1] = NumIndices_;  // 0 x NumIndices_ x NumIndices_ - 1 (0x2x1)
        Tensor<NumIndices_ + 1> T_extended_transposed = T_extended.shuffle(transposed_dim);
        // performing transpose on last two dimensions 
        // now rows becomes columns  :columns contain the probablistic distribution 
        
        const auto prod = T_extended * T_extended_transposed; // go through the book for more details 
                        //         
        const Tensor<NumIndices_ + 1> result = T_extended * diagonal - prod;

        return result;
    }
};

#endif