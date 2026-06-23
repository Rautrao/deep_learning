#ifndef LOSSES_H_
#define LOSSES_H_

#include "definitions.hpp"

template <int NumIndices_>
class LossFunction // abstract class 
{
public:
    LossFunction(std::string name) : name(std::move(name)) {}
    virtual ~LossFunction() {}

    virtual float evaluate(const Tensor<NumIndices_> &Y, const Tensor<NumIndices_> &T) const = 0;

    virtual Tensor<NumIndices_> derivative(const Tensor<NumIndices_> &Y, const Tensor<NumIndices_> &T) const = 0;

    virtual float operator()(const Tensor<NumIndices_> &Y, const Tensor<NumIndices_> &T) const
    {
        return this->evaluate(Y, T);
    }

    const std::string &get_name() const
    {
        return this->name;
    }

protected:
    virtual float loss(const float expected, const float actual) const = 0;
    virtual float cwise_derivative(const float expected, const float actual) const = 0;

private:
    std::string name;
};

template <int NumIndices_>
class CategoricalCrossEntropy : public LossFunction<NumIndices_> // derived class
{
public:
    CategoricalCrossEntropy() : LossFunction<NumIndices_>("book.losses.categorical_cross_entropy") {}
    virtual ~CategoricalCrossEntropy() {}

    virtual float evaluate(const Tensor<NumIndices_> &Y, const Tensor<NumIndices_> &T) const
    {
        Tensor<NumIndices_> temp = Y.binaryExpr(T, [this](const float expected, const float actual)
                                                  { return this->loss(expected, actual); });

        const Tensor<0> red = temp.sum();

        int instance_size = Y.dimension(NumIndices_ - 1);

        float result = red(0) * instance_size / Y.size();
        return result;

    }

    virtual Tensor<NumIndices_> derivative(const Tensor<NumIndices_> &Y, const Tensor<NumIndices_> &T) const
    {
        Tensor<NumIndices_> result = Y.binaryExpr(T, [this](const float expected, const float actual)
                                                    { return this->cwise_derivative(expected, actual); });
        return result;
    }

protected:
    virtual float loss(const float expected, const float actual) const
    {
        float y_true = expected;
        float y_pred = book::utils::clip(actual);
        float result = -y_true * log(y_pred);
        return result;
    }

    virtual float cwise_derivative(const float expected, const float actual) const
    {
        float y_actual = book::utils::clip(actual);
        float result = 1. + -expected / y_actual;
        return result;
    }
};

#endif