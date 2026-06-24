#ifndef UTILS_H_
#define UTILS_H_

namespace book
{
    namespace utils
    {
        const struct
        {
            float EPSILON = 1e-7f;
            int PRECISION = 7;
        } GLOBAL;

        template <int NumIndices_, typename IndexType_>
        const std::vector<IndexType_> convert_to_vector_indexes(int index, const Eigen::DSizes<IndexType_, NumIndices_> &dimensions)
        {
            const int N = dimensions.size();
            std::vector<IndexType_> result(N);
            IndexType_ current = index;
            for (int i = 0; i < N; ++i)
            {
                IndexType_ curr_dim = dimensions[i];
                IndexType_ i_index = current % curr_dim;
                result.at(i) = i_index;
                current /= curr_dim;
            }
            return result;
        }

        template <typename ElementType_>
        const std::string convert_vector_to_string(const std::vector<ElementType_> &vec)
        {
            if (vec.empty())
                return std::string();

            return std::accumulate(vec.begin() + 1, vec.end(),
                                   std::to_string(vec[0]),
                                   [](const std::string &a, ElementType_ b)
                                   {
                                       return a + ", " + std::to_string(b);
                                   });
        }

        // puts the exact output value within [min_value,max_value]
        
        template <typename Scalar_>
        Scalar_ clip(Scalar_ value, Scalar_ min_value = GLOBAL.EPSILON, Scalar_ max_value = (1.f - GLOBAL.EPSILON)) 
        {
            if (value < min_value)
                value = min_value;
            else if (value > max_value)
                value = max_value;

            return value;
        }
    }
}

#endif