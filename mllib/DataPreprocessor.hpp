#pragma once
#include "mllib/Instances.hpp"

namespace husky{
    namespace mllib{
        class DataPreprocessor
        {
        public:
            DataPreprocessor(){};
            virtual void fit(const Instances& instances) = 0;
            virtual void fit_transform(Instances& instances) = 0;
            virtual void transform(Instances& instances) = 0;
            virtual void inverse_transfrom(Instances& instances) = 0; 
        };
    }
}
