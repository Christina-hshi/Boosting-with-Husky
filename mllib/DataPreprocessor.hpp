#include "mllib/Instances.hpp"

namespace husky{
    namespace mllib{
        class DataPreprocessor
        {
        public:
            virtual void fit(const Instances& instances) = 0;
            virtual void fit_transform(Instance& instances) = 0;
            virtual void transform(Instances& instances) = 0;
            virtual void  inverse_transfrom(Instance& instances) = 0; 
        }
    }
}
