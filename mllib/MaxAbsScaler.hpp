#pragma once

#include "mllib/Utility.hpp"
#include "mllib/DataPreprocessor.hpp"
#include "lib/aggregator_factory.hpp"

namespace husky{
    namespace mllib{
        
        class MaxAbsScaler: public DataPreprocessor{
        public:
            MaxAbsScaler(){};
            vec_double& get_params(){ return parameters;}
            void set_params(vec_double params){
                parameters.clear();
                parameters = params;
            }
            void fit(const Instances& instances);
            void fit_transform(Instances& instances);
            void transform(Instances& instances);
            void inverse_transfrom(Instances& instances); 

        private:
            vec_double  parameters;
        };
    }
}
