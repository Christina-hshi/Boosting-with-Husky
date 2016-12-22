#pragma once
#include <mllib/Instances.hpp>

namespace husky{
  namespace mllib{
      class Estimator{
          public:
              virtual void fit(const Instances& instances) = 0;
              virtual void fit(const Instances& instances,std::string instance_weight_name){
                
              }
              virtual AttrList<Instance, double>&  predict(const Instances& instances,std::string prediction_name="prediction")=0;
              // the clone function will only copy hyperparameters but not estimated parameter
              virtual Estimator* clone(int seed=0)=0;
      };
}
}
