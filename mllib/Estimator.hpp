#pragma once
#include <mllib/Instances.hpp>

namespace husky{
  namespace mllib{
      class Estimator{
          public:
              virtual void fit(const Instances& instances) = 0;
              virtual AttrList<Instance, double>&  predict(Instances& instances,std::string prediction_name="prediction")=0;
              // the clone function will only copy hyperparameters but not estimated parameter
              virtual Estimator* clone(int seed=0)=0;
      };
}
}
