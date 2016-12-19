#pragma once
#include <mllib/Instances.hpp>

namespace husky{
  namespace mllib{
      class Classifier{
          public:
              virtual void fit(const Instances& instances) = 0;
              virtual AttrList<Instance, double>&  predict(Instances& instances,int classnow,std::string prediction_name="predicted_probability")=0;
              // using the default method to convert probability to output label
              virtual AttrList<Instance, int>&  predict_label(Instances& instances,std::string prediction_name="predicted_label")=0;
              // the clone function will only copy hyperparameters but not estimated parameter
              virtual Estimator* clone(int seed=0)=0;
      };
}
}
