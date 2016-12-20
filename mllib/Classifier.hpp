#pragma once
#include "mllib/Instances.hpp"

namespace husky{
  namespace mllib{
      class Prediction{
      public:
          int label;
          vec_double proba;
          
          explicit Prediction(int l, vec_double p): label(l), proba(p){}
      };
      class Classifier{
          public:
              virtual void fit(const Instances& instances) = 0;
              virtual AttrList<Instance, Prediction>&  predict(Instances& instances,std::string prediction_name="prediction")=0;
              // the clone function will only copy hyperparameters but not estimated parameter
              virtual Classifier* clone(int seed=0)=0;
      };
}
}
