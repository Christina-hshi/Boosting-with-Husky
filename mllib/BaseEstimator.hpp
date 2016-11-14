#pragma once
#include <mllib/Instances.hpp>

using namespace husky;
namespace husky{
  namespace mllib{
class Estimator{
public:
  virtual void fit(const mllib::Instances& instances) = 0;
  virtual AttrList<Instance, double>&  predict(mllib::Instances& instances,std::string prediction_name="prediction")=0;
  // the clone function will only copy hyperparameters but not estimated parameter
  virtual Estimator* clone(int seed=0)=0;
};
}
}
