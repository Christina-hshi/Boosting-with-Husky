#pragma once
#include <mllib/Instances.hpp>
typedef vector<double> vec_double;
using namespace husky;
class Estimator{
public:
  virtual void fit(const mllib::Instances& instances) = 0;
  virtual void predict(mllib::Instances& instances)=0;
  // the clone function will only copy hyperparameters but not estimated parameter
  virtual Estimator* model clone(int seed=0)=0;
}
