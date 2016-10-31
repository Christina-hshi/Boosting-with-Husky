#pragma once
#include <mllib/Instances.hpp>
typedef vector<double> vec_double;
using namespace husky;
class Estimator{
public:
  virtual void fit(const mllib::Instances& instances) = 0;
  virtual void predict(const mllib::Instances& instances,vec_double& predictions, int label=-1)=0;
  virtual Estimator* model clone(int seed=0)=0;
}
