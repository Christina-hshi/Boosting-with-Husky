#pragma once

#include "mllib/Instances.hpp"
#include "mllib/Estimator.hpp"
#include <float.h>
#include "lib/aggregator_factory.hpp"


namespace husky{
  namespace mllib{




class SimpleLinearRegression : public mllib::Estimator{

private:
  int selected_attribute;
  double intercept;
  double slope;

public:
  SimpleLinearRegression() {selected_attribute=-1;}
  void fit(const mllib::Instances& original_instances);
  void fit(const mllib::Instances& original_instances,std::string weight_name);
  AttrList<Instance, double>&  predict(mllib::Instances& instances,std::string prediction_name="prediction");
  Estimator* clone(int seed=0){
    return new SimpleLinearRegression();

  };
  double get_slope(){
    return this->slope;
  };
  double get_intercept(){
    return this->intercept;
  }
  int get_selected(){
    return this->selected_attribute;
  }

};
}
}
