#pragma once

#include "mllib/Instances.hpp"
#include "mllib/Classifier.hpp"
#include "mllib/Utility.hpp"
#include "mllib/Estimator.hpp"
#include "lib/aggregator_factory.hpp"
#include <math.h>
#include <float.h>

namespace husky{
  namespace mllib{



class LogitBoost : public mllib::Classifier{


private:
  std::vector<std::vector<Estimator*>> baselearners;
  Estimator* baselearnermodel;
  double m_maxIterations;
  double m_heuristicStop;
  std::string gen_fit_name(double j,double m){
    return "fit_inlogit_"+std::to_string(j)+"_"+std::to_string(m);
  }

public:
  LogitBoost(Estimator* baselearner,double maxIter,double heuristicStop){baselearnermodel=baselearner;m_maxIterations=maxIter;m_heuristicStop=heuristicStop;}
  void fit(const mllib::Instances& original_instances);
  void fit(const Instances& instances, std::string instance_weight_name);
  AttrList<Instance, Prediction>&  predict(const Instances& instances,std::string prediction_name="prediction");
  Classifier* clone(int seed=0){
    return new LogitBoost(baselearnermodel,m_maxIterations,m_heuristicStop);

  }
  Estimator* get_baselearner(int j,int m){
    return baselearners[j][m];
  }

};
}
}
