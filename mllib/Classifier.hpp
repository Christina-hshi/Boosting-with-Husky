#pragma once
#include "mllib/Instances.hpp"

namespace husky{
  namespace mllib{
      class Prediction{
      public:
          int label;
          vec_double proba;
          
          Prediction(){};
          explicit Prediction(int l, vec_double p): label(l), proba(p){}
          Prediction& operator=(const Prediction& p){
     		 if (this == &p)
          		return *this;
      // do the copy
      		label=p.label;
      		proba=p.proba;

      // return the existing object so we can chain this operator
      		return *this;
    		} 
          friend husky::BinStream& operator<<(husky::BinStream& stream, const Prediction& p){
            stream << p.label << p.proba;
          }
          friend husky::BinStream& operator>>(husky::BinStream& stream, Prediction& p){
            stream >> p.label >> p.proba;
          }
      };
      class Classifier{
          public:
              virtual void fit(const Instances& instances) = 0;
              virtual void fit(const Instances& instances, std::string instance_weight_name) = 0;
              virtual AttrList<Instance, Prediction>&  predict(const Instances& instances,std::string prediction_name="prediction")=0;
              // the clone function will only copy hyperparameters but not estimated parameter
              virtual Classifier* clone(int seed=0)=0;
      };
}
}
