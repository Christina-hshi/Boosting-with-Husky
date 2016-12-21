/*
    Edit by Christina
*/
#pragma once

#include "mllib/Utility.hpp"
#include "mllib/Instances.hpp"
#include "mllib/Classifier.hpp"
#include "lib/aggregator_factory.hpp"

namespace husky{
  namespace mllib{

    class RealAdaBoost : public Classifier{
      private:
        std::vector<Classifier*> community;
        Classifier* baseModel;
        int max_iter;

      public:
        RealAdaBoost(){}
        explicit RealAdaBoost(Classifier* bM, int mi): max_iter(mi){
          baseModel = bM->clone();
        }

        void fit(const Instances& instances);
        void fit(const Instances& instances, std::string instance_weight_name);

        AttrList<Instance, Prediction>&  predict(const Instances& instances,std::string prediction_name="prediction");
        // the clone function will only copy hyperparameters but not estimated parameter
        Classifier* clone(int seed=0){
          return new RealAdaBoost(baseModel, max_iter);
        }
        ~RealAdaBoost(){
          for(int x = 0; x < community.size(); x++){
            delete community[x];
          }
          delete baseModel;
        }
        
        void clear_community(){
          for(int x = 0; x < community.size(); x++){
            delete community[x];
          }
          community.clear();
        }
    };
  }
}
