/*
    Edit by Christina
*/

#include "mllib/RealAdaBoost.hpp"

namespace husky{
  namespace mllib{
    void RealAdaBoost::fit(const Instances& instances){
      std::string instance_weight_name = "RABweight_temporary";
      clear_community();
      
      husky::lib::Aggregator<double> global_weight(0.0,
          [](double& a, const double& b){ a += b;},
          [](double& v){ v = 0; }
          );
      global_weight.to_reset_each_iter();
        
      auto& ac = husky::lib::AggregatorFactory::get_channel();
      auto& weight = instances.createAttrlist<double>( instance_weight_name);

      //first initialize each attributes with a equal weight;
      double ini_weight = (double)1 / (double)instances.numInstances;
      list_execute(instances.enumerator(), {}, {},
          [&](Instance& instance){
            weight.set(instance, ini_weight);
          }
          );

      std::string prediction_name = "RABprediction_temporary";
      for(int iter = 0; iter < max_iter; iter++){
        //training a classifier and give prediction
        Classifier* model = baseModel->clone();
        model->fit(instances, instance_weight_name);

        community.push_back(model);
        auto& prediction = model->predict(instances, prediction_name);
        
        //update weight for each sample
        list_execute(instances.enumerator(), {}, {&ac},
            [&](Instance& instance){
              int label = instances.get_class(instance);
              double p = prediction.get(instance).proba[label];
              
              //caution: overflow e.g. introduce thresholding to avoid overflow
              double new_weight = weight.get(instance) * (1-p) / p;
              weight.set(instance, new_weight);
              
              global_weight.update(new_weight);
            }
            );
        double global_weight_tmp = global_weight.get_value();

        list_execute(instances.enumerator(), {}, {},
            [&](Instance& instance){
              weight.set(instance, weight.get(instance)/global_weight_tmp);
            }
            );
      }

      instances.deleteAttrlist(instance_weight_name);
      instances.deleteAttrlist(prediction_name);
    }

    /*
     *fit with a customerized initial weight
     */
    void RealAdaBoost::fit(const Instances& instances, std::string instance_weight_name){
      clear_community();
      
      husky::lib::Aggregator<double> global_weight(0.0,
          [](double& a, const double& b){ a += b;},
          [](double& v){ v = 0; }
          );
      global_weight.to_reset_each_iter();
        
      auto& ac = husky::lib::AggregatorFactory::get_channel();
      
      auto& weight = instances.getAttrlist<double>( instance_weight_name);

      //first normalize the weight
      list_execute(instances.enumerator(), {}, {&ac},
          [&](Instance& instance){
            global_weight.update(weight.get(instance));
          }
          );
      double gw_tmp = global_weight.get_value();
      list_execute(instances.enumerator(), {}, {},
          [&](Instance& instance){
            weight.set(instance, weight.get(instance)/gw_tmp);
          }
          );
     
      std::string prediction_name = "RABprediction_temporary";

      for(int iter = 0; iter < max_iter; iter++){
        //training a classifier and give prediction
        Classifier* model = baseModel->clone();
        model->fit(instances, instance_weight_name);

        community.push_back(model);
        auto& prediction = model->predict(instances, prediction_name);
        
        //update weight for each sample
        list_execute(instances.enumerator(), {}, {&ac},
            [&](Instance& instance){
              int label = instances.get_class(instance);
              double p = prediction.get(instance).proba[label];
              
              //caution: overflow e.g. introduce thresholding to avoid overflow
              double new_weight = weight.get(instance) * (1-p) / p;
              weight.set(instance, new_weight);
              
              global_weight.update(new_weight);
            }
            );

        double global_weight_tmp = global_weight.get_value();
        list_execute(instances.enumerator(), {}, {},
            [&](Instance& instance){
              weight.set(instance, weight.get(instance)/global_weight_tmp);
            }
            );

      }

      instances.deleteAttrlist(prediction_name);
    }
    
    AttrList<Instance, Prediction>& RealAdaBoost::predict(const Instances& instances,std::string prediction_name){
      AttrList<Instance, Prediction>&  prediction = instances.enumerator().has_attrlist(prediction_name)? instances.getAttrlist<Prediction>(prediction_name) : instances.createAttrlist<Prediction>(prediction_name);

      //initialize all the prediction
      list_execute(instances.enumerator(), {}, {},
        [&](Instance& instance){
          prediction.set(instance, Prediction(0, std::vector<double>(instances.numClasses, 0.0)));
        });

      std::string predict_name_baseModel = "RABprediction_baseModel";

      //sum up the the probability predicted by all base classifier in community.
      for(int x = 0; x < community.size(); x++){
        AttrList<Instance, Prediction>& baseModel_prediction = community[x]->predict(instances, predict_name_baseModel);
        list_execute(instances.enumerator(), {}, {},
            [&](Instance& instance){
              prediction.set(instance, Prediction(0, baseModel_prediction.get(instance).proba + prediction.get(instance).proba));
            });    
      }

      //normalize probability and select the lable with higheset probability as prediction.
      list_execute(instances.enumerator(), {}, {},
        [&](Instance& instance){
          vec_double p_vec = prediction.get(instance).proba;
          p_vec /= sum(p_vec);
          //choose label with highest probability
          double max_p = 0;
          int label;
          for(int x = 0; x < p_vec.size(); x++){
            if(p_vec[x] > max_p){
              max_p = p_vec[x];
              label = x;
            }
          }

          prediction.set(instance, Prediction(label, p_vec));

        });
      instances.deleteAttrlist(predict_name_baseModel);

      return prediction;
    }

  }
}
