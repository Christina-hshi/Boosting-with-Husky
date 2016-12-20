/*
    Edit by Christina
*/
#include "LogisticRegression.hpp"

namespace husky{
    namespace mllib{
        /*
           train a logistic regression model with even weighted instances.
           */
        void LogisticRegression::fit(const Instances& instances){
            if(this->mode == MODE::GLOBAL){
                global_fit(instances);
            }
            else if(this->mode == MODE::LOCAL){
                local_fit(instances);
            }
            else{
                throw std::invalid_argument("MODE " + std::to_string((int)mode) + " dosen't exit! Only 0(GLOBAL) and 1(LOCAL) mode are provided.");
            }
        }
        
        /*
           train a logistic regression model using instance weight, which is specified as a attribute list of original_instances, name of which is 'instance_weight_name'.
           */
        void LogisticRegression::fit(const Instances& instances, std::string instance_weight_name){
            if(this->mode == MODE::GLOBAL){
                global_fit(instances, instance_weight_name);
            }
            else if(this->mode == MODE::LOCAL){
                local_fit(instances, instance_weight_name);
            }
            else{
                throw std::invalid_argument("MODE " + std::to_string((int)mode) + " dosen't exit! Only 0(GLOBAL) and 1(LOCAL) mode are provided.");
            }        
        }       
        
        /*
           train model in global mode.
        */
        void LogisticRegression::global_fit(const Instances& instances){
            const int num_attri = instances.numAttributes;

            matrix_double init_mat;
            for(int i = 1; i < this->class_num; i++){
                init_mat.push_back(vec_double(num_attri + 1, 0.0));
            }

            husky::lib::Aggregator<matrix_double> para_mat(init_mat, 
                    [](matrix_double& a, const matrix_double& b){ a += b;},
                    [num_attri, this](matrix_double& m){
                        m.clear();
                        for(int i = 1; i < this->class_num; i++){
                            m.push_back(vec_double(num_attri + 1, 0.0));
                        }
                    });

            husky::lib::Aggregator<double> global_squared_error(0.0,
                    [](double& a, const double& b){ a += b;},
                    [](double& v){ v = 0; }
                    );
            global_squared_error.to_reset_each_iter();

            auto& ac = husky::lib::AggregatorFactory::get_channel();
           
            double old_weighted_squared_error = std::numeric_limits<double>::max();
            double eta = eta0;
            int eta_update_counter = 1;
            int trival_improve_iter = 0;

            //max_iter
            for(int iter = 1; iter <= max_iter && trival_improve_iter < max_trival_improve_iter; iter++){
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\teta: " + std::to_string(eta));
                }

                matrix_double para_update;
                for(int i = 1; i < this->class_num; i++){
                    para_update.push_back(vec_double(num_attri + 1, 0.0));
                }

                matrix_double para_old = para_mat.get_value();
                
                vec_double class_prob_tmp(this->class_num, 0.0);

                list_execute(instances.enumerator(), {}, {&ac}, 
                        [&](Instance& instance){
                        vec_double x_with_const_term = instance.X;
                        x_with_const_term.push_back(1);

                        //calculate probability
                        double sum_tmp = 0;
                        for(int x = 0; x < class_num - 1; x++){
                            class_prob_tmp[x] = exp(para_old[x] * x_with_const_term);
                            sum_tmp += class_prob_tmp[x];
                        }
                        sum_tmp += 1;
                        class_prob_tmp.back() = 1;

                        class_prob_tmp /= sum_tmp;
                        
                        
                        //calculate error and udpate with regularization
                        auto label_tmp = instances.get_class(instance);
                        class_prob_tmp[label_tmp] -= 1;
                        global_squared_error.update(class_prob_tmp * class_prob_tmp);

                        for(int x = 0; x < class_num - 1; x++){
                            para_update[x] -= (eta * class_prob_tmp[x] * x_with_const_term);
                        }
                    });

                //to do the division only once.
                para_update /= instances.numInstances;
               
                //add regularization
                double tmp;
                for(int x = 0; x < class_num - 1; x++){
                    tmp = para_old[x].back();
                    para_old[x].back() = 0;
                    para_update[x] -= (eta * alpha * para_old[x]);
                    para_old[x].back() = tmp;
                }
                
                para_mat.update(para_update);

                double weighted_squared_error = global_squared_error.get_value() / instances.numInstances;
                //output the training infor.
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("\tGlobal averaged squared error: " + std::to_string(weighted_squared_error));
                }
                
                if(old_weighted_squared_error < weighted_squared_error){
                    eta = eta0/pow(++eta_update_counter, 0.5);
                    trival_improve_iter = 0;
                }
                else if (fabs(old_weighted_squared_error - weighted_squared_error) < trival_improve_th ){
                    trival_improve_iter ++;
                }
                else{
                    trival_improve_iter = 0;
                }
                
                old_weighted_squared_error = weighted_squared_error;

                husky::lib::AggregatorFactory::sync();
                
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("\tParameters: " + matrix_to_str(para_mat.get_value()));
                }
            }

            this->param_matrix = para_mat.get_value(); 
            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Training completed!");
            }
            return;
        }
        
        void LogisticRegression::global_fit(const Instances& instances, std::string instance_weight_name){
            const int num_attri = instances.numAttributes;

            matrix_double init_mat;
            for(int i = 1; i < this->class_num; i++){
                init_mat.push_back(vec_double(num_attri + 1, 0.0));
            }

            husky::lib::Aggregator<matrix_double> para_mat(init_mat, 
                    [](matrix_double& a, const matrix_double& b){ a += b;},
                    [num_attri, this](matrix_double& m){
                        m.clear();
                        for(int i = 1; i < this->class_num; i++){
                            m.push_back(vec_double(num_attri + 1, 0.0));
                        }
                    });

            husky::lib::Aggregator<double> global_squared_error(0.0,
                    [](double& a, const double& b){ a += b;},
                    [](double& v){ v = 0; }
                    );
            global_squared_error.to_reset_each_iter();

            auto& ac = husky::lib::AggregatorFactory::get_channel();
            auto& weight_attrList = instances.getAttrlist<double>(instance_weight_name);
            
            double old_weighted_squared_error = std::numeric_limits<double>::max();
            double eta = eta0;
            int eta_update_counter = 1;
            int trival_improve_iter = 0;

            //max_iter
            for(int iter = 1; iter <= max_iter && trival_improve_iter < max_trival_improve_iter; iter++){
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\teta: " + std::to_string(eta));
                }

                matrix_double para_update;
                for(int i = 1; i < this->class_num; i++){
                    para_update.push_back(vec_double(num_attri + 1, 0.0));
                }

                matrix_double para_old = para_mat.get_value();
                
                vec_double class_prob_tmp(this->class_num, 0.0);

                list_execute(instances.enumerator(), {}, {&ac}, 
                        [&](Instance& instance){
                        vec_double x_with_const_term = instance.X;
                        x_with_const_term.push_back(1);

                        //calculate probability
                        double sum_tmp = 0;
                        for(int x = 0; x < class_num - 1; x++){
                            class_prob_tmp[x] = exp(x_with_const_term * para_old[x]);
                            sum_tmp += class_prob_tmp[x];
                        }
                        sum_tmp += 1;
                        class_prob_tmp.back() = 1;

                        class_prob_tmp /= sum_tmp;
                        
                        //calculate error and udpate with regularization
                        double weight = weight_attrList.get(instance);
                        
                        auto label_tmp = instances.get_class(instance);
                        class_prob_tmp[label_tmp] -= 1;
                        global_squared_error.update(weight * class_prob_tmp * class_prob_tmp);

                        for(int x = 0; x < class_num - 1; x++){
                            para_update[x] -= (weight * eta * class_prob_tmp[x] * x_with_const_term);
                        }
                        
                });

                double tmp;
                for(int x = 0; x < class_num - 1; x++){
                    tmp = para_old[x].back();
                    para_old[x].back() = 0; 
                    para_update[x] -= (eta * alpha * para_old[x]);
                    para_old[x].back() = tmp;
                }

                para_mat.update(para_update);

                double weighted_squared_error = global_squared_error.get_value();
                //output the training infor.
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("\tGlobal averaged squared error: " + std::to_string(weighted_squared_error));
                }
                
                if(old_weighted_squared_error < weighted_squared_error){
                    eta = eta0/pow(++eta_update_counter, 0.5);
                    trival_improve_iter = 0;
                }
                else if (fabs(old_weighted_squared_error - weighted_squared_error) < trival_improve_th ){
                    trival_improve_iter ++;
                }
                else{
                    trival_improve_iter = 0;
                }
                
                old_weighted_squared_error = weighted_squared_error;

                husky::lib::AggregatorFactory::sync();
                
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("\tParameters: " + matrix_to_str(para_mat.get_value()));
                }
            }

            this->param_matrix = para_mat.get_value(); 
            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Training completed!");
            }
            return;

        }

        /*
           train model in local mode.
        */
        void LogisticRegression::local_fit(const Instances& instances){
            int num_attri = instances.numAttributes;

            matrix_double param_update;
            for(int i = 1; i < this->class_num; i++){
               param_matrix.push_back(vec_double(num_attri + 1, 0.0));
               param_update.push_back(vec_double(num_attri + 1, 0.0));
            }

            double weighted_squared_error = 0;
            double old_weighted_squared_error = std::numeric_limits<double>::max();
            double eta = eta0;
            int eta_update_counter = 1;
            int trival_improve_iter = 0;

            //count how many instances in local machine
            int num_instances_local = 0;
            list_execute(instances.enumerator(),{},{},
                    [&num_instances_local](Instance& instance){
                        num_instances_local++;
                    }
            );

            //max_iter
            for(int iter = 1; iter <= max_iter && trival_improve_iter < max_trival_improve_iter; iter++){
                /*
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\teta: " + std::to_string(eta));
                }
                */
                //reset variables
                weighted_squared_error = 0;

                for(int i = 0; i < this->class_num-1; i++){
                    param_update[i].assign(num_attri + 1, 0.0);
                }
                    
                vec_double class_prob_tmp(this->class_num, 0.0);


                list_execute(instances.enumerator(), {}, {}, 
                        [&](Instance& instance){
                            vec_double x_with_const_term = instance.X;
                            x_with_const_term.push_back(1);
                            
                            //calculate probability
                            double sum_tmp = 0;
                            for(int x = 0; x < class_num-1; x++ ){
                                class_prob_tmp[x] = exp(x_with_const_term * param_matrix[x]);
                                sum_tmp += class_prob_tmp[x];
                            }
                            sum_tmp += 1;
                            class_prob_tmp.back() = 1;

                            class_prob_tmp /= sum_tmp;

                            //calculate error and update
                            auto label_tmp = instances.get_class(instance);
                            class_prob_tmp[label_tmp] -= 1;
                            weighted_squared_error += (class_prob_tmp * class_prob_tmp);

                            for(int x = 0; x < class_num - 1; x++){
                                param_update[x] -= (eta * class_prob_tmp[x] * x_with_const_term);
                            }

                        }
                );

                //update parameters.
                param_update /= num_instances_local;
                //regularization term
                double tmp;
                for(int x = 0; x < class_num - 1; x++){
                    tmp = param_matrix[x].back();
                    param_matrix[x].back() = 0;
                    param_update[x] -= (eta * alpha * param_matrix[x]);
                    param_matrix[x].back() = tmp;
                }
                
                param_matrix += param_update;

                weighted_squared_error /=  num_instances_local;

                //output local error.
                if(husky::Context::get_global_tid() == 0){
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\n"
                    + "Local averaged squared error: " + std::to_string(weighted_squared_error) + "\n"
                    + "Parameters: " + matrix_to_str(param_matrix));
                    //std::cout<<"Parameters: " + matrix_to_str(param_matrix)<<std::endl;
                }
                    
                if(old_weighted_squared_error < weighted_squared_error){
                    eta = eta0/pow(++eta_update_counter, 0.5);
                    trival_improve_iter = 0;
                }
                else if (fabs(old_weighted_squared_error - weighted_squared_error) < trival_improve_th ){
                    trival_improve_iter ++;
                }
                else{
                    trival_improve_iter = 0;
                }

                old_weighted_squared_error = weighted_squared_error;

            }

            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Training completed!");
            }
            return;
        }

        void LogisticRegression::local_fit(const Instances& instances, std::string instance_weight_name){
             int num_attri = instances.numAttributes;

            matrix_double param_update;
            for(int i = 1; i < this->class_num; i++){
               param_matrix.push_back(vec_double(num_attri + 1, 0.0));
               param_update.push_back(vec_double(num_attri + 1, 0.0));
            }

            auto& weight_attrList = instances.getAttrlist<double>(instance_weight_name);
            
            double weighted_squared_error = 0;
            double old_weighted_squared_error = std::numeric_limits<double>::max();
            double eta = eta0;
            int eta_update_counter = 1;
            int trival_improve_iter = 0;

            //count how many instances in local machine
            int num_instances_local = 0;
            list_execute(instances.enumerator(),{},{},
                    [&num_instances_local](Instance& instance){
                        num_instances_local++;
                    }
            );

            //max_iter
            for(int iter = 1; iter <= max_iter && trival_improve_iter < max_trival_improve_iter; iter++){
                /*
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\teta: " + std::to_string(eta));
                }
                */
                //reset variables
                weighted_squared_error = 0;

                for(int i = 0; i < this->class_num-1; i++){
                    param_update[i].assign(num_attri + 1, 0.0);
                }
                    
                vec_double class_prob_tmp(this->class_num, 0.0);


                list_execute(instances.enumerator(), {}, {}, 
                        [&](Instance& instance){
                            vec_double x_with_const_term = instance.X;
                            x_with_const_term.push_back(1);
                            
                            //calculate probability
                            double sum_tmp = 0;
                            for(int x = 0; x < class_num-1; x++ ){
                                class_prob_tmp[x] = exp(x_with_const_term * param_matrix[x]);
                                sum_tmp += class_prob_tmp[x];
                            }
                            sum_tmp += 1;
                            class_prob_tmp.back() = 1;

                            class_prob_tmp /= sum_tmp;

                            //calculate error and update
                            double weight = weight_attrList.get(instance);
                            
                            auto label_tmp = instances.get_class(instance);
                            class_prob_tmp[label_tmp] -= 1;
                            weighted_squared_error += (weight * class_prob_tmp * class_prob_tmp);

                            for(int x = 0; x < class_num - 1; x++){
                                param_update[x] -= (weight * eta * class_prob_tmp[x] * x_with_const_term);
                            }

                        }
                );

                //regularization term
                double tmp;
                for(int x = 0; x < class_num - 1; x++){
                    tmp = param_matrix[x].back();
                    param_matrix[x].back() = 0;
                    param_update[x] -= (eta * alpha * param_matrix[x]);
                    param_matrix[x].back() = tmp;
                }
                
                param_matrix += param_update;

                weighted_squared_error /=  num_instances_local;

                //output local error.
                if(husky::Context::get_global_tid() == 0){
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\n"
                    + "Local averaged squared error: " + std::to_string(weighted_squared_error) + "\n"
                    + "Parameters: " + matrix_to_str(param_matrix));
                }
                    
                if(old_weighted_squared_error < weighted_squared_error){
                    eta = eta0/pow(++eta_update_counter, 0.5);
                    trival_improve_iter = 0;
                }
                else if (fabs(old_weighted_squared_error - weighted_squared_error) < trival_improve_th ){
                    trival_improve_iter ++;
                }
                else{
                    trival_improve_iter = 0;
                }

                old_weighted_squared_error = weighted_squared_error;

            }

            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Training completed!");
            }
            return;
        }

        /*
         * predict #prediciton contains label and class_proba in it.
         */
        AttrList<Instance, Prediction>&  LogisticRegression::predict(Instances& instances,std::string prediction_name){
          if(this->mode == MODE::LOCAL){
            throw std::invalid_argument("Prediciton is not provided after training in LOCAL mode!");
          }

          AttrList<Instance, Prediction>&  prediction= instances.createAttrlist<Prediction>(prediction_name);
          list_execute(instances.enumerator(), [&prediction, this](Instance& instance) {
              vec_double feature_vector=instance.X;
              feature_vector.push_back(1);

              //calculate probability
              vec_double class_prob_tmp(class_num, 0.0);
              double sum_tmp = 0;
              for(int x = 0; x < class_num-1; x++ ){
                class_prob_tmp[x] = exp(feature_vector * param_matrix[x]);
                sum_tmp += class_prob_tmp[x];
              }
              sum_tmp += 1;
              class_prob_tmp.back() = 1;
              class_prob_tmp /= sum_tmp;

              //choose label with highest probability
              double max_p = 0;
              int label;
              for(int x = 0; x < class_num; x++){
                if(class_prob_tmp[x] > max_p){
                  max_p = class_prob_tmp[x];
                  label = x;
                }
              }

              prediction.set(instance, Prediction(label, class_prob_tmp));
          });
          return prediction;
        }

    }
}


