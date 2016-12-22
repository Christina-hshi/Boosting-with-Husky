/*
    Edit by Christina
*/
#include "LinearRegression_SGD.hpp"

namespace husky{
    namespace mllib{
        
        /*
           train a linear regression model with even weighted instances.
         */
        void LinearRegression_SGD::fit(const Instances& instances){
            int num_attri = instances.numAttributes;
            
            vec_double init_vec(num_attri + 1, 0.0);//constant term is also considered.
            husky::lib::Aggregator<vec_double> para_vec(init_vec,
                    [](vec_double& a, const vec_double& b){ a += b;},
                    [num_attri](vec_double& v){ v = std::move(vec_double(num_attri + 1, 0.0)); }
                    );
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

                vec_double para_update(num_attri + 1, 0.0);
                vec_double para_old = para_vec.get_value();


                list_execute(instances.enumerator(), {}, {&ac}, 
                        [&](Instance& instance){
                        vec_double x_with_const_term = instance.X;
                        x_with_const_term.push_back(1);

                        double error = x_with_const_term * para_old - instances.get_y(instance); // not include regularization term.
                        global_squared_error.update(error * error);

                        para_update -= eta * (error * x_with_const_term + alpha * para_old);
                        }
                    );

                //to do the division only once.
                para_update /= instances.numInstances;
                
                //add regualarization
                double tmp = para_old.back();
                para_old.back() = 0;
                para_update -= eta * alpha * para_old;
                para_old.back() = tmp;

                para_vec.update(para_update);

                double weighted_squared_error = global_squared_error.get_value() / instances.numInstances;
                //output the training infor.
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\tGlobal averaged squared error: " + std::to_string(weighted_squared_error));
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
                    husky::base::log_msg("\tparameters: " + vec_to_str(para_vec.get_value()));
                }
            }

            this->param_vec = para_vec.get_value(); 
            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Training completed!");
            }
            return;
        }

         /*
           train a linear regression model using instance weight, which is specified as a attribute list of original_instances, name of which is 'instance_weight name'.
         */
        void LinearRegression_SGD::fit(const Instances& instances, std::string instance_weight_name){
            int num_attri = instances.numAttributes;

            vec_double init_vec(num_attri + 1, 0.0);//constant term is also considered.
            husky::lib::Aggregator<vec_double> para_vec(init_vec,
                    [](vec_double& a, const vec_double& b){ a += b;},
                    [num_attri](vec_double& v){ v = std::move(vec_double(num_attri + 1, 0.0)); }
                    );
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
                
                vec_double para_old = para_vec.get_value();

                list_execute(instances.enumerator(), {}, {&ac}, 
                    [&](Instance& instance){
                        vec_double x_with_const_term = instance.X;
                        x_with_const_term.push_back(1);

                        double weight = weight_attrList.get(instance);
                        double error = x_with_const_term * para_old - instances.get_y(instance); // not include regularization term.
                        global_squared_error.update(weight * error * error);

                        para_vec.update( -1 * weight * eta * (error * x_with_const_term));
                    }
                );
                //add regularization; w0 is not penalized.
                double m0 = para_old.back();
                para_old.back() = 0;
                para_vec.update(-eta * alpha * para_old);
                para_old.back() = m0;
                
                double weighted_squared_error = global_squared_error.get_value();
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

                husky::lib::AggregatorFactory::sync();

                //output the training infor.
                if (husky::Context::get_global_tid() == 0) {
                    husky::base::log_msg("Iter#" + std::to_string(iter) + "\tGlobal weighted squared error: " + std::to_string(weighted_squared_error));
                    husky::base::log_msg("\tparameters: " + vec_to_str(para_vec.get_value()));
                }

                old_weighted_squared_error = weighted_squared_error;
            }

            this->param_vec = para_vec.get_value(); 
            
            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Training completed!");
                husky::base::log_msg(vec_to_str(param_vec));
            }
            return;

        }

        AttrList<Instance, double>&  LinearRegression_SGD::predict(const Instances& instances,std::string prediction_name){

            AttrList<Instance, double>&  prediction= instances.createAttrlist<double>(prediction_name);
            list_execute(instances.enumerator(), [&prediction, this](Instance& instance) {
                    vec_double feature_vector=instance.X;
                    feature_vector.push_back(1);
                    prediction.set(instance,feature_vector*param_vec);
                    });
            return prediction;

        }

    }
}

