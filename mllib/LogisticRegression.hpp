/*
    Edit by Christina
*/
#pragma once

#include <limits>
#include "mllib/Utility.hpp"
#include "mllib/Instances.hpp"
#include "lib/aggregator_factory.hpp"
#include "mllib/Classifier.hpp"

namespace husky{
    namespace mllib{
   
        /*
         MODE
            GLOBAL: global training
            LOCAL: training using local data
         */
        enum class MODE {GLOBAL=0, LOCAL=1};
        
        class LogisticRegression : public Classifier{

            private:
                matrix_double param_matrix;//parameter matrix
                int max_iter; //Maximun iterations|pass through the data, default:5
                double eta0; //The initial learning rate. The default value is 1.
                double alpha; //Constant that multuplies the regularization term, defualt: 0.0001.
                double trival_improve_th; //If weighted squared error improve is less than trival_improve_th, then this improve is trival.
                int max_trival_improve_iter; //If there are max_trival_improve_iter sequential iteration with trival_improve, then training will be stopped. 
                MODE mode; //default: GLOBAL
                int class_num; //default: 2
                /* 
                Default settings
                    maximize conditional log likelihood 
                    gradient descent update rule
                    L2 penalty (least squares of 'param_vec'), a regularizer
                    invscaling learning_rate : eta = eta0/pow(t, 0.5) NOTE: learning rate will only be updated when the loss increases, at the same time t will be increased by 1.  
                */
                
                /* Variables for extension
                std::string loss;//The loss function to be used. Defaults to "squared_loss". Only "Squared loss" is implemented in this stage.
                std::string penalty; //The penalty to be used. Defaults to "L2", second norm of 'param_vec', which is a standard regularizer.
                std::string learning_rate; //Type of learning rate to be used. Options include (1)'constant': eta = eta0;(2)'invscaling': eta = eta0/pow(t, 0.5);
                */

            public:
                LogisticRegression(){
                    //default parameters
                    max_iter = 5;
                    eta0 = 1;
                    alpha = 0.0001;
                    trival_improve_th = 0.001;
                    max_trival_improve_iter = 100;
                    mode = MODE::GLOBAL;
                    class_num = 2;
                }
                LogisticRegression(int m_iter, double e0, double al,  double trival_im, double max_trival_im, MODE m, int classNum) : max_iter(m_iter), eta0(e0), alpha(al), trival_improve_th(trival_im), max_trival_improve_iter(max_trival_im), mode(m), class_num(classNum){}

                /*
                    train a logistic regression model with even weighted instances.
                */
                void fit(const Instances& instances);
               
                /*
                    train model in global or local mode.
                 */
                void local_fit(const Instances& instances);
                void global_fit(const Instances& instances);
                void local_fit(const Instances& instances, std::string instance_weight_name);
                void global_fit(const Instances& instances, std::string instance_weight_name);

                 /*
                    train a logistic regression model using instance weight, which is specified as a attribute list of original_instances, name of which is 'instance_weight_name'.
                */
                void fit(const Instances& instances, std::string instance_weight_name);
                
                AttrList<Instance, Prediction>&  predict(const Instances& instances,std::string prediction_name="prediction");
               
                Classifier* clone(int seed=0){
                    return new LogisticRegression(this->max_iter, this->eta0, this->alpha, this->trival_improve_th, this->max_trival_improve_iter, this->mode, this->class_num);

                }

                matrix_double get_parameters(){
                    return param_matrix;
                }

        };

    }
}
