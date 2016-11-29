/*
    Edit by Christina
*/
#pragma once

#include <limits>
#include "mllib/Utility.hpp"
#include "mllib/Instances.hpp"
#include "mllib/Estimator.hpp"
#include "lib/aggregator_factory.hpp"

namespace husky{
    namespace mllib{
   
        class LinearRegression_SGD : public Estimator{

            private:
                vec_double param_vec;
                int max_iter; //Maximun iterations|pass through the data, default:5
                double eta0; //The initial learning rate. The default value is 1.
                double alpha; //Constant that multuplies the regularization term, defualt: 0.0001.
                double trival_improve_th; //If weighted squared error improve is less than trival_improve_th, then this improve is trival.
                int max_trival_improve_iter; //If there are max_trival_improve_iter sequential iteration with trival_improve, then training will be stopped. 
                
                /* 
                Default settings
                    Squared loss
                    L2 penalty (least squares of 'param_vec'), a regularizer
                    invscaling learning_rate : eta = eta0/pow(t, 0.5)
                */
                
                /* Variables for extension
                std::string loss;//The loss function to be used. Defaults to "squared_loss". Only "Squared loss" is implemented in this stage.
                std::string penalty; //The penalty to be used. Defaults to "L2", second norm of 'param_vec', which is a standard regularizer.
                std::string learning_rate; //Type of learning rate to be used. Options include (1)'constant': eta = eta0;(2)'invscaling': eta = eta0/pow(t, 0.5);
                */

            public:
                LinearRegression_SGD(){
                    //default parameters
                    max_iter = 5;
                    eta0 = 1;
                    alpha = 0.0001;
                    trival_improve_th = 0.001;
                    max_trival_improve_iter = 100;
                }
                LinearRegression_SGD(int m_iter, double e0, double al,  double trival_im, double max_trival_im) : max_iter(m_iter), eta0(e0), alpha(al), trival_improve_th(trival_im), max_trival_improve_iter(max_trival_im){}

                /*
                    train a linear regression model with even weighted instances.
                */
                void fit(const Instances& instances);
               
                 /*
                    train a linear regression model using instance weight, which is specified as a attribute list of original_instances, name of which is 'instance_weight name'.
                */
                void fit(const Instances& original_instances, std::string instance_weight_name);

                AttrList<Instance, double>&  predict(Instances& instances,std::string prediction_name="prediction");

                Estimator* clone(int seed=0){
                    return new LinearRegression_SGD(this->max_iter, this->eta0, this->alpha, this->trival_improve_th, this->max_trival_improve_iter);

                }

                vec_double get_parameters(){
                    return param_vec;
                }

        };

    }
}
