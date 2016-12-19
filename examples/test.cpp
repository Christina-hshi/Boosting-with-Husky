/*
    Edit by Christina
*/

#include "mllib/DataReader.hpp"
#include "mllib/MaxAbsScaler.hpp"
#include "mllib/LinearRegression_SGD.hpp"
#include "mllib/LogisticRegression.hpp"

void MaxAbsScaer_test();
void LinearRegression_SGD_test();
void LogisticRegression_test();

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    if (husky::init_with_args(argc, argv,args)) {
        husky::run_job(LogisticRegression_test);
        //husky::run_job(LinearRegression_SGD_test);
        return 0;
    }
    return 1;
}

void test(){
    matrix_double m1;
}

void LogisticRegression_test(){
    using namespace husky::mllib;
    using namespace std;

    Instances instances;
    svReader(instances, husky::Context::get_param("input"), boost::char_separator<char>(","), LABEL_TYPE::CLASS);

    MaxAbsScaler scaler;
    scaler.fit_transform(instances);
   
    /*
    auto& weight_attrList = instances.createAttrlist<double>("weight");
    double num_intances = instances.numInstances;
    list_execute(instances.enumerator(), {}, {}, [&](Instance& instance){
        weight_attrList.set(instance, (double)1/num_intances);
    });
    */

    /*
     * Parameters specification
     *  max_iter
     *  eta0: initial leaarning rate
     *  
     */
    LogisticRegression log_r(50000, 1, 0.001, 0.00001, 100, MODE::GLOBAL, instances.numClasses);
    
    log_r.fit(instances);

    auto& prediction = log_r.predict(instances);

    //calculate classification error
    husky::lib::Aggregator<long int> c_error(0, [](long int& a, const long int b){a += b;},
            [](long int& v){v = 0;});

    auto& ac = husky::lib::AggregatorFactory::get_channel();

    list_execute(instances.enumerator(), {}, {&ac},
            [&](Instance& instance){
                if(prediction.get(instance) != instances.get_class(instance)){
                    c_error.update(1);
                }
            }
        );

    if(husky::Context::get_global_tid() == 0){
        husky::base::log_msg("Misclassified " + std::to_string(c_error.get_value()) + " out of " + std::to_string(instances.numInstances));
    }
    //scaler.inverse_transfrom(instances);
    return;
}

void LinearRegression_SGD_test(){
    using namespace husky::mllib;

    Instances instances;
    svReader(instances, husky::Context::get_param("input"), boost::char_separator<char>("\t"), LABEL_TYPE::Y);

    /* check for load balance
    int count = 0;
    list_execute(instances.enumerator(), {}, {}, [&count](Instance& instance){
        count++;   
    });
    husky::base::log_msg(std::to_string(husky::Context::get_global_tid()) + std::to_string(count));
    */
    
    auto& weight_attrList = instances.createAttrlist<double>("weight");
    double num_intances = instances.numInstances;
    list_execute(instances.enumerator(), {}, {}, [&](Instance& instance){
        weight_attrList.set(instance, (double)1/num_intances);
    });

    MaxAbsScaler scaler;
    scaler.fit_transform(instances);
    
    LinearRegression_SGD lr_SGD(50000, 2, 0.0001, 0.0001, 100);

    
    lr_SGD.fit(instances);

    //scaler.inverse_transfrom(instances);
    return;
}

void MaxAbsScaer_test(){
    using namespace husky::mllib;

    Instances instances;
    svReader(instances, husky::Context::get_param("input"), boost::char_separator<char>("\t"), LABEL_TYPE::NO_LABEL);

    list_execute(instances.enumerator(), [&instances](Instance& instance){
        husky::base::log_msg(vec_to_str(instance.X));
    });
    
    husky::base::log_msg("Finished load data!");
    
    MaxAbsScaler scaler;
    scaler.fit_transform(instances);
    
    husky::base::log_msg("Scaler");
    husky::base::log_msg(vec_to_str(scaler.get_params()));
    husky::base::log_msg("Data after scaling");
    list_execute(instances.enumerator(), [&instances](Instance& instance){
        husky::base::log_msg(vec_to_str(instance.X));
    });
   
    husky::base::log_msg("Finished scaling with MaxAbsScaler!");

    scaler.inverse_transfrom(instances);
    
    husky::base::log_msg("Data is transferred back");
    list_execute(instances.enumerator(), [&instances](Instance& instance){
        husky::base::log_msg(vec_to_str(instance.X));
    });
//husky::base::log_msg("Finished scaling with MaxAbsScaler!");
    scaler.fit(instances);
    husky::base::log_msg("Data is fitted.");

    husky::base::log_msg("Scaler");

    husky::base::log_msg(vec_to_str(scaler.get_params()));
    husky::base::log_msg("Data no transferred");
    list_execute(instances.enumerator(), [&instances](Instance& instance){
        husky::base::log_msg(vec_to_str(instance.X));
    });

    scaler.transform(instances);
 
    husky::base::log_msg("Data after scaling");
    list_execute(instances.enumerator(), [&instances](Instance& instance){
        husky::base::log_msg(vec_to_str(instance.X));
    });
   
    husky::base::log_msg("Finished scaling with MaxAbsScaler!");

    scaler.inverse_transfrom(instances);
    
    husky::base::log_msg("Data is transferred back");
    list_execute(instances.enumerator(), [&instances](Instance& instance){
        husky::base::log_msg(vec_to_str(instance.X));
    });
}

