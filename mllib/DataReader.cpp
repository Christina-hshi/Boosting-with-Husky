#include "mllib/DataReader.hpp"

namespace husky{
    namespace mllib{
        bool svReader(Instances& instances,std::string filepath, std::string delimiter = " \t", LABEL_TYPE label_type = LABEL_TYPE::Y){

            husky::lib::Aggregator<int> total_num_examples(0, [](int & a, const int & b) { a += b;});
            husky::lib::Aggregator<int> num_features(0, [](int & a, const int & b) { if(a==0)a = b;else if(a==b) return; else throw std::length_error( "input data: inconsistant dimensionality!" );});
            husky::lib::Aggregator<int> num_classes(0, [](int & a, const int & b) { a= std::max(a,b);});

            string key_prefix = std::to_string(husky::Context::get_global_tid) + "_";
            unsigned long long num_local_instances = 0;

            auto& ac = husky::lib::AggregatorFactory::get_channel();
            auto parser = [&](boost::string_ref & chunk) {
                if (chunk.size() == 0)
                    return;

                // seperate the string
                boost::char_separator<char> sep(delimiter);
                boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
                husky::mllib::Instance instance;
                instance.key = key_prefix + std::to_string(num_local_instances++);
                
                for(auto& w : tok)
                {
                    instance.X.push_back(std::stod(w));    
                }
                
                instances.add(std::move(instance));
                total_num_examples.update(1);
            };

            husky::io::HDFSLineInputFormat infmt;
            infmt.set_input(filepath);
            husky::base::log_msg("start loading "+filepath);
            husky::load(infmt, {&ac},parser);
            
            //create corresponding label
            list_execute(instances.enumerator(), {}, {&ac}, [&](Instance& instance){
                    switch(label_type)
                    {
                    case LABEL_TYPE::NO_LABEL :
                        num_features.update(instance.X.size());
                        break;
                    case LABEL_TYPE::Y :
                        double last = instance.X.back();
                        instance.X.pop_back();

                        num_features.update(instance.X.size() - 1);
                        // set y attributes instances.set
                        instances.set_y(instance, last);
                        break;
                    case LABEL_TYPE::CLASS :
                        int c_label = instance.X.back(); //tested from 1-1000000000
                        instance.X.pop_back();

                        num_classes.update(c_label + 1);
                        num_features.update(instance.X.size() - 1);
                        //set class attributes
                        instances.set_class(instances, c_label);
                        break;

                    default :
                        throw std::invalid_argument("label_type " + std::to_string(label_type) +  " dosen't exit!");
                    }                

                });
            husky::lib::AggregatorFactory::sync();
            instances.numClasses=num_classes.get_value();
            instances.numAttributes=num_features.get_value();
            instances.numInstances=total_num_examples.get_value();

            husky::base::log_msg("finished loading "+filepath);
            instances.globalize();
            husky::base::log_msg("finished globalizing instances");
            
            return instances;
        }
    }
}
