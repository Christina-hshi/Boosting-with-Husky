#include "mllib/DataReader.hpp"

namespace husky{
    namespace mllib{
        void svReader(Instances& instances,std::string filepath, boost::char_separator<char> delimiter, LABEL_TYPE label_type){

            husky::lib::Aggregator<unsigned long long> total_num_examples(0, [](unsigned long long & a, const unsigned long long & b) { a += b;});
            husky::lib::Aggregator<int> num_features(0, [](int & a, const int & b) { if(a==0)a = b;else if(a==b) return; else throw std::length_error( "input data: inconsistant dimensionality!" );});
            husky::lib::Aggregator<int> num_classes(0, [](int & a, const int & b) { a= std::max(a,b);});

            std::string key_prefix = std::to_string(husky::Context::get_global_tid()) + "_";
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

            husky::io::LineInputFormat infmt;
            infmt.set_input(filepath);
            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Start loading data from " + filepath);
            }
            husky::load(infmt, {&ac},parser);


            instances.globalize();
            husky::base::log_msg("finished loading "+filepath);
            //husky::base::log_msg("finished globalizing instances");
            //create corresponding label
            //husky::base::log_msg("start constructing y and class column");
            switch(label_type)
            {
                case LABEL_TYPE::NO_LABEL :
                    list_execute(instances.enumerator(), {}, {&ac}, [&](Instance& instance){
                            num_features.update(static_cast<int>(instance.X.size()));
                            });
                    break;
                case LABEL_TYPE::Y :
                    list_execute(instances.enumerator(), {}, {&ac}, [&](Instance& instance){
                            double last = instance.X.back();
                            instance.X.pop_back();

                            num_features.update(static_cast<int>(instance.X.size()));
                            // set y attributes instances.set
                            instances.set_y(instance, last);
                            });
                    break;
                case LABEL_TYPE::CLASS :
                    list_execute(instances.enumerator(), {}, {&ac}, [&](Instance& instance){
                            int c_label = std::lround(instance.X.back()); //tested from 1-1000000000
                            instance.X.pop_back();

                            num_classes.update(c_label + 1);
                            num_features.update(static_cast<int>(instance.X.size()));
                            //set class attributes
                            instances.set_class(instance, c_label);
                            });
                    break;

                default :
                    throw std::invalid_argument("label_type " + std::to_string((int)label_type) +  " dosen't exit!");
            }
            //husky::base::log_msg("finished constructing y and class column");
            husky::lib::AggregatorFactory::sync();
            instances.numClasses=num_classes.get_value();
            instances.numAttributes=num_features.get_value();
            instances.numInstances=total_num_examples.get_value();

            if (husky::Context::get_global_tid() == 0) {
                husky::base::log_msg("Data loading completed!");
            }
        }
    }
}
