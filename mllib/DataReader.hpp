#pragma once
#include <string>
#include <mllib/Instances.hpp>
#include "io/input/hdfs_line_inputformat.hpp"
#include "boost/tokenizer.hpp"
#include <stdexcept>
#include "lib/aggregator_factory.hpp"

namespace husky{
  namespace mllib{
    husky::mllib::Instances& tsvReader(husky::mllib::Instances& instances,std::string filepath){
      int total_num_workers=husky::Context::get_worker_info()->get_num_workers();
      int key_giver=((int)INT_MAX/total_num_workers) * husky::Context::get_global_tid();

      husky::lib::Aggregator<int> total_num_examples(0, [](int & a, const int & b) { a += b;});
      husky::lib::Aggregator<int> num_features(0, [](int & a, const int & b) { if(a==0)a = b;else if(a==b) return; else throw std::runtime_error( "input data error: the dimensionality" );});
      husky::lib::Aggregator<int> num_classes(0, [](int & a, const int & b) { a= std::max(a,b);});
      auto& ac = husky::lib::AggregatorFactory::get_channel();
      auto parser = [&instances,&key_giver,&total_num_examples,&num_features](boost::string_ref & chunk) {
          if (chunk.size() == 0)
            return;

          // seperate the string
          boost::char_separator<char> sep(" \t");
          boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
          husky::mllib::Instance instance;
          instance.key = key_giver++;
          int length=0;
          boost::tokenizer<boost::char_separator<char>>::iterator it = tok.begin();
          while (it != tok.end()) {
              length++;
              it++;
          }
	  // length = std:distance(tok.begin(),tok.end());
          it=tok.begin();
          num_features.update(length-1);
          while (length >1){
            instance.X.push_back(std::stod(*it++));
            length--;

          }
          instance.last=*it;
          instances.add(std::move(instance));
          total_num_examples.update(1);


      };

      husky::lib::AggregatorFactory::sync();
      husky::io::HDFSLineInputFormat infmt;
      infmt.set_input(filepath);
      husky::base::log_msg("start loading "+filepath);
      husky::load(infmt, {&ac},parser);
      husky::base::log_msg("finished loading "+filepath);
      instances.globalize();
      husky::base::log_msg("finished globalizing instances");
      husky::base::log_msg("start constructing y and class column");
      list_execute(instances.enumerator(), {},{&ac},[&instances,&num_classes](Instance& instance) {
          instances.set_y(instance,std::stod(instance.last));
          instances.set_class(instance,std::stoi(instance.last));
          num_classes.update(std::stoi(instance.last)+1);

      });
      instances.numClasses=num_classes.get_value();
      instances.numAttributes=num_features.get_value();
      instances.numInstances=total_num_examples.get_value();
      husky::base::log_msg("finished constructing y and class column");
      return instances;

}
}
}
