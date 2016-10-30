#pragma once
#include <string>
#include <mllib/Instances.hpp>
#include "io/input/hdfs_line_inputformat.hpp"
#include "boost/tokenizer.hpp"
#include <stdexcept>

namespace husky{
  namespace mllib{
    husky::mllib::Instances tsvReader(std::string filepath){
      husky::mllib::Instances instances;
      int total_num_workers=husky::Context::get_worker_info()->get_num_workers();
      int key_giver=((int)INT_MAX/total_num_workers) * husky::Context::get_global_tid();


      auto parser = [&instances,&key_giver](boost::string_ref & chunk) {
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
          it=tok.begin();
          while (length >1){
            instance.X.push_back(std::stod(*it++));
            length--;

          }
          instance.last=*it;
          instances.add(std::move(instance));

      };


      husky::io::HDFSLineInputFormat infmt;
      infmt.set_input(filepath);
      husky::base::log_msg("start loading "+filepath);
      husky::load(infmt, parser);
      husky::base::log_msg("finished loading "+filepath);
      instances.globalize();
      husky::base::log_msg("finished globalizing instances");
      husky::base::log_msg("start constructing y and class column");
      list_execute(instances.enumerator(), [&instances](Instance& instance) {
          instances.set_y(instance,std::stod(instance.last));
          instances.set_class(instance,std::stoi(instance.last));

      });
      husky::base::log_msg("finished constructing y and class column");
      return instances;

}
}
}
