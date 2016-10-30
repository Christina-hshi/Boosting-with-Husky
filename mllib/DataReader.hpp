#pragma once
#include <string>
#include <mllib/Instances.hpp>
#include <ctime>


namespace husky{
  namespace mllib{
    husky::mllib::Instances tsvReader(std::string filepath){
      husky::mllib::Instances instances();
      std::mt19937 generator(std::time(0));
      std::uniform_real_distribution<int> distribution(0, INT_MAX);
      auto parser = [](boost::string_ref & chunk) {
          if (chunk.size() == 0)
            return;

          // seperate the string
          boost::char_separator<char> sep(" \t");
          boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
          husky::mllib:Instance instance;
          instance.key = distribution(generator);
          boost::tokenizer<boost::char_separator<char>>::iterator it = tok.begin();
          while ((it+1) != tok.end()) {
              instance.X.push_back(stod(*it++));
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
