#pragma once
#include <string>
#include <stdexcept>
#include "mllib/Instances.hpp"
#include "io/input/hdfs_line_inputformat.hpp"
#include "boost/tokenizer.hpp"
#include "lib/aggregator_factory.hpp"

namespace husky{
    namespace mllib{
        
        /*
         LABEL_TYPE
            NO_LABEL: no label in data, e.g. in test dataset
            Y: for regression problem, LABEL will be stored using double
            CLASS: for classification problem, LABEL will be stored using int. So maximum number of different class shouldn't exceed MAX_INT. And class should start from 0, followed by continuous intergers.
         */
        enum class LABEL_TYPE {NO_LABEL = 0, Y = 1, CLASS = 2};

        /*
         * svReader: separated value reader
         */
        bool svReader(Instances& instances,std::string filepath, std::string delimiter = " \t", LABEL_TYPE label_type = LABEL_TYPE::Y);


    }
}
