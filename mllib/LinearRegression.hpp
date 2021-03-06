#pragma once

#include "mllib/Instances.hpp"
#include "mllib/Estimator.hpp"

#include "lib/aggregator_factory.hpp"


namespace husky{
  namespace mllib{
using namespace husky;
class PseudoObject {
   public:
    typedef int KeyT;
    int key;
    PseudoObject() {}
    explicit PseudoObject(KeyT key) { this->key = key; }

    const int& id() const { return key; }
    friend husky::BinStream& operator<<(husky::BinStream& stream, const PseudoObject& u) {
        stream << u.key;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, PseudoObject& u) {
        stream >> u.key;
        return stream;
    }

};


class LinearRegression : public mllib::Estimator{

private:
  vec_double param_vec;

public:
  LinearRegression(){}
  void fit(const mllib::Instances& original_instances);
  void fit(const Instances& instances,std::string instance_weight_name);
  AttrList<Instance, double>&  predict(const mllib::Instances& instances,std::string prediction_name="prediction");

  Estimator* clone(int seed=0){
    return new LinearRegression();

  }
  vec_double get_parameters(){
    return param_vec;
  }

};
}
}
