#pragma once
#include <vector>
#include <string>
typedef std::vector<double> vec_double;
#include <core/engine.hpp>
namespace husky{
  namespace mllib{
class Instance{
public:
    typedef int KeyT;
    KeyT key;
    XYNode(){}
    explicit XYNode(const KeyT& k) : key(k) {}

    virtual KeyT const & id() const { return key;}

    vec_double X;
    double y;
    int label;
    string last;
    Instance& operator=(const Instance& instance){
      if (this == &instance)
          return *this;
      // do the copy
      key=instance.key;
      X=instance.X;
      y=instance.y
      label=instance.label;
      last=instance.last;
      // return the existing object so we can chain this operator
      return *this;
    }
};

class Instances{
private:
  auto & list;
public:
  int numAttributes;
  int numInstances;
  Instances(){
    list=husky::ObjListFactory::create_objlist<Instance>();
  }
  add(const Instance& instance){
    list.add_object(instance);
  }
  auto& enumerator(){
    return list;
  }
  Instances& operator= (const Instances& instances)
  {
    // self-assignment guard
    if (this == &instances)
        return *this;

    // do the copy
    numAttributes = instances.numAttributes;
    numInstances = instances.numInstances;
    list_execute(instances.enumerator(), [](Instance& instance) {
        Instance copy=instance;
        add(copy);
    });

    // return the existing object so we can chain this operator
    return *this;
  }


};

}
}
