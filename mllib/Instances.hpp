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
    Instance(){}
    explicit Instance(const KeyT& k) : key(k) {}

    virtual KeyT const & id() const { return key;}

    vec_double X;
    string last;
    Instance& operator=(const Instance& instance){
      if (this == &instance)
          return *this;
      // do the copy
      key=instance.key;
      X=instance.X;
      last=instance.last;
      // return the existing object so we can chain this operator
      return *this;
    }
    friend husky::BinStream& operator<<(husky::BinStream& stream, const Instance& u) {
        stream << u.key << u.X << u.last;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, Instance& u) {
        stream >> u.key >> u.X >> u.last;
        return stream;
    }
};

class Instances{
private:
  auto & list;
  auto & ylist;
  auto & classlist;
public:
  int numAttributes;
  int numInstances;
  int numClasses;
  Instances(){
    list=husky::ObjListFactory::create_objlist<Instance>();
    ylist=list.create_attrlist<double>("y");
    classlist=list.create_attrlist<int>("class");

  }
  void add(const Instance& instance){
    list.add_object(instance);
  }
  void globalize(){
    husky::globalize(list);
  }
  void set_y(const Instance& instance,double y){
    ylist.set(instance,y);
  }
  double get_y(const Instance& instance){
    return ylist.get(instance);
  }
  void set_class(const Instance& instance,int label){
    classlist.set(instance,label);
  }
  int get_class(const Instance& instance){
    return classlist.get(instance);

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
    numClasses = instances.numClasses;
    list_execute(instances.enumerator(), [this,&instances](Instance& instance) {
        Instance copy=instance;
        add(copy);
        set_y(copy,instances.get_y(instance));
        set_class(copy,instances.get_class(instance))

    });
    husky::globalize(*this);

    // return the existing object so we can chain this operator
    return *this;
  }


};

}
}
