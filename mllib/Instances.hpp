#pragma once
#include <vector>
#include <string>
typedef std::vector<double> vec_double;
#include <core/engine.hpp>
namespace husky{
  namespace mllib{
class Instance{
public:
    using KeyT = std::string;
    KeyT key;
    Instance(){}
    explicit Instance(const KeyT& k) : key(k) {}

    virtual KeyT const & id() const { return key;}

    vec_double X;

    Instance& operator=(const Instance& instance){
      if (this == &instance)
          return *this;
      // do the copy
      key=instance.key;
      X=instance.X;

      // return the existing object so we can chain this operator
      return *this;
    }
    friend husky::BinStream& operator<<(husky::BinStream& stream, const Instance& u) {
        stream << u.key << u.X;
        return stream;
    }
    friend husky::BinStream& operator>>(husky::BinStream& stream, Instance& u) {
        stream >> u.key >> u.X;
        return stream;
    }
};

class Instances{
private:
  ObjList<Instance>& list;
  std::set<std::string> attr_namelist;
  AttrList<Instance, double>&  ylist;
  AttrList<Instance, int>& classlist;
public:
  int numAttributes;
  int numInstances;
  int numClasses;
  Instances() : list(husky::ObjListFactory::create_objlist<Instance>()),ylist(list.create_attrlist<double>("y")),classlist(list.create_attrlist<int>("class")){
    attr_namelist.insert("y");
    attr_namelist.insert("class");
  }
  size_t add(const Instance& instance){
    return list.add_object(instance);
  }
  void globalize(){
    husky::globalize(list);
  }
  template <typename AttrT>
  auto& createAttrlist(std::string name){
    if(attr_namelist.find(name) != attr_namelist.end())
      throw std::runtime_error("duplicated name of attribute lists");
    attr_namelist.insert(name);
    return list.create_attrlist<AttrT>(name);

  }
  template <typename AttrT>
  auto& getAttrlist(std::string name){
    if(attr_namelist.find(name) == attr_namelist.end())
      throw std::runtime_error("attribute list -"+ name+" doesn't exists");
    return list.get_attrlist<AttrT>(name);
  }
  void deleteAttrlist(std::string name){
    if(attr_namelist.find(name) == attr_namelist.end())
      throw std::runtime_error("attribute list -"+ name+" doesn't exists");
    list.del_attrlist(name);
    attr_namelist.erase(name);
  }
  void set_y(const Instance& instance,double y){
    ylist.set(instance,y);
  }
  double get_y(const Instance& instance) const{
    return ylist.get(instance);
  }
  void set_class(const Instance& instance,int label){
    classlist.set(instance,label);
  }
  int get_class(const Instance& instance) const{
    return classlist.get(instance);

  }
  auto& enumerator() const{
    return list;
  }
  //= will not copy extra attribute list(only copy ylist and classlist)
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
        set_class(copy,instances.get_class(instance));

    });
    globalize();

    // return the existing object so we can chain this operator
    return *this;
  }


};

}
}
