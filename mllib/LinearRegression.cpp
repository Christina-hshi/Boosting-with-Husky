#include "mllib/LinearRegression.hpp"
#include "mllib/Utility.hpp"

namespace husky{
  namespace mllib{






void LinearRegression::fit(const mllib::Instances& original_instances){

  mllib::Instances instances=original_instances;
  auto& pseudo_list=husky::ObjListFactory::create_objlist<PseudoObject>();
  if(husky::Context::get_global_tid()==0)
    pseudo_list.add_object(PseudoObject(0));

  globalize(pseudo_list);

  auto& ch1 =husky::ChannelFactory::create_push_combined_channel<matrix, husky::SumCombiner<matrix>>(instances.enumerator(), pseudo_list);
  auto& ch2 =husky::ChannelFactory::create_push_combined_channel<vec_double, husky::SumCombiner<vec_double>>(instances.enumerator(), pseudo_list);

  int dimension=instances.numAttributes+1;


  lib::Aggregator<vec_double> pv(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va = vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});


  list_execute(instances.enumerator(),{},{&ch1,&ch2}, [&instances,&ch1,&ch2,dimension](Instance& instance) {
    instance.X.push_back(1);
    matrix A(dimension, vec_double(dimension));
    for(int a=0;a<dimension;a++){
      for(int b=0;b<dimension;b++){
        A[a][b]=instance.X[a]*instance.X[b];
      }
    }

    ch1.push(A,0);

    vec_double B(dimension);
    for(int a=0;a<dimension;a++){
        B[a]=instance.X[a]*instances.get_y(instance);
    }

    ch2.push(B,0);
  });


  auto& ac = lib::AggregatorFactory::get_channel();

  list_execute(pseudo_list, {&ch1,&ch2},{&ac},[&ch1,&ch2,dimension,&pv](PseudoObject& obj) {
      matrix M = ch1.get(obj);
      vec_double XTy=ch2.get(obj);
      MatrixInversion(M);

      vec_double output(dimension,0.0);
      MatrixVectormultiplication(M,XTy,output);

      pv.update(output);
  });

  param_vec=pv.get_value();


}

void LinearRegression::fit(const Instances& instances,std::string instance_weight_name){


  auto& weight_list=instances.getAttrlist<double>(instance_weight_name);


  auto& pseudo_list=husky::ObjListFactory::create_objlist<PseudoObject>();
  if(husky::Context::get_global_tid()==0)
    pseudo_list.add_object(PseudoObject(0));

  globalize(pseudo_list);

  auto& ch1 =husky::ChannelFactory::create_push_combined_channel<matrix, husky::SumCombiner<matrix>>(instances.enumerator(), pseudo_list);
  auto& ch2 =husky::ChannelFactory::create_push_combined_channel<vec_double, husky::SumCombiner<vec_double>>(instances.enumerator(), pseudo_list);

  int dimension=instances.numAttributes+1;


  lib::Aggregator<vec_double> pv(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va = vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});


  list_execute(instances.enumerator(),{},{&ch1,&ch2}, [&instances,&ch1,&ch2,dimension,&weight_list](Instance& instance) {
    vec_double here=instance.X;
    here.push_back(1);
    instance.X.push_back(1);
    matrix A(dimension, vec_double(dimension));
    for(int a=0;a<dimension;a++){
      for(int b=0;b<dimension;b++){
        A[a][b]=here[a]*here[b];
      }
    }

    ch1.push(A,0);

    vec_double B(dimension);
    for(int a=0;a<dimension;a++){
        B[a]=here[a]*instances.get_y(instance);
    }

    ch2.push(B,0);
    /*
    vec_double this_instance=instance.X;
    this_instance.push_back(1);

    double this_weight=weight_list.get(instance);

    matrix A(dimension, vec_double(dimension));
    for(int a=0;a<dimension;a++){
      for(int b=0;b<dimension;b++){
        A[a][b]=this_instance[a]*this_instance[b]*this_weight;
      }
    }

    ch1.push(A,0);

    vec_double B(dimension);
    for(int a=0;a<dimension;a++){
        B[a]=this_instance[a]*instances.get_y(instance)*this_weight;
    }

    ch2.push(B,0);
    */
  });


  auto& ac = lib::AggregatorFactory::get_channel();

  list_execute(pseudo_list, {&ch1,&ch2},{&ac},[&ch1,&ch2,dimension,&pv](PseudoObject& obj) {
      matrix M = ch1.get(obj);
      vec_double XTy=ch2.get(obj);

      MatrixInversion(M);

      vec_double output(dimension,0.0);

      MatrixVectormultiplication(M,XTy,output);
      //base::log_msg("no seg fault");
      pv.update(output);
  });

  param_vec=pv.get_value();
  //base::log_msg("no seg fault");

}

AttrList<Instance, double>&  LinearRegression::predict(const mllib::Instances& instances,std::string prediction_name){
  AttrList<Instance, double>&  prediction= instances.createAttrlist<double>(prediction_name);
  list_execute(instances.enumerator(), [&prediction,this](Instance& instance) {
    vec_double feature_vector=instance.X;
    feature_vector.push_back(1);
    prediction.set(instance,feature_vector*param_vec);
  });
  return prediction;


}




}}
