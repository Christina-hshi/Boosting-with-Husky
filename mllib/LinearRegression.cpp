#include "mllib/LinearRegression.hpp"
#include "mllib/Utility.hpp"

namespace husky{
  namespace mllib{



using namespace husky;


void LinearRegression::fit(const mllib::Instances& original_instances){

  mllib::Instances instances=original_instances;
  auto& pseudo_list=husky::ObjListFactory::create_objlist<PseudoObject>();

//  pseudo_list.add_object(PseudoObject(0));

  globalize(pseudo_list);

  auto& ch1 =husky::ChannelFactory::create_push_combined_channel<matrix, husky::SumCombiner<matrix>>(instances.enumerator(), pseudo_list);
  auto& ch2 =husky::ChannelFactory::create_push_combined_channel<vec_double, husky::SumCombiner<vec_double>>(instances.enumerator(), pseudo_list);

  list_execute(instances.enumerator(), [](Instance& instance) {
      instance.X.push_back(1);
  });
  int dimension=instances.numAttributes+1;


  lib::Aggregator<vec_double> pv(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va = vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});


  list_execute(instances.enumerator(), [&instances,&ch1,&ch2,dimension](Instance& instance) {

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
  globalize(pseudo_list);
    base::log_msg(std::to_string(pseudo_list.get_size()));
  list_execute(pseudo_list, [&ch1,&ch2,dimension,&pv](PseudoObject& obj) {
      if(obj.key==0)
        std::cout << "size=0 but execute" << std::endl;
      matrix M = ch1.get(obj);
      std::cout << "sizeA:" << M.size()<<std::endl;
      vec_double XTy=ch2.get(obj);
      std::cout << "sizeB:" << XTy.size()<<std::endl;
      MatrixInversion(M);

      vec_double output(dimension,0.0);
      MatrixVectormultiplication(M,XTy,output);

      pv.update(output);
  });

  param_vec=pv.get_value();


}

void LinearRegression::predict(mllib::Instances& instances,std::string prediction_name){


};




}}
