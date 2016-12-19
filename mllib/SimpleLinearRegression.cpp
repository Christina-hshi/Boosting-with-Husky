#include "mllib/SimpleLinearRegression.hpp"
#include "mllib/Utility.hpp"

namespace husky{
  namespace mllib{



void SimpleLinearRegression::fit(const mllib::Instances& instances){
  auto& ac = husky::lib::AggregatorFactory::get_channel();

  husky::lib::Aggregator<double> classmean(0.0,
          [](double& a, const double& b){ a += b;}
          );



  int dimension=instances.numAttributes;
  husky::lib::Aggregator<vec_double> mean(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});
  husky::lib::Aggregator<vec_double> SD(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});
  husky::lib::Aggregator<vec_double> SyD(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});
  husky::lib::Aggregator<vec_double> slope(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});


  double finalweightSum=instances.numInstances;
  list_execute(instances.enumerator(),{},{&ac}, [dimension,finalweightSum,&instances,&mean,&classmean](Instance& instance) {

    for(int i=0;i<dimension;i++){
      mean.update_any([i,&instance,finalweightSum](vec_double& d){
        d[i] += (instance.X[i])/finalweightSum;
      });
    }
    classmean.update(instances.get_y(instance));

  });
  double finalclassmean=classmean.get_value()/finalweightSum;
  vec_double finalmean=mean.get_value();
  list_execute(instances.enumerator(),{},{&ac}, [&instances,&SD,&SyD,&slope,finalclassmean,finalweightSum,dimension,&finalmean](Instance& instance) {

    for(int i=0;i<dimension;i++){
      /*
      yDiff=y[k]-classmean
        weightYDiff=sample_weight[k]*yDiff
        diff=X[k,i]-mean[i]
        weightDiff=sample_weight[k]*diff
        slope[i]+=weightYDiff*diff
        SD[i]+=weightDiff*diff
        SyD[i]+=weightYDiff*yDiff
      */
      double yDiff=instances.get_y(instance)-finalclassmean;

      double diff=instance.X[i]-finalmean[i];

      slope.update_any([i,yDiff,diff](vec_double& d){
        d[i] += yDiff*diff;
      });
      SD.update_any([i,diff](vec_double& d){
        d[i] += diff*diff;
      });
      SyD.update_any([i,yDiff](vec_double& d){
        d[i] += yDiff*yDiff;
      });
    }

  });

  husky::lib::Aggregator<int> selected(-1,
          [](int& a, const int& b){ a = b;}
          );

  husky::lib::Aggregator<double> Cslope(0.0,
          [](double& a, const double& b){ a = b;}
          );

  husky::lib::Aggregator<double> Cintercept(0.0,
          [](double& a, const double& b){ a = b;}
          );
  this->selected_attribute=-1;
  this->slope=0.0;
  this->intercept=0.0;
  if (husky::Context::get_global_tid() == 0) {
    vec_double finalslope=slope.get_value();
    vec_double finalSD=SD.get_value();
    vec_double finalSyD=SyD.get_value();
    double minSSE=DBL_MAX;
    for(int i=0;i<dimension;i++){
        double numerator = finalslope[i];
        finalslope[i]/=finalSD[i];
        double intercept = finalclassmean-finalslope[i]*finalmean[i];
        double sse=finalSyD[i]-finalslope[i]*numerator;
        if (sse< minSSE){
          minSSE=sse;
          this->selected_attribute=i;
          this->slope=finalslope[i];
          this->intercept=intercept;
        }
    }
    selected.update(this->selected_attribute);
    Cslope.update(this->slope);
    Cintercept.update(this->intercept);


  }
  husky::lib::AggregatorFactory::sync();
  this->selected_attribute=selected.get_value();
  this->slope=Cslope.get_value();
  this->intercept=Cintercept.get_value();






}

void SimpleLinearRegression::fit(const mllib::Instances& instances,std::string weight_name){

  auto& ac = husky::lib::AggregatorFactory::get_channel();

  husky::lib::Aggregator<double> classmean(0.0,
          [](double& a, const double& b){ a += b;}
          );

  husky::lib::Aggregator<double> weightSum(0.0,
          [](double& a, const double& b){ a += b;}
          );

  int dimension=instances.numAttributes;
  husky::lib::Aggregator<vec_double> mean(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});
  husky::lib::Aggregator<vec_double> SD(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});
  husky::lib::Aggregator<vec_double> SyD(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});
  husky::lib::Aggregator<vec_double> slope(vec_double(dimension,0), [](vec_double & va, const vec_double & vb) { va += vb;}, [dimension](vec_double & v){v = std::move(vec_double(dimension, 0));});
  auto& weight_attrList = instances.getAttrlist<double>(weight_name);
  list_execute(instances.enumerator(),{},{&ac}, [&weightSum,&weight_attrList](Instance& instance) {
    weightSum.update(weight_attrList.get(instance));

  });
  double finalweightSum=weightSum.get_value();
  list_execute(instances.enumerator(),{},{&ac}, [dimension,finalweightSum,&instances,&weight_attrList,&mean,&classmean](Instance& instance) {
    double current_weight=weight_attrList.get(instance);
    for(int i=0;i<dimension;i++){
      mean.update_any([i,&instance,current_weight,finalweightSum](vec_double& d){
        d[i] += (current_weight*instance.X[i])/finalweightSum;
      });
    }
    classmean.update(instances.get_y(instance)*current_weight);

  });
  double finalclassmean=classmean.get_value()/finalweightSum;
  vec_double finalmean=mean.get_value();
  list_execute(instances.enumerator(),{},{&ac}, [&weight_attrList,&instances,&SD,&SyD,&slope,finalclassmean,finalweightSum,dimension,&finalmean](Instance& instance) {
    double current_weight=weight_attrList.get(instance);
    for(int i=0;i<dimension;i++){
      /*
      yDiff=y[k]-classmean
        weightYDiff=sample_weight[k]*yDiff
        diff=X[k,i]-mean[i]
        weightDiff=sample_weight[k]*diff
        slope[i]+=weightYDiff*diff
        SD[i]+=weightDiff*diff
        SyD[i]+=weightYDiff*yDiff
      */
      double yDiff=instances.get_y(instance)-finalclassmean;
      double weightYDiff= current_weight * yDiff;
      double diff=instance.X[i]-finalmean[i];
      double weightDiff=current_weight*diff;
      slope.update_any([i,weightYDiff,diff](vec_double& d){
        d[i] += weightYDiff*diff;
      });
      SD.update_any([i,weightDiff,diff](vec_double& d){
        d[i] += weightDiff*diff;
      });
      SyD.update_any([i,weightYDiff,yDiff](vec_double& d){
        d[i] += weightYDiff*yDiff;
      });
    }

  });

  husky::lib::Aggregator<int> selected(-1,
          [](int& a, const int& b){ a = b;}
          );

  husky::lib::Aggregator<double> Cslope(0.0,
          [](double& a, const double& b){ a = b;}
          );

  husky::lib::Aggregator<double> Cintercept(0.0,
          [](double& a, const double& b){ a = b;}
          );
  this->selected_attribute=-1;
  this->slope=0.0;
  this->intercept=0.0;
  if (husky::Context::get_global_tid() == 0) {
    vec_double finalslope=slope.get_value();
    vec_double finalSD=SD.get_value();
    vec_double finalSyD=SyD.get_value();
    double minSSE=DBL_MAX;
    for(int i=0;i<dimension;i++){
        double numerator = finalslope[i];
        finalslope[i]/=finalSD[i];
        double intercept = finalclassmean-finalslope[i]*finalmean[i];
        double sse=finalSyD[i]-finalslope[i]*numerator;
        if (sse< minSSE){
          minSSE=sse;
          this->selected_attribute=i;
          this->slope=finalslope[i];
          this->intercept=intercept;
        }
    }
    selected.update(this->selected_attribute);
    Cslope.update(this->slope);
    Cintercept.update(this->intercept);


  }
  husky::lib::AggregatorFactory::sync();
  this->selected_attribute=selected.get_value();
  this->slope=Cslope.get_value();
  this->intercept=Cintercept.get_value();




}
AttrList<Instance, double>&  SimpleLinearRegression::predict(mllib::Instances& instances,std::string prediction_name){
  AttrList<Instance, double>&  prediction= instances.createAttrlist<double>(prediction_name);
  list_execute(instances.enumerator(), [&prediction,this](Instance& instance) {
    vec_double feature_vector=instance.X;

    prediction.set(instance,instance.X[this->selected_attribute]*this->slope+this->intercept);
  });
  return prediction;

}

  }
}
