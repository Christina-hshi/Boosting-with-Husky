#include "mllib/LogitBoost.hpp"


namespace husky{
  namespace mllib{


void LogitBoost::fit(const mllib::Instances& instances){
  int nClass=instances.numClasses;

  Instances z_instances=instances;
  auto& ylist=z_instances.createAttrlist<int>("y_inlogit");
  auto& templist=z_instances.createAttrlist<double>("temp_inlogit");
  auto& templist_2=z_instances.createAttrlist<double>("temp_inlogit_2");
  list_execute(z_instances.enumerator(), [&ylist,&z_instances](Instance& instance) {
    ylist.set(instance,z_instances.get_class(instance));

  });

  //allocate maxf
  //the bug from husky Aggregator



  auto& ac = husky::lib::AggregatorFactory::get_channel();

  std::vector<std::vector<Estimator*>> temp_baselearners;
  for(int j=0;j<nClass;j++){
    std::vector<Estimator*> v;
    this->baselearners.push_back(v);
    temp_baselearners.push_back(v);
  }


  std::vector<AttrList<Instance, double>*> weight_lists;
  std::vector<AttrList<Instance, double>*> prediction_lists;
  std::vector<AttrList<Instance, double>*> f_lists;

//initialize p and f
  for(int j=0;j<nClass;j++){
    std::string wname="weight_inlogit_class_"+std::to_string(j);
    std::string pname="prediction_inlogit_class_"+std::to_string(j);
    std::string fname="f_inlogit_class_"+std::to_string(j);



    prediction_lists.push_back(&(z_instances.createAttrlist<double>(pname)));
    weight_lists.push_back(&(z_instances.createAttrlist<double>(wname)));
    f_lists.push_back(&(z_instances.createAttrlist<double>(fname)));



    list_execute(z_instances.enumerator(), [j,&prediction_lists,&f_lists,nClass](Instance& instance) {
      (*(f_lists[j])).set(instance,0.0);
      (*(prediction_lists[j])).set(instance,(double)1.0/nClass);

    });
  }

  husky::lib::Aggregator<double> weightSum(0.0,
          [](double& a, const double& b){ a += b;}
          );
  weightSum.to_reset_each_iter();
  for(int m=0;m<this->m_maxIterations;m++){

    for(int j=0;j<nClass;j++){
      auto& nowp=*(prediction_lists[j]);
      auto& noww=*(weight_lists[j]);
      double finalweightSum;
      auto& nowlearner=temp_baselearners[j];
      list_execute(z_instances.enumerator(), {},{&ac},[j,&noww,&z_instances,&nowp,&ylist,&weightSum](Instance& instance) {
        double z;
        double y;
        double p=nowp.get(instance);

        if(ylist.get(instance)==j){
          y=1;
          z=1.0/p;
          if(z>3)
            z_instances.set_y(instance,3);
          else
            z_instances.set_y(instance,z);
        }
        else
        {
          y=0;
          z=(-1.0)/(1-p);
          if(z<-3)
            z_instances.set_y(instance,-3);
          else
            z_instances.set_y(instance,z);
        }
        double w=((y-p)/z);
        noww.set(instance,w);
        weightSum.update(w);


      });
      finalweightSum=weightSum.get_value();
      if(finalweightSum<=0)
        goto enditer;

      list_execute(z_instances.enumerator(), [&noww,finalweightSum](Instance& instance) {
        double w=noww.get(instance);
        noww.set(instance,w/finalweightSum);

      });

      nowlearner.push_back((this->baselearnermodel)->clone());//

      nowlearner[m]->fit(z_instances,"weight_inlogit_class_"+std::to_string(j));

      nowlearner[m]->predict(z_instances,gen_fit_name(j,m));
    }
    //cannot get here


    // computing F
    list_execute(z_instances.enumerator(), [&z_instances,nClass,&templist,m,this](Instance& instance) {
      double sum=0;

      for(int j=0;j<nClass;j++){
        auto& nowfit=z_instances.getAttrlist<double>(gen_fit_name(j,m));
        sum+= nowfit.get(instance);

      }
      sum/=nClass;

      templist.set(instance,sum);

    });
    list_execute(z_instances.enumerator(), [&z_instances,nClass,&templist,m,this,&f_lists](Instance& instance) {
      for(int j=0;j<nClass;j++){
        auto& nowf=*(f_lists[j]);
        auto& nowfit=z_instances.getAttrlist<double>(gen_fit_name(j,m));
        double now=nowfit.get(instance)-templist.get(instance);
        now*=((nClass-1.0)/nClass);

        double in=nowf.get(instance);
        nowf.set(instance,in+now);

      }
    });
    //compute P

    list_execute(z_instances.enumerator(), [&templist_2,&f_lists,nClass](Instance& instance) {
      double maximum=-DBL_MAX;
      for(int j=0;j<nClass;j++){
        auto& nowf=*(f_lists[j]);
        double now=nowf.get(instance);
        if(now>maximum)
          maximum=now;
      }
      templist_2.set(instance,maximum);
    });

    list_execute(z_instances.enumerator(), [&templist,&f_lists,nClass,&templist_2](Instance& instance) {
      double sum=0;
      for(int j=0;j<nClass;j++){
        auto& nowf=*(f_lists[j]);

        sum+= exp(nowf.get(instance)-templist_2.get(instance));
      }
      templist.set(instance,sum);


    });
    list_execute(z_instances.enumerator(), [&templist,&prediction_lists,&f_lists,nClass,&templist_2](Instance& instance) {
      for(int j=0;j<nClass;j++){
        auto& nowp=*(prediction_lists[j]);
        auto& nowf=*(f_lists[j]);
        double now=exp(nowf.get(instance)-templist_2.get(instance))/templist.get(instance);
        nowp.set(instance,now);


      }

    });









  }
  enditer:

  for(int j=0;j<nClass;j++){
    for(int m=0;m<this->m_maxIterations;m++){
      (this->baselearners)[j].push_back(temp_baselearners[j][m]);

    }
  }
  //yes it solved some of the problem
  for(int j=0;j<nClass;j++){
    std::string fname="f_inlogit_class_"+std::to_string(j);
    z_instances.deleteAttrlist(fname);

  }
  for(int m=0;m<this->m_maxIterations;m++){
    for(int j=0;j<nClass;j++){
      z_instances.deleteAttrlist(gen_fit_name(j,m));
    }
  }
  z_instances.deleteAttrlist("temp_inlogit");
  z_instances.deleteAttrlist("temp_inlogit_2");




  //deallocate the AggregatorObject


}

void LogitBoost::fit(const Instances& instances, std::string instance_weight_name){
  //cannot implement weighted version

}

AttrList<Instance, Prediction>&  LogitBoost::predict(const Instances& z_instances,std::string prediction_name){
  AttrList<Instance, Prediction>&  prediction = z_instances.enumerator().has_attrlist(prediction_name)? z_instances.getAttrlist<Prediction>(prediction_name) : z_instances.createAttrlist<Prediction>(prediction_name);
    int nClass=z_instances.numClasses;

     auto& templist=z_instances.createAttrlist<double>("temp_inlogit");
     //no reasons of name copying of attribute list
     auto& templist_2=z_instances.createAttrlist<double>("temp_inlogit_2");

     std::vector<AttrList<Instance, double>*> f_lists;
     for(int j=0;j<nClass;j++){
       std::string fname="f_inlogit_class_"+std::to_string(j);

       f_lists.push_back(&(z_instances.createAttrlist<double>(fname)));



       list_execute(z_instances.enumerator(), [j,&f_lists,nClass](Instance& instance) {
         (*(f_lists[j])).set(instance,0.0);

       });
     }


     //compute F
     for(int m=0;m<this->m_maxIterations;m++){
       for(int j=0;j<nClass;j++){
         auto& nowlearner=this->baselearners[j];
         nowlearner[m]->predict(z_instances,gen_fit_name(j,m));
       }

       list_execute(z_instances.enumerator(), [&z_instances,nClass,&templist,m,this](Instance& instance) {
         double sum=0;

         for(int j=0;j<nClass;j++){
           auto& nowfit=z_instances.getAttrlist<double>(gen_fit_name(j,m));
           sum+= nowfit.get(instance);

         }
         sum/=nClass;

         templist.set(instance,sum);

       });
       list_execute(z_instances.enumerator(), [&z_instances,nClass,&templist,m,this,&f_lists](Instance& instance) {
         for(int j=0;j<nClass;j++){
           auto& nowf=*(f_lists[j]);
           auto& nowfit=z_instances.getAttrlist<double>(gen_fit_name(j,m));
           double now=nowfit.get(instance)-templist.get(instance);
           now*=((nClass-1.0)/nClass);

           double in=nowf.get(instance);
           nowf.set(instance,in+now);

         }
       });
     }
     //compute P
     list_execute(z_instances.enumerator(), [&templist_2,&f_lists,nClass](Instance& instance) {
       double maximum=-DBL_MAX;
       for(int j=0;j<nClass;j++){
         auto& nowf=*(f_lists[j]);
         double now=nowf.get(instance);
         if(now>maximum)
           maximum=now;
       }
       templist_2.set(instance,maximum);
     });

     list_execute(z_instances.enumerator(), [&templist,&f_lists,nClass,&templist_2](Instance& instance) {
       double sum=0;
       for(int j=0;j<nClass;j++){
         auto& nowf=*(f_lists[j]);

         sum+= exp(nowf.get(instance)-templist_2.get(instance));
       }
       templist.set(instance,sum);


     });
     list_execute(z_instances.enumerator(), [&templist,&f_lists,nClass,&templist_2,&prediction](Instance& instance) {
       std::vector<double> probs;
       int label=0;
       double labelp=0;
       for(int j=0;j<nClass;j++){

         auto& nowf=*(f_lists[j]);
         double now=exp(nowf.get(instance)-templist_2.get(instance))/templist.get(instance);
         probs.push_back(now);


         if(now>labelp){
           labelp=now;
           label=j;
         }

       }

       prediction.set(instance,Prediction(label,probs));


     });









     for(int j=0;j<nClass;j++){
       std::string fname="f_inlogit_class_"+std::to_string(j);
       z_instances.deleteAttrlist(fname);

     }
     for(int m=0;m<this->m_maxIterations;m++){
       for(int j=0;j<nClass;j++){
         z_instances.deleteAttrlist(gen_fit_name(j,m));
       }
     }
     z_instances.deleteAttrlist("temp_inlogit");
     z_instances.deleteAttrlist("temp_inlogit_2");

     //base::log_msg("no seg fault");
     return prediction;

}




}}
