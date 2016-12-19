
#include <mllib/Instances.hpp>
#include <mllib/DataReader.hpp>
#include <mllib/SimpleLinearRegression.hpp>
std::string vec_to_str(vec_double v) {
    std::string str("");
    for (auto& x : v) {
        str += std::to_string(x);
        str += " ";
    }
    return str;
}





void testcompiling(){
  using namespace husky;
  mllib::Instances instances;
  husky::mllib::svReader(instances,"hdfs:///1155032497/mycpu2.tsv");
  mllib::SimpleLinearRegression model=mllib::SimpleLinearRegression();
  model.fit(instances);
  husky::base::log_msg("slope:"+std::to_string(model.get_slope()));
  husky::base::log_msg("intercept:"+std::to_string(model.get_intercept()));
  husky::base::log_msg("selected:"+std::to_string(model.get_selected()));
  auto& hello=instances.createAttrlist<double>("weight");
  list_execute(instances.enumerator(), [&hello](mllib::Instance& instance) {
   hello.set(instance,100);


  });
  model.fit(instances,"weight");
    husky::base::log_msg("slope:"+std::to_string(model.get_slope()));
  husky::base::log_msg("intercept:"+std::to_string(model.get_intercept()));
  husky::base::log_msg("selected:"+std::to_string(model.get_selected()));
  auto& prediction= model.predict(instances);
  list_execute(instances.enumerator(), [&prediction](mllib::Instance& instance) {
husky::base::log_msg("prediction:"+std::to_string(prediction.get(instance)));

  });

}




int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    if (husky::init_with_args(argc, argv,args)) {
        husky::run_job(testcompiling);
        return 0;
    }
    return 1;
}
