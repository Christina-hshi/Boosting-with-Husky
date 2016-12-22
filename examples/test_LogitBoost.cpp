
#include <mllib/Instances.hpp>
#include <mllib/DataReader.hpp>
#include <mllib/SimpleLinearRegression.hpp>
#include <mllib/LogitBoost.hpp>
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
  husky::mllib::svReader(instances,"hdfs:///1155032497/test_bi_class.csv",boost::char_separator<char>(","),mllib::LABEL_TYPE::CLASS);
  mllib::LogitBoost model=mllib::LogitBoost(new mllib::SimpleLinearRegression(),10,1);
  model.fit(instances);
auto& pred = model.predict(instances);
	if (husky::Context::get_global_tid() == 0) {
  for(int i=0;i<10;i++){
	//base::log_msg("slope in "+std::to_string(i)+" :" +std::to_string(((mllib::SimpleLinearRegression*)(model.get_baselearner(1,i)))->get_slope()));
	//base::log_msg("intercept in "+std::to_string(i)+" :" +std::to_string(((mllib::SimpleLinearRegression*)(model.get_baselearner(1,i)))->get_intercept()));
	//base::log_msg("selected in "+std::to_string(i)+" :" +std::to_string(((mllib::SimpleLinearRegression*)(model.get_baselearner(1,i)))->get_selected()));
}

}
list_execute(instances.enumerator(), [&pred](mllib::Instance& instance) {

base::log_msg("prediction: "+std::to_string(pred.get(instance).label));
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
