
#include <mllib/Instances.hpp>
#include <mllib/DataReader.hpp>

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
  base::log_msg("checking X y and class,key");
  list_execute(instances.enumerator(), [&instances](mllib::Instance& instance) {
    std::vector<double> here=instance.X;

    here.push_back(instances.get_y(instance));
    here.push_back(instances.get_class(instance));
    base::log_msg(vec_to_str(here)+" " +instance.key);

  });
  base::log_msg("...");
  base::log_msg("...");
  base::log_msg("...");
  base::log_msg("...");
  base::log_msg("checking load balancing");
  base::log_msg(std::to_string(instances.enumerator().get_size()));
  base::log_msg("...");
  base::log_msg("...");
  base::log_msg("...");
  base::log_msg("...");
  base::log_msg("checking statistics");
  base::log_msg("numClasses: "+std::to_string(instances.numClasses));
  base::log_msg("numAttributes: "+std::to_string(instances.numAttributes));
  base::log_msg("numInstances: "+std::to_string(instances.numInstances));


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
