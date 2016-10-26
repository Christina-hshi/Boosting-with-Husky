//Linear regression

#include <string>
#include <vector>
#include <limits>

#include "boost/tokenizer.hpp"
#include "core/engine.hpp"
#include "io/input/hdfs_line_inputformat.hpp"
#include "lib/aggregator_factory.hpp"

typedef std::vector<double> vec_double;

class XYNode{
public:
    typedef int KeyT;
    KeyT key;
    XYNode(){}
    explicit XYNode(const KeyT& k) : key(k) {}

    virtual KeyT const & id() const { return key;}

    vec_double X;
    double y;
};

std::string vec_to_str(vec_double v) {
    std::string str("");
    for (auto& x : v) {
        str += std::to_string(x);
        str += " ";
    }
    return str;
}

// Inner Product
double operator* (const vec_double& va, const vec_double& vb) {
    int n = va.size();
    double sum = 0.0;
    for (int i=0; i < n; i++) sum += va[i] * vb[i];
    return sum;
}

// Vector Addition
vec_double& operator+= (vec_double& va, const vec_double& vb) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] += vb[i];
    return va;
}

vec_double& operator-= (vec_double& va, const vec_double& vb) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] -= vb[i];
    return va;
}

// Vector multiplication
vec_double& operator*= (vec_double& va, const double& c) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] *= c;
    return va;
}

// Vector division
vec_double& operator/= (vec_double& va, const double& c) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] /= c;
    return va;
}

void max_abs_vec(vec_double& va, const vec_double& vb) {
    int n = va.size();
    if (va.size() != vb.size())
    {
    	std::cout<<"va.size= "<<va.size()<<" vb.size: "<<vb.size()<<std::endl;
	
	/*
	for (int x = 0; x < va.size(); x++)
	{
	    std::cout<<va[x]<<" ";
	}
	std::cout<<std::endl;
	for (int x = 0; x < vb.size(); x++)
	{
	    std::cout<<vb[x]<<" ";
	}
	std::cout<<std::endl;
	*/
    }
    for (int i=0; i < n; i++) va[i] = std::max(abs(va[i]), abs(vb[i]));
}

void max_vec(vec_double& va, const vec_double& vb) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] = std::max(va[i], vb[i]);
}

void min_vec(vec_double& va, const vec_double& vb) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] = std::min(va[i], vb[i]);
}

// Serialization and deserialization
husky::BinStream& serialization_vec_double(husky::BinStream& stream, const vec_double& u) {
	stream << u.size();
	for (double x : u)
	{
		stream << x;
	}
	return stream;
}
husky::BinStream& deserialization_vec_double(husky::BinStream& stream, vec_double& u) {
	size_t n;
	stream >> n;
	u.clear();
	double x;
	while(n--)
	{
		stream >> x;
		u.push_back(x);
	}
	return stream;
}

void linear_regression() {
    //auto & worker = husky::Context::get_worker<husky::BaseWorker>();
    
    auto & xynode_list = husky::ObjListFactory::create_objlist<XYNode>();

    auto & empty_list = husky::ObjListFactory::create_objlist<XYNode>();
    
    husky::lib::Aggregator<int> sum_examples(0, [](int & a, const int & b) { a += b;});
    //sum_examples.to_reset_each_iter();

    // Some machine may not handle the input data
    // And thus they dont know the true num_features.
    husky::lib::Aggregator<int> global_num_features(1, [](int & a, const int & b) { a = b;});
    
    //sum_examples.to_reset_each_iter();
    
	int num_features = 1;
	int num_worker_examples = 0;
	auto parse_feature_label = [&](boost::string_ref & chunk) {

		// seperate the string
		boost::char_separator<char> sep(" \t");
		boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);
		XYNode this_xynode;
		this_xynode.key = num_worker_examples++;

		for (auto& w : tok) {
			this_xynode.X.push_back(std::stod(w));
		}
		this_xynode.y = this_xynode.X.back();
		this_xynode.X.pop_back();
		// The intercept term
		this_xynode.X.push_back(1.0);

		num_features = (this_xynode.X).size();
		global_num_features.update(num_features);

		xynode_list.add_object(this_xynode);
		sum_examples.update(1);
	};

    auto& ac = husky::lib::AggregatorFactory::get_channel();
    husky::io::HDFSLineInputFormat infmt;
    infmt.set_input(husky::Context::get_param("input"));
    husky::load(infmt, {&ac}, parse_feature_label);
    
    //husky::base::log_msg("sum_examples: " + std::to_string(sum_examples.get_value()));
    //husky::base::log_msg("num_worker_examples: " + std::to_string(num_worker_examples));

    int num_examples = sum_examples.get_value();
    num_features = global_num_features.get_value();

    // normailze the init_pv
    vec_double inti_vec(num_features, 0.0);
    
    // parameter vector, Aggreator
    husky::lib::Aggregator<vec_double> param_vec(inti_vec,
            [](vec_double & va, const vec_double & vb) { va += vb;},
			[&](vec_double & v){v = std::move(vec_double(num_features, 0));},
			deserialization_vec_double,
			serialization_vec_double);

	/*
    // Aggreate the scaling vector of each machine
    // Even all values abs < 1, they still need to be scaling to avoid the error is too small to calculate.
    //std::cout<<"num_features: "<<num_features<<std::endl;
    vec_double max_X(num_features, 0);
    //std::cout<<vec_to_str(max_X)<<std::endl;

    //vec_double min_X(num_features, std::numeric_limits<double>::min());
    double max_y = 0; //std::numeric_limits<double>::min();
    //double min_y = std::numeric_limits<double>::min();

    husky::lib::Aggregator<vec_double> scaling_X(max_X, 
			max_abs_vec,
			[&](vec_double & v){v = std::move(vec_double(num_features, 0));},
			deserialization_vec_double,
			serialization_vec_double);


    husky::lib::Aggregator<double> scaling_y(max_y,
            [](double & a, const double & b) {
            a = std::max(abs(a), abs(b));});


    // normalize the value in each instace
    //auto& ch = husky::lib::AggregatorFactory::get_channel();
    husky::list_execute(xynode_list, {}, {&ac}, [&](XYNode& this_xy) {
		//std::cout<<vec_to_str(max_X)<<std::endl;
		
		//std::cout<<"this_xy.X "<<vec_to_str(this_xy.X)<<std::endl;
        max_abs_vec(max_X, this_xy.X);
		
		//std::cout<<"after first direct all"<<std::endl;
        max_y = std::max(abs(max_y), abs(this_xy.y));
		
		//std::cout<<vec_to_str(max_X)<<std::endl;
		scaling_X.update(max_X);
		//std::cout<<scaling_X.get_value().size()<<std::endl;
		scaling_y.update(max_y);
	
        //std::cout<<max_y<<std::endl;

    });

	//std::cout<<"after list execute"<<std::endl;
    //std::cout<<scaling_y.get_value()<<std::endl;
    
    // reduce and get the result
    max_X = scaling_X.get_value();
    max_y = scaling_y.get_value();

    // complete the scaling for each instance
    husky::list_execute(xynode_list, {}, {&ac}, [&](XYNode& this_xy) {
        int n = this_xy.X.size();
        for (int i=0; i < n; i++) {
            if (max_X[i] > 0.0) this_xy.X[i] /= max_X[i];
        }
        if (max_y > 0.0) this_xy.y /= max_y;
    });

	std::cout<<"after scaling"<<std::endl;

	*/
    double alpha = std::stod(husky::Context::get_param("alpha"));
    int num_iter = std::stoi(husky::Context::get_param("num_iter"));

    // Evaluation of the cost, sum of all thread
    husky::lib::Aggregator<double> global_cost(0.0,
            [](double & a, const double & b) {a += b;},
			[](double& a){a = 0.0;});
    global_cost.to_reset_each_iter();

    // Parallel Stochastic Gradient Descent
    auto SGD = [&](){
        // get the parameter vector (reference)
		husky::lib::AggregatorFactory::sync();
        
		//husky::base::log_msg("after sync " + std::to_string(global_cost.get_value()));

		vec_double & old_pv = param_vec.get_value();
        vec_double pv = vec_double(num_features, 0);//param_vec.get_value();
		
		//std::cout<<"before list execute: "<<global_cost.get_value()<<std::endl;

        husky::list_execute(xynode_list, {}, {&ac}, [&](XYNode& this_xy) {
            double error = this_xy.y - (old_pv * this_xy.X);
			
            
			global_cost.update(error*error);
            for (int i=0; i < num_features; i++) {
                pv[i] += alpha * error * this_xy.X[i];
            }
			//std::cout<<"inside list execute: "<<global_cost.get_value()<<std::endl;
        });
	
		//std::cout<<"pv: "<<vec_to_str(pv)<<std::endl;
		//std::cout<<"after list execute: "<<global_cost.get_value()<<std::endl;
        //pv -= old_pv;
		
		//husky::base::log_msg("pv before: " + vec_to_str(pv));
		//husky::base::log_msg(std::to_string(static_cast<double>(num_worker_examples) / num_examples) + " " + std::to_string((num_worker_examples) / num_examples));
        pv /= num_examples;
		//husky::base::log_msg("pv after: " + vec_to_str(pv));
		
		//std::cout<<"num_worker_examples: "<<num_worker_examples<<" num_example_global: "<<num_examples<<std::endl;

		//husky::base::log_msg("before sync " + std::to_string(global_cost.get_value()));
		param_vec.update(pv);

		//husky::base::log_msg("after SGD pv: " + vec_to_str(pv));
		//husky::lib::AggregatorFactory::sync();

		//std::cout<<"param_vec: "<<vec_to_str(old_pv)<<std::endl;
    };

    // easy for extend it to execute other GD algo.
    auto iter_exec = [&](auto GD_algo){
        for (int iter_turn=0; iter_turn < num_iter; iter_turn++) {
            GD_algo();

            if (husky::Context::get_global_tid() == 0) {
                double sum_cost = global_cost.get_value();
                husky::base::log_msg("The error in iter "
                        + std::to_string(iter_turn)
                        + ": "
                        + std::to_string(sum_cost/num_examples));
            }

            // Empty list_excute to make sure the agg update is effective
            // husky::list_execute(empty_list, [&](XYNode& this_xy){});
        }

		husky::lib::AggregatorFactory::sync();
        // Show the result
		
        if (husky::Context::get_global_tid() == 0) {
            /*
			husky::base::log_msg("#example: "+std::to_string(num_examples));
            husky::base::log_msg("scaling features vector: ");
            husky::base::log_msg(vec_to_str(max_X));
            husky::base::log_msg("scaling of y: ");
            husky::base::log_msg(std::to_string(max_y));
            */
			vec_double pv = param_vec.get_value();
            husky::base::log_msg("Parameters: " + vec_to_str(pv));
			/*
			vec_double unscaled_params(num_features, 0);
			for (int i = 0; i < num_features; i++)
			{
				unscaled_params[i] = pv[i] * max_X[i]; 
			}
            husky::base::log_msg("Unscaled parameters: " + vec_to_str(unscaled_params));
			*/
			/*
			int iter_param = 0;
            for (auto& par : pv) {
                husky::base::log_msg("Param of "
                        + std::to_string(iter_param+1)
                        + ": "
                        + std::to_string(par) );
                iter_param++;
            }
			*/
        }
		
    };

    iter_exec(SGD);
	
    //husky::Context::free_worker<husky::BaseWorker>();
}

int main(int argc, char ** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");
    args.push_back("alpha");
    args.push_back("num_iter");

    if (husky::init_with_args(argc, argv, args)) {
	husky::run_job(linear_regression);
	return 0;
    }
    return 1;
}

