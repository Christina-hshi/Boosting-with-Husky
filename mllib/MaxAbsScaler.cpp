/*
Edit by Christina
*/
#include <algorithm>
#include "mllib/MaxAbsScaler.hpp"

namespace husky{
    namespace mllib{
        void max_abs_vec(vec_double& va, const vec_double& vb) {
            int n = va.size();
            for (int i=0; i < n; i++) va[i] = std::max(abs(va[i]), abs(vb[i]));
        }

        void MaxAbsScaler::fit(const Instances& instances){
            vec_double init_value = vec_double(instances.numAttributes, 0.0);
            husky::lib::Aggregator<vec_double> scaling_X(init_value, max_abs_vec);

            auto& ac = husky::lib::AggregatorFactory::get_channel();
            
            list_execute(instances.enumerator(), {},{&ac},[&](Instance& instance) {
                    scaling_X.update(instance.X);
                    });
            husky::lib::AggregatorFactory::sync();

            parameters.clear();
            parameters = scaling_X.get_value();
            return;
        }

        void MaxAbsScaler::fit_transform(Instances& instances){
            fit(instances);

            list_execute(instances.enumerator(), [this](Instance& instance) {
                    instance.X /= parameters;
                    });  
        }

        void MaxAbsScaler::transform(Instances& instances){
            list_execute(instances.enumerator(), [this](Instance& instance) {
                    instance.X /= parameters;
                    });  
        }
        
        void MaxAbsScaler::inverse_transfrom(Instances& instances){
            list_execute(instances.enumerator(), [this](Instance& instance) {
                    instance.X *= parameters;
                    }); 
        }
    }
}
