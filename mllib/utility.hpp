/*
put generic functions in this file.
*/
#pragma once

#include<cstddef>
#include<string>
#include<vector>

namespace huksy{
    namespace mllib{
        typedef std::vector<double> vec_double;

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
        
        vec_double& operator /=(vec_double& a, const vec_double& b)
        {
            std::size_t a_size = a.size();
            for(std::size_t i = 0; i < a_size; i++)
            {
                a[i] = a[i]/b[i];
            }
            return a;
        }
        
        vec_double& operator *=(vec_double& a, const vec_double& b)
        {
            std::size_t a_size = a.size();
            for(std::size_t i = 0; i < a_size; i++)
            {
                a[i] = a[i]*b[i];
            }
            return a;
        }
    }
}
