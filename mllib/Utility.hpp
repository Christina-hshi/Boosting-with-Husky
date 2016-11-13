/*
put generic functions in this file.
vector operation::
. dot product
*= inplace scalar or elementwise multiplication
/= inplace elementwise division
+= inplace vector addition
-= inplace vector substraction

matrix operation::
+= inplace matrix addition

*/
#pragma once

#include<cstddef>
#include<string>
#include<vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace husky{
    namespace mllib{
        typedef std::vector<double> vec_double;
        typedef std::vector<std::vector<double>> matrix;

        std::string vec_to_str(vec_double v) {
            std::string str("");
            for (auto& x : v) {
                str += std::to_string(x);
                str += " ";
            }
            return str;
        }

        // Inner Product changed the operator from * to . to distinguish dot product from elementwise multiplication
        double operator. (const vec_double& va, const vec_double& vb) {
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

        // Scalar multiplication
        vec_double& operator*= (vec_double& va, const double& c) {
            int n = va.size();
            for (int i=0; i < n; i++) va[i] *= c;
            return va;
        }
        // elementwise division
        vec_double& operator /=(vec_double& a, const vec_double& b)
        {
            std::size_t a_size = a.size();
            for(std::size_t i = 0; i < a_size; i++)
            {
                a[i] = a[i]/b[i];
            }
            return a;
        }
        // elementwise multiplcation
        vec_double& operator *=(vec_double& a, const vec_double& b)
        {
            std::size_t a_size = a.size();
            for(std::size_t i = 0; i < a_size; i++)
            {
                a[i] = a[i]*b[i];
            }
            return a;
        }


        matrix& operator+= (matrix& ma, const matrix& mb) {

            int m = ma.size();
            int n = ma[0].size();
            for (int i=0; i < m; i++)
              for (int j=0; j< n; j++)
                ma[i][j] += mb[i][j];
            return ma;
        }
        namespace ublas = boost::numeric::ublas;
        /* Matrix inversion routine.
           Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
        template<class T>
        bool InvertMatrix (const ublas::matrix<T>& input, ublas::matrix<T>& inverse) {

         typedef ublas::permutation_matrix<std::size_t> pmatrix;
         // create a working copy of the input
         ublas::matrix<T> A(input);
         // create a permutation matrix for the LU-factorization
         pmatrix pm(A.size1());
         // perform LU-factorization
         int res = lu_factorize(A,pm);
               if( res != 0 ) return false;
         // create identity matrix of "inverse"
         inverse.assign(ublas::identity_matrix<T>(A.size1()));
         // backsubstitute to get the inverse
         lu_substitute(A, pm, inverse);
         return true;
        }


        bool MatrixInversion(matrix& input){
          int n=input.size();
          ublas::matrix<double> input2 (n,n);
          ublas::matrix<double> output2 (n,n);
          for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
              input2(i,j)=input[i][j];
            }
          }

          if(!InvertMatrix(input2,output2))
            return false;


          for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
              input[i][j]=output2(i,j);
            }
          }
        }

        void MatrixVectormultiplication(const matrix& A,const vec_double& B,vec_double& output){
          int m=A.size();
          int n=A[0].size();
          for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
              output[i]+=A[i][j]*B[j];
            }
          }


        }
    }
}
