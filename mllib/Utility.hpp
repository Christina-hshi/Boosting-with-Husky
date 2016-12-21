/*
put generic functions in this file.
vector operation::
* dot product
*= inplace scalar or elementwise multiplication
/= inplace elementwise division
+= inplace vector addition
-= inplace vector substraction

matrix operation::
+= inplace matrix addition
/= usage: matrix /= cons
*= usage: matrix *= cons
*/
#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <limits>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
typedef std::vector<double> vec_double;
typedef std::vector<std::vector<double>> matrix;
typedef std::vector<std::vector<double>> matrix_double;

inline std::string vec_to_str(const vec_double& v) {
    std::string str("");
    for (auto& x : v) {
        str += std::to_string(x);
        str += " ";
    }
    return str;
}

inline std::string matrix_to_str(matrix_double& m){
    std::stringstream ss;
    ss.precision(2);
    for(auto& row : m){
        for(auto& ele : row){
            ss<< ele <<" ";
        }
        ss << "\n";
    }
    return ss.str();
}

inline matrix_double& operator+= (matrix_double& m1, const matrix_double& m2){
    //for debug 
    if(m1.size() != m2.size()){
        std::cout<<"different size!"<<std::endl;
        std::cout<<"m1 "<<m1.size()<<" ";
        for(size_t x = 0; x < m1.size(); x++){
            std::cout<<m1[x].size()<<" ";
        }
        std::cout<<std::endl<<"m2 "<<m2.size()<<" ";
        for(size_t x = 0; x < m2.size(); x++){
            std::cout<<m2[x].size()<<" ";
        }
        std::cout<<std::endl;
    }
    //end
    
    for(size_t r = 0; r < m1.size(); r++){
        for(size_t c = 0; c < m1[r].size(); c++){
            m1[r][c] += m2[r][c];
        }
    }
    return m1;
}

inline matrix_double& operator/= (matrix_double& m, const double coe){
    for(size_t r = 0; r < m.size(); r++){
        for(size_t c = 0; c < m[r].size(); c++){
            m[r][c] /= coe;
        }
    }
    return m;
}
inline matrix_double& operator*= (matrix_double& m, const double coe){
    for(size_t r = 0; r < m.size(); r++){
        for(size_t c = 0; c < m[r].size(); c++){
            m[r][c] *= coe;
        }
    }
    return m;
}

inline matrix_double operator*(const matrix_double& m, const double coe){
    matrix_double result;
    for(size_t r = 0; r < m.size(); r++){
        result.push_back(vec_double(m[r].size(), 0));
        for(size_t c = 0; c < m[r].size(); c++){
            result[r][c] = m[r][c] * coe;
        }
    }
    return result;
}

/*
inline matrix& operator+= (matrix& ma, const matrix& mb) {

    int m = ma.size();
    int n = ma[0].size();
    for (int i=0; i < m; i++)
      for (int j=0; j< n; j++)
        ma[i][j] += mb[i][j];
    return ma;
}
*/

inline double sum(const vec_double& v){
  double result = 0;
  for(auto ele : v){
    result += ele;
  }
  return result;
}
// Inner Product : tolerance the case where length(va) <= length(vb)
inline double operator* (const vec_double& va, const vec_double& vb) {
    int n = va.size();
    double sum = 0.0;
    for (int i=0; i < n; i++) sum += va[i] * vb[i];
    return sum;
}

// Vector Addition
inline vec_double& operator+= (vec_double& va, const vec_double& vb) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] += vb[i];
    return va;
}
inline vec_double operator+ (const vec_double& va, const vec_double& vb) {
    int n = va.size();
    vec_double result(n, 0.0);
    for (int i=0; i < n; i++) result[i] = va[i] + vb[i];
    return result;
}

inline vec_double& operator-= (vec_double& va, const vec_double& vb) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] -= vb[i];
    return va;
}

// Scalar multiplication and division
inline vec_double& operator*= (vec_double& va, const double& c) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] *= c;
    return va;
}
inline vec_double operator* (const double& c, const vec_double& va) {
    int n = va.size();
    vec_double result(n, 0.0);
    for (int i=0; i < n; i++) result[i] = va[i] * c;
    return result;
}
inline vec_double& operator/= (vec_double& va, const double& c) {
    int n = va.size();
    for (int i=0; i < n; i++) va[i] /= c;
    return va;
}
// elementwise division
inline vec_double& operator /=(vec_double& a, const vec_double& b)
{
    std::size_t a_size = a.size();
    for(std::size_t i = 0; i < a_size; i++)
    {
        a[i] = a[i]/b[i];
    }
    return a;
}
// elementwise multiplcation
inline vec_double& operator *=(vec_double& a, const vec_double& b)
{
    std::size_t a_size = a.size();
    for(std::size_t i = 0; i < a_size; i++)
    {
        a[i] = a[i]*b[i];
    }
    return a;
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


inline bool MatrixInversion(matrix& input){
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

inline void MatrixVectormultiplication(const matrix& A,const vec_double& B,vec_double& output){
  int m=A.size();
  int n=A[0].size();
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      output[i]+=A[i][j]*B[j];
    }
  }


}
