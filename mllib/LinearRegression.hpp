#include <mllib/Instances.hpp>
#include <mllib/BaseEstimator.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
typedef std::vector<double> vec_double;
typedef std::vector<std::vector<double>> matrix;
#include <stdexcept>


namespace husky{
  namespace mllib{
using namespace husky;

class PseudoObject {
   public:
    typedef int KeyT;
    int key;

    explicit PseudoObject(KeyT key) { this->key = key; }

    const int& id() const { return key; }
};

vec_double& operator+= (vec_double& va, const vec_double& vb) {

    int n = va.size();
    for (int i=0; i < n; i++) va[i] += vb[i];
    return va;
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


bool MatrixInversion(const matrix& input, matrix& output){
  int n=input.size();
  ublas::matrix<double> input2 (n,n);
  ublas::matrix<double> output2 (n,n);
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      input2(i,j)=input[i][j];
    }
  }

  InvertMatrix(input2,output2);


  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      output[i][j]=output2(i,j);
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



class LinearRegression : public mllib::Estimator{
private:
  vector_double param_vec;
public:
  void fit(const mllib::Instances& original_instances){
    Instances instances=original_instances;
    auto& pseudo_list=husky::ObjListFactory::create_objlist<PseudoObject>();
    pseudo_list.add_object(PseudoObject(0));
    globalize(pseudo_list);
    auto& ch1 =husky::ChannelFactory::create_push_combined_channel<matrix, husky::SumCombiner<matrix>>(pseudo_list, pseudo_list);
    auto& ch2 =husky::ChannelFactory::create_push_combined_channel<vec_double, husky::SumCombiner<vec_double>>(pseudo_list, pseudo_list);

    list_execute(instances.enumerator(), [](Instance& instance) {
        instance.X.push_back(1);
    });
    int dimension=instances.numAttributes+1;
    list_execute(instances.enumerator(), [&instances,&ch1,dimension](Instance& instance) {

      matrix A(dimension, vec_double(dimension));
      for(int a=0;a<dimension;a++){
        for(int b=0;b<dimension;b++){
          A[a][b]=instance.X[a]*instance.X[b];
        }
      }
      ch1.push(A,0);
    });
    list_execute(instances.enumerator(), [&instances,&ch2,dimension](Instance& instance) {
      vec_double B(dimension);
      for(int a=0;a<dimension;a++){
          B[a]=instance.X[a]*instances.get_y(instance);
      }
      ch2.push(B,0);
    });

    ch1.flush();
    ch2.flush();
    list_execute(pseudo_list, [&ch1,&ch2,this,dimension](PseudoObject& obj) {
        matrix XTX = ch1.get(obj);

        vec_double XTy=ch2.get(obj);

        matrix IA(dimension,vec_double(dimension));
        MatrixInversion(XTX,IA);
        vec_double output(dimension,0.0);
        MatrixVectormultiplication(IA,XTy,output);
        param_vec=output;
    });






  };
  void predict(const mllib::Instances& instances,vec_double& predictions, int label=-1){
    if(label!=-1)
      throw std::runtime_error("Linear regression can only do regression");

  };
  Estimator* clone(int seed=0){
    return new LinearRegression();

  };
}
}
}
