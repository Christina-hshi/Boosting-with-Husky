//(XTX)-1XTY
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <string>
#include <vector>
#include <core/engine.hpp>
#include "io/input/hdfs_line_inputformat.hpp"
#include "boost/tokenizer.hpp"
typedef std::vector<double> fvec;
typedef std::vector<std::vector<double>> matrix;



class PIObject {
   public:
    typedef int KeyT;
    int key;

    explicit PIObject(KeyT key) { this->key = key; }

    const int& id() const { return key; }
};

std::string vec_to_str(fvec v) {
    std::string str("");
    for (auto& x : v) {
        str += std::to_string(x);
        str += " ";
    }
    return str;
}


fvec& operator+= (fvec& va, const fvec& vb) {

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

void MatrixVectormultiplication(const matrix& A,const fvec& B,fvec& output){
  int m=A.size();
  int n=A[0].size();
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      output[i]+=A[i][j]*B[j];
    }
  }


}


void linear_regression() {
  husky::io::HDFSLineInputFormat infmt;
  infmt.set_input(husky::Context::get_param("input"));
  husky::ObjList<PIObject> pi_list;
  std::vector<fvec> X;
  std::vector<double> y;

  auto parse_wc = [&](boost::string_ref& chunk) {
      if (chunk.size() == 0)
          return;
      fvec here;
      boost::char_separator<char> sep(" \t");
      boost::tokenizer<boost::char_separator<char>> tok(chunk, sep);

      for (auto& w : tok) {
          here.push_back(std::stod(w));
      }
      y.push_back(here.back());
      here.pop_back();
      // The intercept term
      here.push_back(1.0);
      X.push_back(here);

  };

  husky::load(infmt, parse_wc);

  std::cout << "helloworld" << std::endl;


  auto& ch1 =husky::ChannelFactory::create_push_combined_channel<matrix, husky::SumCombiner<matrix>>(pi_list, pi_list);
  auto& ch2 =husky::ChannelFactory::create_push_combined_channel<fvec, husky::SumCombiner<fvec>>(pi_list, pi_list);
  std::cout << "helloworld" << std::endl;
  int dimension=0;
  if (X.size()>0)
    dimension=X[0].size();
  std::cout << "dimension: "<<dimension<<" numberofexamples"<<X.size() << std::endl;
  matrix A(dimension, fvec(dimension));
  for(int i=0;i<X.size();i++){
    fvec now=X[i];
    for(int a=0;a<dimension;a++){
      for(int b=0;b<dimension;b++){
        A[a][b]+=now[a]*now[b];
      }
    }

  }
  std::cout << "helloworld2" << std::endl;
  if(dimension!=0)
    ch1.push(A,0);
  std::cout << "helloworld3" << std::endl;
  fvec B(dimension);
  for(int i=0;i<X.size();i++){
    fvec now=X[i];
    for(int a=0;a<dimension;a++){
        B[a]+=now[a]*y[i];
    }
  }
  if(dimension!=0)
    ch2.push(B,0);


  ch1.flush();
  ch2.flush();
  list_execute(pi_list, [&ch1,&ch2](PIObject& obj) {
      matrix XTX = ch1.get(obj);

      fvec XTy=ch2.get(obj);

      int dimension=XTX.size();
      matrix IA(dimension,fvec(dimension));
      MatrixInversion(XTX,IA);
      fvec output(dimension,0.0);
      MatrixVectormultiplication(IA,XTy,output);
      husky::base::log_msg(vec_to_str(output));
  });






}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.push_back("hdfs_namenode");
    args.push_back("hdfs_namenode_port");
    args.push_back("input");

    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(linear_regression);
        return 0;
    }
    return 1;
}
