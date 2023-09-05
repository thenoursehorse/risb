#pragma once
#include <nda/nda.hpp>
#include <triqs/utility/is_complex.hpp>

namespace risb {

  using namespace nda;  
  
  template <class M> void set_zero(M &A) {
    #ifdef __GNUC__
    auto F = map([](__attribute__ ((unused)) auto i) { return  0; }); // because this warning is annoying
    #else
    auto F = map([](auto i) { return  0; });
    #endif
    A = F(A);
  }

  template <class M> void set_approx_zero(M &A, double tol = 1e-8) {
    auto F = map([tol](auto i) { return (abs(i) > tol) ? i : 0; });
    A = F(A);
  }


  // Minimize numerical rounding errors
  template<typename T, typename Iter> T fsum(Iter begin, Iter end) {
    T result = 0.0;
    T c = 0.0, y, t;
    for (; begin != end; ++begin)
    {
      y = *begin - c;
      t = result + y;
      c = (t - result) - y;
      result = t;
    }
    return result;
  }


  // Return the matrix square root of A
  /**
    * @param A Hermitian matrix to square root.
    * @return square root of the matrix A
    * @note Uses triqs::linalg::eigenelements, but likely better to use Shur decomposition or a power series.
    */
  template <typename M>
  matrix<typename M::value_type> matrix_sq(M const &B, double tol = 1e-12) {

    matrix<typename M::value_type> A = B; // FIXME this is a terrible hack because a matrix_exp gets passed (because my matrices are lazy)
                                          // and I can't figure out how to template this properly without eigenelements complaining
                                          // Also: for the same reason matrix_view doesn't work because no data has been created yet.
    
    bool const is_complex = triqs::is_complex<typename M::value_type>::value;
    using sq_scalar_t = std::conditional_t<is_complex, std::complex<double>, double>;

    // Check is symmetric/hermitian
    auto A_dag = dagger(A);
    foreach(A, [&A, &A_dag, tol](auto i, auto j) {
      if (std::abs((A(i, j) - A_dag(i, j))) > tol)
        TRIQS_RUNTIME_ERROR << "Matrix A = " << A << "must be Hermitian for matrix_sq !";
    });

    auto eig = linalg::eigenelements(A); // assumes A is Hermitian
    
    // Check is positive definite
    //foreach(eig.first, [&eig,&A, tol](auto i) {
    //  if ((eig.first(i) < 0.0) && (std::abs(eig.first(i)) > tol))
    //    TRIQS_RUNTIME_ERROR << "Matrix A = " << A << "must be positive definite for matrix_sq !";
    //});

    matrix<sq_scalar_t> D_sq(get_shape(A));
    auto eig_sqrt = sqrt(eig.first);
    assign_foreach(D_sq, [&eig_sqrt](auto i, auto j) {
        return (i == j ? eig_sqrt(i) : 0);
    });
    if (is_complex) return eig.second.transpose() * D_sq * dagger(eig.second.transpose());
    else return real(eig.second.transpose() * D_sq * dagger(eig.second.transpose()));
  }
  
  
  
  // Return the pseudo inverse of A, $pinv(A) = V S^{-1} U^T$.
  /**
    * @param A Matrix to find pseudo inverse of.
    * @return B the pseudo inverse.
    * @note Uses gelss in LAPACK which finds the lienar least squares between two matrices A and B. If B is the
    * identity then the gelss modifes B such that it is the pseudo inverse of A by the SVD method.
    * See gelss_cache class in triqs/arrays/blas_lapack/gelss.hpp which explicitly computes pinv(A).
    * See: https://zhuanlan.zhihu.com/p/84188411
    * See: https://software.intel.com/content/www/us/en/develop/articles/implement-pseudoinverse-of-a-matrix-by-intel-mkl.html
    */
  template <typename T> 
  matrix<typename T::value_type> pseudo_inverse(T const &A) {
    int M    = A.shape()[0];
    int N    = A.shape()[1];

    // Output B of identity to gess is the pseudo inverse of A
    matrix<typename T::value_type> B(N,N);
    set_zero(B);
    B() = 1.0;

    //int NRHS = B.shape()[1];

    auto S = vector<double>(std::min(M, N));

    int i;
    auto info = lapack::gelss(A, B, S, 1e-18, i); // rcond = 1e-18 does what exactly?
    if (info != 0) TRIQS_RUNTIME_ERROR << "psuedo_inverse error: gelss info = " << info;
    return B;
    //return B(range(N), range(NRHS)); // ??
  }
  //template matrix<double> pseudo_inverse(matrix<double> const &);


} // namespace risb
