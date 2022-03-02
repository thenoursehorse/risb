#pragma once
#include <vector>
#include <string>
#include <triqs/arrays.hpp>
#include <triqs/arrays/vector.hpp>
#include <triqs/arrays/matrix.hpp>
#include <triqs/operators/many_body_operator.hpp>
#include <triqs/hilbert_space/fundamental_operator_set.hpp>
#include <triqs/utility/first_include.hpp>

#include <triqs/gfs.hpp>

namespace risb {
  namespace functions {

    using dcomplex = std::complex<double>;
    using namespace triqs::arrays;
    using namespace triqs::hilbert_space;

    using namespace triqs::gfs;
    using namespace triqs::lattice;
  
    template <typename T> struct EigSystem {
      array<double,2> val;
      array<T,3> vec;
      array<T,3> vec_dag;
    };
    template struct EigSystem<double>;
    template struct EigSystem<dcomplex>;
 
    
    /**
     * @param fops The fundamental operator set of the local cluster
     * @return The fundamental operator set of the bath and the embedding space (contains local+bath)
     */
    std::pair<fundamental_operator_set, fundamental_operator_set> get_embedding_space(fundamental_operator_set const &fops_local);


    /**
     * @param R The renormalization matrix on each cluster as a list
     * @param lambda The correlation potential matrix on each cluster as a list
     * @param dispersion The dispersion relation indexed as disp[i,j,alpha,beta,k] (cluster,cluster,orbital,orbital,k)
     * @return eigensystem of h_qp (the f-fermions) calculated from the mean-field
     */
    template <typename H, typename T> EigSystem<T> get_h_qp(std::vector<matrix<H>> const &R, std::vector<matrix<double>> const &lambda, array<T,5> const &dispersion, double const mu = 0);
    
   
    /**
     * @param R The renormalization matrix on each cluster as a list
     * @param dispersion The dispersion relation indexed as disp[i,j,alpha,beta,k] (cluster,cluster,orbital,orbital,k)
     * @param eigensystem of h_qp (the f-fermions) calculated from the mean-field
     * @return disp * R * vec to calculate ke
     */
    template <typename H, typename T> array<T,3> get_disp_R(std::vector<matrix<H>> const &R, array<T,5> const &dispersion, EigSystem<T> const &h_qp);
    

    /**
     * @param disp_R the matrix elements of dispersion * R * vec where vec is the eigenvectors of H_qp (f-fermions) indexed as [a,b,k]
     * @param h_qp The eigensystem of h_qp
     * @param wks the integration weights at each k point indexed as [band,k]
     * @return ke the local kinetic energy of each cluster calculated from the mean-field
     */
    template <typename H> matrix<H> get_ke(array<H,3> const &disp_R, array<H,3> const &vec_dag, array<double,2> const &wks);
    

    /**
     * @param h_qp The eigensystem of h_qp
     * @param wks the integration weights at each k point indexed as [band,k]
     * @return pdensity the local density matrix of the f-fermions calculated from the mean-field
     */
    template <typename T> matrix<double> get_pdensity(array<T,3> const &vec, array<T,3> const &vec_dag, array<double,2> const &wks);
    

    /**
     * @param pdensity Local density matrix of f-fermions
     * @param ke The local kinetic energy matrix calculated from the mean-field
     * @return D the hyrbridization matrix of the local impurity problem
     */
    template <typename H> matrix<H> get_d(matrix_const_view<double> pdensity, matrix_const_view<H> ke);
    template <typename H> matrix<H> get_d(matrix<double> const &pdensity, matrix<H> const &ke) {
      return get_d(matrix_const_view<double>(pdensity), matrix_const_view<H>(ke));
    }
    
    
    /**
     * @param pdensity Local density matrix of f-fermions
     * @param R Renormalization matrix
     * @param lambda Correlation potential matrix
     * @param D Hybridization matrix of the impurity problem
     * @return lambda_c the bath levels of the local impurity problem
     */
    template <typename H> matrix<double> get_lambda_c(matrix_const_view<double> pdensity, matrix_const_view<H> R, matrix_const_view<double> lambda, matrix_const_view<H> D);
    template <typename H> matrix<double> get_lambda_c(matrix<double> const &pdensity, matrix<H> const &R, matrix<double> const &lambda, matrix<H> const &D) {
      return get_lambda_c(matrix_const_view<double>(pdensity), matrix_const_view<H>(R), matrix_const_view<double>(lambda), matrix_const_view<H>(D));
    }
    

    /**
     * @param R Renormalization matrix
     * @param D Hybridization matrix of the impurity problem
     * @param lambda_c Bath levels of the impurity problem
     * @param Nf Density matrix of the f-fermions of the impurity problem
     * @return lambda calculated from the impurity
     */
    template <typename H> matrix<double> get_lambda(matrix_const_view<H> R, matrix_const_view<H> D, matrix_const_view<double> lambda_c, matrix_const_view<double> Nf);
    template <typename H> matrix<double> get_lambda(matrix<H> const &R, matrix<H> const &D, matrix<double> const &lambda_c, matrix<double> const &Nf) {
      return get_lambda(matrix_const_view<H>(R), matrix_const_view<H>(D), matrix_const_view<double>(lambda_c), matrix_const_view<double>(Nf));
    }
    
    
    /**
     * @param Mcf Density matrix of the hybridization of the impurity problem
     * @param Nf Density matrix of the f-fermions of the impurity problem
     * @return R calculated from the impurity
     */
    template <typename H> matrix<H> get_r(matrix_const_view<H> Mcf, matrix_const_view<double> Nf);
    template <typename H> matrix<H> get_r(matrix<H> const &Mcf, matrix<double> const &Nf) {
      return get_r(matrix_const_view<H>(Mcf), matrix_const_view<double>(Nf));
    }
    
    
    /**
     * @param h_loc The local Hamiltonian of the cluster
     * @param D Hybridization matrix of the impurity problem
     * @param lambda_c Bath levels of the impurity problem
     * @param fops_local The fundamental operator set of the local cluster
     * @param fops_bath The fundamental operator set of the bath
     * @return h_emb the embedding Hamiltonian $H^{\mathrm{emb}}$
     */
    template <typename H> triqs::operators::many_body_operator_generic<H> get_h_emb(triqs::operators::many_body_operator_generic<H> const &h_loc, matrix<H> const &D, matrix<double> const &lambda_c, fundamental_operator_set const &fops_local, fundamental_operator_set const &fops_bath);
    

    /**
      * @param coeff A vector of coefficients indexed by s
      * @param hs A vector of basis matrices corresponding to coeff indexed by s
      * @return A dense matrix mat = sum_s coeff_s * hs_s
      */
    template <typename H> matrix<H> matrix_construct(std::vector<H> const &coeff, std::vector<matrix<dcomplex>> const &hs);


    /**
     * @param A The dense matrix
     * @param hs A vector of basis matrices corresponding to coeff indexed by s
     * @return coeff_s where coeff_s = Tr(hs_s^+ * A) / Tr(hs_s^+ * hs_s)
     */
    template <typename H> std::vector<H> coeff_construct(matrix<H> const &A, std::vector<matrix<dcomplex>> const &hs);
    
    
    /**
     * @param g_iw The local imaginary frequency Green's function
     * @param R The renormalization matrix
     * @return pdensity the local density matrix of the f-fermions calculated from the mean-field
     */
    template <typename H> matrix<double> get_pdensity_gf(gf_const_view<imfreq> g_z, matrix_const_view<H> R);
    template <typename H> matrix<double> get_pdensity_gf(gf_const_view<imfreq> g_z, matrix<H> const &R) {
      return get_pdensity_gf(g_z, matrix_const_view<H>(R));
    }


    /**
     * @param g_iw The local imaginary frequency Green's function
     * @param delta_iw The hybridization function
     * @param R The renormalization matrix
     * @return ke the local kinetic energy calcaulted from the mean-field
     */
    template <typename H> matrix<H> get_ke_gf(gf_const_view<imfreq> g_z, gf_const_view<imfreq> delta_z, matrix_const_view<H> R);
    template <typename H> matrix<H> get_ke_gf(gf_const_view<imfreq> g_z, gf_const_view<imfreq> delta_z, matrix<H> const &R) {
      return get_ke_gf(g_z, delta_z, matrix_const_view<H>(R));
    }
    
    
    /**
     * @param g0_iw The non-interacting part of the local imaginary frequency Green's function
     * @return delta_iw The hybridization function
     */
    gf<imfreq> get_delta_z(gf_const_view<imfreq> g0_z);


    /**
     * @param g_iw The local imaginary frequency Green's function (to get the correct structure)
     * @param R The renormalization matrix
     * @param lambda The correlation potential matrix matrix
     * @param mu The chemical potential
     * @return sigma_iw The local self-energy
     */
    template <typename H> gf<imfreq> get_sigma_z(gf_const_view<imfreq> g_z, matrix_const_view<H> R, matrix_const_view<double> lambda, double const mu = 0.0);
    template <typename H> gf<imfreq> get_sigma_z(gf_const_view<imfreq> g_z, matrix<H> const &R, matrix<double> const &lambda, double const mu = 0.0) {
      return get_sigma_z(g_z, matrix_const_view<H>(R), matrix_const_view<double>(lambda), mu);
    }
    
    // Explicit instantiations to expose to c++2py // FIXME this is incredibly ugly, can I do this a better way?
    
    //extern template class risb_solver<false,tetras>;
    //extern template class risb_solver<true,tetras>;
    
    //template <bool C, class K>
    //struct risb_solver_real {
    //  typedef risb_solver<C,K> type;
    //};
    //extern template class risb_solver_real<false,tetras>;
    
    // Real risb solver
    /**
      * @param R Renormalization matrix.
      * @param D Hybridization matrix of the impurity problem.
      * @param lambda_c Bath levels of the impurity problem.
      * @param Na Density matrix of the f-fermions of the impurity problem.
      * @return lambda
      */
    //struct risb_solver_real {
    //  typedef risb_solver<false,tetras> type;
    //};

    //struct risb_solver_complex {
    //  typedef risb_solver<true,tetras> type;
    //};
    

    
    //template EigSystem<double> risb_solver<false,tetras>::get_h_qp(std::vector<matrix<double>> const, std::vector<matrix<double>> const, array<double,5> const);

  } // namespace solver
} // namespace risb
