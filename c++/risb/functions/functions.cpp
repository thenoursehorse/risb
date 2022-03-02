#include "./functions.hpp"
#include <triqs/arrays/linalg/eigenelements.hpp>
#include <triqs/arrays/blas_lapack/gelss.hpp>
#include <triqs/utility/is_complex.hpp>
#include "../common_functions.hpp"

namespace risb {
  namespace functions {

    using namespace triqs::arrays;

    std::pair<fundamental_operator_set, fundamental_operator_set> get_embedding_space(fundamental_operator_set const &fops_local) {

      fundamental_operator_set fops_emb(fops_local);
      fundamental_operator_set fops_bath;

      for (auto const &o : fops_local) { 
        auto bath_index = o.index;
        //bath_index.push_back("bath"); // is there a better way to label this? I can't access this in python
        bath_index.back() = std::to_string(bath_index.back()) + "_bath";
        fops_bath.insert(bath_index);
      }
      for (auto const &o : fops_bath) {
        fops_emb.insert(o.index);
      }

      return std::pair<fundamental_operator_set, fundamental_operator_set> (fops_bath, fops_emb);
    }

    // -----------------------------------------------------------------
    
    // Returns the eigenenergies/eigenvectors of the f-fermions from the mean-field 
    template <typename H, typename T>  // FIXME is there a way I can get eks_scalar_t as the return type without templating a bool Complex? look into conditional types
    auto get_h_qp(std::vector<matrix<H>> const &R, std::vector<matrix<double>> const &lambda, array<T,5> const &dispersion, double const mu)->EigSystem<T> {

      bool const is_complex = triqs::is_complex<T>::value || triqs::is_complex<H>::value;
      using eks_scalar_t = std::conditional_t<is_complex, std::complex<double>, double>;
      
      triqs::clef::placeholder<0> i_;
      triqs::clef::placeholder<1> j_;
      auto _ = range();
      
      auto n_qp = first_dim(dispersion);
      auto dim = third_dim(dispersion);
      auto mesh_num = fifth_dim(dispersion);
      
      array<eks_scalar_t,3> h_qp(n_qp * dim, n_qp * dim, mesh_num); // FIXME these dimensions are not very general
      array<double,2> h_qp_eig(first_dim(h_qp), third_dim(h_qp));
      array<eks_scalar_t,3> h_qp_vec(first_dim(h_qp), second_dim(h_qp), third_dim(h_qp));
      array<eks_scalar_t,3> h_qp_vec_dag(first_dim(h_qp), second_dim(h_qp), third_dim(h_qp));

      // Set the quasiparticle Hamiltonian to diagononalise
      for (auto const i : range(n_qp)) {
        matrix<double> mu_matrix(first_dim(lambda[i]), first_dim(lambda[i]));
        mu_matrix(i_,j_) << (i_ == j_) * mu;
        for (auto const j : range(n_qp)) {
          for (auto const k : range(mesh_num)) {
            // Take a slice of the relevant part of the dispersion relation and H_qp
            matrix_const_view<T> dispersion_slice = dispersion(i,j,_,_,k);
            matrix_view<eks_scalar_t> h_qp_slice = h_qp(range(i * dim, (i + 1) * dim),
                                                        range(j * dim, (j + 1) * dim), k);
            h_qp_slice = R[i] * dispersion_slice * dagger(R[j]);
            if (i == j) {
              h_qp_slice += lambda[i] - mu_matrix;
            }
          }
        }
      }

      // Diagonalise
      for (auto const k : range(mesh_num)) {
        matrix_const_view<eks_scalar_t> h_qp_slice = h_qp(_,_,k);
        auto eig = linalg::eigenelements(h_qp_slice);
        h_qp_eig(_,k) = eig.first;
        // NOTE: TRIQS eigensolver has the right eigenvectors as rows, but I want to think of them
        // as columns so I transpose, so the maths makes sense and I can multiply
        h_qp_vec(_,_,k) = eig.second.transpose();
        matrix_const_view<eks_scalar_t> h_qp_vec_slice = h_qp_vec(_,_,k);
        h_qp_vec_dag(_,_,k) = dagger(h_qp_vec_slice);
        //h_qp_vec_dag(_,_,k) = dagger(eig.second.transpose());
      }
 
      return EigSystem<eks_scalar_t> {h_qp_eig, h_qp_vec, h_qp_vec_dag};
    }
    template EigSystem<double> get_h_qp(std::vector<matrix<double>> const &, std::vector<matrix<double>> const &, array<double,5> const &, double);
    template EigSystem<dcomplex> get_h_qp(std::vector<matrix<double>> const &, std::vector<matrix<double>> const &, array<dcomplex,5> const &, double);
    
    // -----------------------------------------------------------------
    
    template <typename H, typename T> // as above with the conditional type
    auto get_disp_R(std::vector<matrix<H>> const &R, array<T,5> const &dispersion, EigSystem<T> const &h_qp)->array<T,3> {
      
      //bool const is_complex = triqs::is_complex<T>::value || triqs::is_complex<H>::value || triqs::is_complex<E>::value;
      bool const is_complex = triqs::is_complex<T>::value || triqs::is_complex<H>::value;
      using eks_scalar_t = std::conditional_t<is_complex, std::complex<double>, double>;
      
      auto _ = range();
      
      auto n_qp = first_dim(dispersion);
      auto dim = third_dim(dispersion); // FIXME this should be more general and be inside the some in some way
      auto mesh_num = fifth_dim(dispersion);
      auto num_bands = first_dim(h_qp.vec);

      array<eks_scalar_t,3> disp_R(num_bands, num_bands, mesh_num);

      // Build e_k R^+
      for (auto const i : range(n_qp)) {
        for (auto const j : range(n_qp)) {
          for (auto const k : range(mesh_num)) {
            matrix_const_view<eks_scalar_t> dispersion_slice = dispersion(i,j,_,_,k);
            matrix_view<eks_scalar_t> disp_R_slice = disp_R(range(i * dim, (i + 1) * dim),
                                                            range(j * dim, (j + 1) * dim), k);
            disp_R_slice = dispersion_slice * dagger(R[j]); 
          }
        }
      }

      // NOTE: instead of summing over p I can right multiply disp_R with h_qp.vec
      //       i'm not sure which is faster but I doubt it matters 
      for (auto const k : range(mesh_num)) {
        matrix_const_view<eks_scalar_t> h_qp_vec_slice = h_qp.vec(_,_,k);
        matrix_view<eks_scalar_t> disp_R_slice = disp_R(_,_,k);
        disp_R_slice = disp_R_slice * h_qp_vec_slice;
      }
     return disp_R; 
    }
    template array<double,3> get_disp_R(std::vector<matrix<double>> const &, array<double,5> const &, EigSystem<double> const &);
    template array<dcomplex,3> get_disp_R(std::vector<matrix<double>> const &, array<dcomplex,5> const &, EigSystem<dcomplex> const &);

    // -----------------------------------------------------------------
    
    // Returns the local kinetic energy matrix R ke = sum_k R e_k R^+ f(e_k), where f(e_k) is the Fermi function, needed to calculate the hybridization matrix D
    template <typename T>
    auto get_ke(array<T,3> const &disp_R, array<T,3> const &vec_dag, array<double,2> const &wks)->matrix<T> {
      
      bool const is_complex = triqs::is_complex<T>::value;
      using h_scalar_t = std::conditional_t<is_complex, std::complex<double>, double>;
      
      auto _ = range();
      
      auto num_bands = first_dim(vec_dag);
      auto dim = second_dim(vec_dag);
      
      // Calculate the local kinetic energy on each cluster
      matrix<h_scalar_t> ke_local(dim, dim);
      set_zero(ke_local);
      
      for (auto const alpha : range(dim)) {
        for (auto const a : range(dim)) {
          for (auto const j : range(num_bands)) { 
            array<h_scalar_t,1> slice = disp_R(alpha,j,_) * wks(j,_) * vec_dag(j,a,_);
            ke_local(alpha,a) += fsum<h_scalar_t>(slice.begin(), slice.end());
          }
        }
      }

      return ke_local; // this is ordered alph,a but not that D is ordered a,alpha, so remember to transpose for D
    }
    template matrix<double> get_ke(array<double,3> const &, array<double,3> const &, array<double,2> const &);
    template matrix<dcomplex> get_ke(array<dcomplex,3> const &, array<dcomplex,3> const &, array<double,2> const &);
 
    // -----------------------------------------------------------------
  
    // NOTE: A general way to compute [\sum_k A_k f(e_k) B_k]_{ab} is
    //       \sum_n \sum_k [A_k P_k]_{an} [D_k]_n  [P_k^+ B_k]_{nb} where
    //       D_k is the diagonal matrix of eigenvalues of e_k with corresponding
    //       eigenvectors in P_k in the columns
    //       pdensity: A_k = B_k = I
    //       ke: A_k = e_k R^+, B_k = I
    // Returns the local density matrix of the f-fermions from the mean-field
    template <typename T>  
    auto get_pdensity(array<T,3> const &vec, array<T,3> const &vec_dag, array<double,2> const &wks)->matrix<double> {
      
      auto _ = range();

      auto num_bands = first_dim(vec_dag);
      auto dim = second_dim(vec_dag);
      
      matrix<double> qp_num(dim, dim);
      set_zero(qp_num);
      
      for (auto const a : range(dim)) {
        for (auto const b : range(dim)) {
          for (auto const j : range(num_bands)) {
            // this without being a view is probably not efficient? This probably allocates memory FIXME
            // and having to cast to array is just because my integrator does not take array_view (so this part probably copies?) 
            array<double,1> slice = real( vec(a,j,_) * wks(j,_) * vec_dag(j,b,_) );
            qp_num(a,b) += fsum<double>(slice.begin(), slice.end());
          }
        }
      }
      return qp_num.transpose();
    }
    template matrix<double> get_pdensity(array<double,3> const &, array<double,3> const &, array<double,2> const &);
    template matrix<double> get_pdensity(array<dcomplex,3> const &, array<dcomplex,3> const &, array<double,2> const &);

    // -----------------------------------------------------------------
    
    // Returns the hybridization matrix of the impurity
    template <typename H>  
    auto get_d(matrix_const_view<double> pdensity, matrix_const_view<H> ke)->matrix<H> {
      
      auto K = pdensity - pdensity * pdensity;
      auto K_sq = matrix_sq(K);
      //auto K_sq_inv = pseudo_inverse(K_sq);
      auto K_sq_inv = inverse(K_sq);
      return K_sq_inv * ke.transpose();
    }
    template matrix<double> get_d(matrix_const_view<double>, matrix_const_view<double>);
    template matrix<dcomplex> get_d(matrix_const_view<double>, matrix_const_view<dcomplex>);

    // -----------------------------------------------------------------
    
    // Returns the bath levels lambda_c
    template <typename H>  
    auto get_lambda_c(matrix_const_view<double> pdensity, matrix_const_view<H> R, matrix_const_view<double> lambda, matrix_const_view<H> D)->matrix<double> {
     
      auto dim = first_dim(pdensity);
      
      triqs::clef::placeholder<0> i_;
      triqs::clef::placeholder<1> j_;
      matrix<double> identity(dim,dim);
      identity(i_,j_) << (i_ == j_) * 1.0;
      
      auto P = identity - 2.0*pdensity;
      auto K = pdensity - pdensity * pdensity;
      auto K_sq = matrix_sq(K);
      //auto K_sq_inv = pseudo_inverse(K_sq);
      auto K_sq_inv = inverse(K_sq);
      // Because lhs + conj(lhs) I could just take the real part and multiply by 2
      //auto lhs = 0.5*( (R*D).transpose()*K_sq_inv*P ).transpose();
      //return -real(lhs + conj(lhs)) - lambda; // - _mu * identity;
      return -real(( (R*D).transpose()*K_sq_inv*P ).transpose()) - lambda;
    }
    template matrix<double> get_lambda_c(matrix_const_view<double>, matrix_const_view<double>, matrix_const_view<double>, matrix_const_view<double>);
    template matrix<double> get_lambda_c(matrix_const_view<double>, matrix_const_view<dcomplex>, matrix_const_view<double>, matrix_const_view<dcomplex>);

    // -----------------------------------------------------------------
    
    // Returns a new guess for Lambda
    template <typename H>  
    auto get_lambda(matrix_const_view<H> R, matrix_const_view<H> D, matrix_const_view<double> lambda_c, matrix_const_view<double> Nf)->matrix<double> {

      auto dim = first_dim(Nf);
      //matrix<double> identity(dim, dim);
      //set_zero(identity);
      //identity() = 1.0;
      
      triqs::clef::placeholder<0> i_;
      triqs::clef::placeholder<1> j_;
      matrix<double> identity(dim,dim);
      identity(i_,j_) << (i_ == j_) * 1.0;
      
      auto P = identity - 2.0 * Nf;
      auto K = Nf - Nf * Nf;
      auto K_sq = matrix_sq(K);
      //auto K_sq_inv = pseudo_inverse(K_sq);
      auto K_sq_inv = inverse(K_sq);
      // Because lhs + conj(lhs) I could just take the real part and multiply by 2
      //auto lhs = 0.5 * ((R * D).transpose() * Nf_sq_inv * P).transpose(); // FIXME auto causes a bug here?
      //return -real(lhs + conj(lhs)) - lambda_c;
      return -real(( (R*D).transpose()*K_sq_inv*P ).transpose()) - lambda_c;
    }
    template matrix<double> get_lambda(matrix_const_view<double>, matrix_const_view<double>, matrix_const_view<double>, matrix_const_view<double>);
    template matrix<double> get_lambda(matrix_const_view<dcomplex>, matrix_const_view<dcomplex>, matrix_const_view<double>, matrix_const_view<double>);
    
    // -----------------------------------------------------------------

    // Returns a new guess for R
    template <typename H>  
    auto get_r(matrix_const_view<H> Mcf, matrix_const_view<double> Nf)->matrix<H> {
     
      auto K = Nf - Nf * Nf;
      auto K_sq = matrix_sq(K);
      //auto K_sq_inv = pseudo_inverse(K_sq);
      auto K_sq_inv = inverse(K_sq);
      return (Mcf * K_sq_inv).transpose();
    }
    template matrix<double> get_r(matrix_const_view<double>, matrix_const_view<double>);
    template matrix<dcomplex> get_r(matrix_const_view<dcomplex>, matrix_const_view<double>); // Can this have a complex density?

    // -----------------------------------------------------------------
    
    template <typename H>
    auto get_h_emb(triqs::operators::many_body_operator_generic<H> const &h_loc, matrix<H> const &D, matrix<double> const &lambda_c, fundamental_operator_set const &fops_local, fundamental_operator_set const &fops_bath)->triqs::operators::many_body_operator_generic<H> {
      
      using many_body_op_t = triqs::operators::many_body_operator_generic<H>;

      auto h_emb = h_loc;
    
      for (auto const &o : fops_bath) {
        auto i = o.linear_index;
        for (auto const &oo : fops_bath) {
          auto j = oo.linear_index;
          auto fa  = many_body_op_t::make_canonical(true, o.index);
          auto fb = many_body_op_t::make_canonical(false, oo.index);
          h_emb += lambda_c(i,j) * fb * fa;
        }
      }
      
      for (auto const &o : fops_bath) {
        auto i = o.linear_index;
        for (auto const &oo : fops_local) {
          auto j = oo.linear_index;
          auto fa  = many_body_op_t::make_canonical(false, o.index);
          auto ca = many_body_op_t::make_canonical(true, oo.index);
          auto res = D(i,j) * ca * fa;
          h_emb += res + dagger(res);
        }
      }

      return h_emb;
    }
    template triqs::operators::many_body_operator_generic<double> get_h_emb(triqs::operators::many_body_operator_generic<double> const &, matrix<double> const &, matrix<double> const &, fundamental_operator_set const &, fundamental_operator_set const &);
    template triqs::operators::many_body_operator_generic<dcomplex> get_h_emb(triqs::operators::many_body_operator_generic<dcomplex> const &, matrix<dcomplex> const &, matrix<double> const &, fundamental_operator_set const &, fundamental_operator_set const &);
    
    // -----------------------------------------------------------------
    
    // Given a basis of matrices (hs), and coefficients (coef), returns
    //   - the dense matrix mat = sum_s coeff_s * hs_s
    template <typename H> matrix<H> matrix_construct(std::vector<H> const &coeff, std::vector<matrix<dcomplex>> const &hs) {
      
      bool const is_complex = triqs::is_complex<H>::value;
      //using coeff_scalar_t = std::conditional_t<is_complex, std::complex<double>, double>;
     
      auto dim = first_dim(hs[0]);
      matrix<dcomplex> A(dim, dim);
      set_zero(A);
      for(auto s : range(hs.size())) A += dcomplex(coeff[s]) * hs[s];
      if (is_complex) return A;
      else return real(A);
    } 
    //template matrix<double> matrix_construct(std::vector<double> const &, std::vector<matrix<dcomplex>> const &);
    template matrix<dcomplex> matrix_construct(std::vector<dcomplex> const &, std::vector<matrix<dcomplex>> const &);

    // -----------------------------------------------------------------
    
    // Given a dense matrix (A), and basis matrices (hs), returns
    //   - the coefficients as a vector
    // 
    // Orthonormal matrices inner product is 
    // (A,B) = Tr(A^+ * B), and Tr(hs_s^+ * h_s') = delta_s,s'
    // Tr(hs_s^+ * A) = sum_s' Tr(coeff_s' * * hs_s^+ * hs_s')
    //                = sum_s' coeff_s' Tr(hs_s^+ * hs_s')
    //                = coeff_s
    // -> coeff_s = Tr(hs_s^+ * A)
    template <typename H> std::vector<H> coeff_construct(matrix<H> const &A, std::vector<matrix<dcomplex>> const &hs) {
      
      bool const is_complex = triqs::is_complex<H>::value;
      
      auto dim = hs.size();
      std::vector<H> coeff(dim);
      if (is_complex) {
        for (auto s : range(dim)) {
          coeff[s] = trace(dagger(hs[s]) * A) / trace(dagger(hs[s]) * hs[s]);
        }
      }
      else {
        for (auto s : range(dim)) {
          //coeff(s) = trace(hs(s).transpose() * A); // assumes hs is an orthonormal basis
          coeff[s] = real( trace(hs[s].transpose() * A) / trace(hs[s].transpose() * hs[s]) );
        }
      }
      return coeff;
    } 
    //template std::vector<double> coeff_construct(matrix<double> const &, std::vector<matrix<dcomplex>> const &);
    template std::vector<dcomplex> coeff_construct(matrix<dcomplex> const &, std::vector<matrix<dcomplex>> const &);
    
    // -----------------------------------------------------------------

    template <typename H>
    auto get_pdensity_gf(gf_const_view<imfreq> g_z, matrix_const_view<H> R)->matrix<double> {

      auto const &z_mesh = g_z.mesh();

      /*
      int dim = first_dim(R);
      auto beta = g_z.mesh()[0].beta;
      matrix<double> pdensity(dim,dim);
      pdensity(i_,j_) << 0.5 * (i_ == j_);
      for (auto const &z : z_mesh) {
        pdensity += real( inverse(dagger(R)) * g_z[z] * inverse(R) ) / beta;
      }
      return pdensity;
      */

      auto gqp_z = make_gf(g_z);
      for (auto const &z : z_mesh) {
        gqp_z[z] = inverse(dagger(R)) *  gqp_z[z] * inverse(R);
      }
      return real(density(gqp_z));
    }
    template matrix<double> get_pdensity_gf(gf_const_view<imfreq>, matrix_const_view<double>);
    template matrix<double> get_pdensity_gf(gf_const_view<imfreq>, matrix_const_view<dcomplex>);

    // -----------------------------------------------------------------
    
    template <typename H>
    auto get_ke_gf(gf_const_view<imfreq> g_z, gf_const_view<imfreq> delta_z, matrix_const_view<H> R)->matrix<H> {

      triqs::clef::placeholder<0> i_;
      triqs::clef::placeholder<1> j_;
      
      auto g_ke = make_gf(g_z);
      auto const &z_mesh = g_ke.mesh();

      for (auto const &z : z_mesh) {
        g_ke[z] = delta_z[z] * g_z[z];
      }

      int dim = first_dim(R);
      auto beta = g_z.mesh()[0].beta;
      matrix<H> ke(dim,dim);
      ke(i_,j_) << 0.0 * (i_ == j_);
      for (auto const &z : z_mesh) {
        ke += real(g_ke[z]) / beta;
      }
      return real( ke * inverse(R) ); // FIXME for non-real?

      /*
      int dim = first_dim(R);
      auto ke = (density(g_ke) - 0.5*identity) * inverse(R);
      return real(ke);
      */

      //return real( density(g_ke) * inverse(R) );
    }
    template matrix<double> get_ke_gf(gf_const_view<imfreq>, gf_const_view<imfreq>, matrix_const_view<double>);
    template matrix<dcomplex> get_ke_gf(gf_const_view<imfreq>, gf_const_view<imfreq>, matrix_const_view<dcomplex>);

    // -----------------------------------------------------------------

    auto get_delta_z(gf_const_view<imfreq> g0_z)->gf<imfreq> {
      auto delta_z = inverse(g0_z);
      auto const &z_mesh = delta_z.mesh();
      for (auto const &z : z_mesh) {
        delta_z[z] = z - delta_z[z];
      }
      return delta_z;
    }

    // -----------------------------------------------------------------
    template <typename H>
    auto get_sigma_z(gf_const_view<imfreq> g_z, matrix_const_view<H> R, matrix_const_view<double> lambda, double const mu)->gf<imfreq> {
      
      auto dim = first_dim(R);
      triqs::clef::placeholder<0> i_;
      triqs::clef::placeholder<1> j_;
      matrix<double> identity(dim,dim);
      identity(i_,j_) << 1.0 * (i_ == j_);
      
      auto Z_inv = inverse(dagger(R)*R);
      //auto sigma_z = gf<imfreq>(g_z);
      auto sigma_z = make_gf(g_z);

      auto const &z_mesh = sigma_z.mesh();
      for (auto const &z : z_mesh) {
        sigma_z[z] = z*(identity-Z_inv) + inverse(R)*lambda*inverse(dagger(R)); // - one_body_terms e0?
        sigma_z[z] += mu*(identity-Z_inv);
      }

      return sigma_z;
    }
    template gf<imfreq> get_sigma_z(gf_const_view<imfreq>, matrix_const_view<double>, matrix_const_view<double>, double const);
    template gf<imfreq> get_sigma_z(gf_const_view<imfreq>, matrix_const_view<dcomplex>, matrix_const_view<double>, double const);

    

  } // namespace solver
} // namespace risb
