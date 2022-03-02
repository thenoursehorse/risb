#pragma once
#include <vector>
#include <triqs/arrays/vector.hpp>
#include <triqs/arrays/matrix.hpp>
#include <triqs/arrays/block_matrix.hpp>
#include <triqs/operators/many_body_operator.hpp>
#include <triqs/hilbert_space/fundamental_operator_set.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/atom_diag/functions.hpp>
#include <triqs/utility/first_include.hpp>


namespace risb {
  namespace embedding_atom_diag {

    using namespace triqs::hilbert_space;
    using namespace triqs::arrays;
    using namespace triqs::atom_diag;

    template <bool Complex> class embedding_atom_diag {

      static constexpr bool is_complex = Complex;

      public:
      
      /// Type of matrix elements
      using scalar_t = std::conditional_t<Complex, std::complex<double>, double>;
      /// Many-body operator type
      using many_body_op_t = triqs::operators::many_body_operator_generic<scalar_t>;
      ///
      using gf_struct_t  = triqs::hilbert_space::gf_struct_t;
      ///
      using atom_diag_t = triqs::atom_diag::atom_diag<Complex>;
      ///
      using block_matrix_t = typename triqs::atom_diag::atom_diag<Complex>::block_matrix_t;
      ///
      using full_hilbert_space_state_t = typename triqs::atom_diag::atom_diag<Complex>::full_hilbert_space_state_t;
      
      
      /// Construct in an uninitialized state.
      TRIQS_CPP2PY_IGNORE embedding_atom_diag() = default;


      /// A wrapper for the RISB impurity model using atom_diag from TRIQS
      /**
       * This constructor uses atom_diag in TRIQS with the auto-partition procedure.
       *
       * @param h_loc The local Hamiltonian describing the impurity.
       * @param gf_struct The structure of the matrices/Green's functions in the local space
       * @note See atom_diag in TRIQS for more details on how the diagonalization is performed.
       */
      embedding_atom_diag(gf_struct_t const &gf_struct, double const beta = 1e6);

      
      /// 
      atom_diag_t const &get_ad() const {
        return _ad;
      }


      ///
      block_matrix_t const &get_dm() const {
        return _dm;
      }


      ///
      void solve();
      

      /// 
      template <class M, class N> void set_h_emb(many_body_op_t const &h_loc, M const &lambda_c, N const &D, double const mu = 0);


      /// Density matrix of the f-fermions
      /**
       * @return The matrix Nf with dimension of the local/bath hilbert space
       */
      matrix<double> get_nf(std::string const& block) const;
      
      
      /// Density matrix of the c-fermions
      /**
       * @return The matrix Nc with dimension of the local hilbert space
       */
      matrix<double> get_nc(std::string const& block) const;
      
      
      /// Density matrix of the hybridization 
      /**
       * @return The matrix Mcf with dimension of the local hilbert space
       */
      matrix<scalar_t> get_mcf(std::string const& block) const;

      
      /// Set a new density matrix for a given temperature beta
      void set_density_matrix(double const beta) {
        _dm = atomic_density_matrix(_ad, beta);
      }

      private:
     
      gf_struct_t const _gf_struct;
      double const _beta;
      
      // Group fops by the structure in gf_struct
      void _set_structure();
     
      fundamental_operator_set _fops_emb;
      std::vector<fundamental_operator_set> _fops_loc;
      std::vector<fundamental_operator_set> _fops_bath;
      many_body_op_t _h_emb;
      
      atom_diag_t _ad;
      block_matrix_t _dm;
      full_hilbert_space_state_t _gs_vec;

      // For mapping gf_struct to integers
      std::map<std::string,int> _str_map;
      std::vector<std::string> _block_names;
    };

    // To expose to cpp2py // FIXME how to expose the methods?
    struct embedding_atom_diag_real {
      typedef embedding_atom_diag<false> type;
    };
    
    struct embedding_atom_diag_complex {
      typedef embedding_atom_diag<true> type;
    };


    // For some of the possible different ways to do block matrices
    template <typename T>
    matrix<T> block_pick(std::map<std::string,matrix<T>> const &mat, std::string const &b) {
      return mat.at(b);
    }

    //template <typename T>
    //matrix<T> block_pick(block_matrix<T> const &mat, std::string const &b) {
    //  return mat(b); // complains about a const issue with using ()
    //}


  } // namespace embedding_atom_diag
} // namespace risb
