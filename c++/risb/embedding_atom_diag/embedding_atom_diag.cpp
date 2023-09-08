#include "./embedding_atom_diag.hpp"
#include <nda/blas/dot.hpp>

namespace risb {
  namespace embedding_atom_diag {

#define EMBEDDING_ATOM_DIAG_CONSTRUCTOR(ARGS) template <bool Complex> embedding_atom_diag<Complex>::embedding_atom_diag ARGS
#define EMBEDDING_ATOM_DIAG_METHOD(RET, F) template <bool Complex> auto embedding_atom_diag<Complex>::F->RET

    EMBEDDING_ATOM_DIAG_CONSTRUCTOR((gf_struct_t const &gf_struct, double const beta))
      : _gf_struct(gf_struct), _beta(beta) {
      _set_structure();
    }

    // -----------------------------------------------------------------
    
    EMBEDDING_ATOM_DIAG_METHOD(void, _set_structure()) {
      
      // Setup map from blocks to integers
      //for (auto it = _gf_struct.begin(); it != _gf_struct.end(); ++it) {
      //  auto const &block = *it;
      //  _str_map.insert(std::make_pair(block.first, it));
      //}
      int i = 0;
      for (auto const &block : _gf_struct) {
        _str_map.insert(std::make_pair(block.first,i));
        i++;

        _block_names.push_back(block.first);
      }

      // Construct local and bath hilbert space in terms of block structure
      //for (auto const &block : _gf_struct) {
      for (auto const &[block, bl_size] : _gf_struct) {
        auto fops1 = fundamental_operator_set();
        auto fops2 = fundamental_operator_set();
        //for (auto const &inner : block.second) {
        for (int inner : range(bl_size)) {
          fops1.insert(block, inner);
          fops2.insert("bath_" + block, inner);
        }
        _fops_loc.push_back(fops1);
        _fops_bath.push_back(fops2);
      }
      
      // Construct embedding hilbert space
      for (auto const &fops : _fops_loc)
        for (auto const &o : fops) _fops_emb.insert(o.index);
      
      for (auto const &fops : _fops_bath)
        for (auto const &o : fops) _fops_emb.insert(o.index);
    }
    
    // -----------------------------------------------------------------
    
    
    EMBEDDING_ATOM_DIAG_METHOD(void, solve()) {
      //_ad = atom_diag_t(_h_emb, _fops_emb);
      
      // Test to reduce how expensive it is to solve by restricting particle sectors
      // NOTE: atom_diag only needs particle sectors around half-filling for the interaction term
      //       and for calculating the densities
      int M = _fops_emb.size() / 2;
      //_ad = atom_diag_t(_h_emb, _fops_emb);
      //_ad = atom_diag_t(_h_emb, _fops_emb, M-2, M+2);
      _ad = atom_diag_t(_h_emb, _fops_emb, M, M);

      std::cout << "Found " << _ad.n_subspaces() << " subspaces." << std::endl;

      // Test for some observables
      //std::cerr << "n_subspaces = " << _ad.n_subspaces() << " ";
      //std::cerr << "full_hilbert_space_dim = " << _ad.get_full_hilbert_space_dim() << " ";
      //std::cerr << "gs_energy = " << _ad.get_gs_energy() << std::endl;
      //auto energies = _ad.get_energies();
      //for (auto en_b : energies)
      //  for (auto en : en_b)
      //    std::cerr << en + _ad.get_gs_energy() << " ";
      //std::cerr << std::endl;

      //_dm = atomic_density_matrix(_ad,_beta);

      // Below is very inefficient because it has to create the full vector and take the overlap every time 
      // Find lowest energy vector in in subspace of M particles
      // We require \hat{N} |Phi> = M |Phi> for risb
      many_body_op_t N;
      for (auto const block : range(_fops_loc.size())) {
        auto const& fops_l = _fops_loc[block];
        auto const& fops_b = _fops_bath[block];
        for (auto const &o : fops_l) {
          auto nalpha  = many_body_op_t::make_canonical(true, o.index)
                        * many_body_op_t::make_canonical(false, o.index);
          N += nalpha;
        }
        for (auto const &o : fops_b) {
          auto na  = many_body_op_t::make_canonical(true, o.index)
                        * many_body_op_t::make_canonical(false, o.index);
          N += na;
        }
      }

      // Basis of vectors are in the eigenbasis,
      // organised from lowest energy to highest energy (in each block)
      // FIXME if using this it is not safe because the block ordering can mess it up
      // FIXME Should get all states with M particles, and then sort their energies and pick the lowest one
      //_gs_vec.resize(_ad.get_full_hilbert_space_dim());
      //_gs_vec() = 0;
      //for (auto i : range(_ad.get_full_hilbert_space_dim())) {
      //  _gs_vec(i) = 1;
      //  auto N_part = real( dot( _gs_vec, act(N, _gs_vec, _ad) ) );
      //  if (std::abs(N_part - (double)M) < 1e-12) break;
      //  _gs_vec(i) = 0;
      //}

      // If constraining to only M subsector then the groundstate will always be this instead
      _gs_vec.resize(_ad.get_full_hilbert_space_dim());
      _gs_vec() = 0;
      _gs_vec(0) = 1; // Ground state vector in the eigenbasis of the full hilbert space
    }
    
    
    
    // -----------------------------------------------------------------
    EMBEDDING_ATOM_DIAG_METHOD(double, overlap(many_body_op_t const& Op) const) {
      return real( dot( _gs_vec, act(Op, _gs_vec, _ad)) );
    }


    // -----------------------------------------------------------------
    
    //EMBEDDING_ATOM_DIAG_METHOD(template <class M, class N> void, set_h_emb(many_body_op_t const &h_loc, M const &lambda_c, N const &D, double const mu)) {
    template <bool Complex> 
    template<class M, class N> auto embedding_atom_diag<Complex>::set_h_emb(many_body_op_t const &h_loc, M const &lambda_c, N const &D, double const mu)->void {

      _h_emb = h_loc;

      for (auto const block : range(_fops_loc.size())) {
        auto const& fops_b = _fops_bath[block];
        auto const& fops_l = _fops_loc[block];
        std::string bn = _block_names[block];
        auto const& D_b = block_pick(D,bn);
        auto const& lambda_c_b = block_pick(lambda_c,bn);

        for (auto const &o : fops_l) {
          auto c_dag  = many_body_op_t::make_canonical(true, o.index);
          auto c  = many_body_op_t::make_canonical(false, o.index);
          _h_emb -= mu * c_dag * c;
        }
        
        for (auto const &o : fops_b) {
          auto i = o.linear_index;
          auto fa  = many_body_op_t::make_canonical(true, o.index);
          for (auto const &oo : fops_b) {
            auto j = oo.linear_index;
            auto fb = many_body_op_t::make_canonical(false, oo.index);
            if (i == j) _h_emb += (lambda_c_b(i,j) + mu) * fb * fa; // if adding mu to impurity, subtract off lambda_c contribution
            //if (i == j) _h_emb += lambda_c_b(i,j) * fb * fa - mu * fa * fb;
            else _h_emb += lambda_c_b(i,j) * fb * fa;
          }
        }

        for (auto const &o : fops_b) {
          auto i = o.linear_index;
          auto fa  = many_body_op_t::make_canonical(false, o.index);
          for (auto const &oo : fops_l) {
            auto j = oo.linear_index;
            auto ca = many_body_op_t::make_canonical(true, oo.index);
            auto res = D_b(i,j) * ca * fa;
            _h_emb += res + dagger(res);
          }
        }
      }
    }
    //template void embedding_atom_diag<false>::set_h_emb(many_body_op_t const &, block_matrix<double> const &, block_matrix<embedding_atom_diag::scalar_t> const &, double const);
    //template void embedding_atom_diag<true>::set_h_emb(many_body_op_t const &, block_matrix<double> const &, block_matrix<embedding_atom_diag::scalar_t> const &, double const);

    template void embedding_atom_diag<false>::set_h_emb(many_body_op_t const &, std::map<std::string,matrix<double>> const &, std::map<std::string,matrix<scalar_t>> const &, double const);
    template void embedding_atom_diag<true>::set_h_emb(many_body_op_t const &, std::map<std::string,matrix<double>> const &, std::map<std::string,matrix<scalar_t>> const &, double const);
    
    // -----------------------------------------------------------------
    
    // Density matrix of the f-electrons
    EMBEDDING_ATOM_DIAG_METHOD(matrix<double>, get_nf(std::string const& block) const) {

      auto const& fops = _fops_bath[_str_map.at(block)];
      auto dim = fops.size();
      matrix<double> Nf(dim,dim);

      for (auto const &o : fops) {
        auto const i = o.linear_index;
        for (auto const &oo : fops) {
          auto const j = oo.linear_index;
          auto fa  = many_body_op_t::make_canonical(true, o.index);
          auto fb = many_body_op_t::make_canonical(false, oo.index);
          //Nf(i,j) = real( overlap(fb * fa) );
          Nf(i,j) = real( dot( _gs_vec, act(fb * fa, _gs_vec, _ad) ) );
          //Nf(i,j) = real(trace_rho_op(_dm, fb * fa, _ad));
        }
      }
      return Nf;
    }
    
    // -----------------------------------------------------------------
    
    // Density matrix of the c-electrons
    EMBEDDING_ATOM_DIAG_METHOD(matrix<double>, get_nc(std::string const& block) const) {
      
      auto const& fops = _fops_loc[_str_map.at(block)];
      auto dim = fops.size();
      matrix<double> Nc(dim,dim);
      
      for (auto const &o : fops) {
        auto const i = o.linear_index;
        for (auto const &oo : fops) {
          auto const j = oo.linear_index;
          auto ca  = many_body_op_t::make_canonical(true, o.index);
          auto cb = many_body_op_t::make_canonical(false, oo.index);
          //Nc(i,j) = real( overlap(ca * cb) );
          Nc(i,j) = real( dot( _gs_vec, act(ca * cb, _gs_vec, _ad) ) );
          //Nc(i,j) = real(trace_rho_op(_dm, ca * cb, _ad));
        }
      }
      return Nc;
    }
    
    // -----------------------------------------------------------------
    
    // Density matrix of the hybrdization
    EMBEDDING_ATOM_DIAG_METHOD(matrix<scalar_t>, get_mcf(std::string const& block) const) {
      
      auto const& fops_b = _fops_bath[_str_map.at(block)];
      auto const& fops_l = _fops_loc[_str_map.at(block)];
      auto dim = fops_l.size();
      matrix<scalar_t> Mcf(dim,dim);
      
      for (auto const &o : fops_l) {
        auto const i = o.linear_index;
        for (auto const &oo : fops_b) {
          auto const j = oo.linear_index;
          auto ca  = many_body_op_t::make_canonical(true, o.index);
          auto fa = many_body_op_t::make_canonical(false, oo.index);
          //auto res = overlap(ca * fa);
          auto res = dot( _gs_vec, act(ca * fa, _gs_vec, _ad) );
          //auto res = trace_rho_op(_dm, ca * fa, _ad);
          if (is_complex) Mcf(i,j) = res;
          else Mcf(i,j) = real(res);
        }
      }
      return Mcf;
    }


    // -----------------------------------------------------------------
    
    // Explicit instantiations
    template class embedding_atom_diag<false>;
    template class embedding_atom_diag<true>;

  } // namespace embedding_atom_diag
} // namespace risb

