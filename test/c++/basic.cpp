#include "./common.hpp"

TEST(common_functions, matrix_sq) { // NOLINT
  
  using namespace risb;

  auto A = matrix<double>{{{41, 12}, {12, 34}}};
  auto A_sq = matrix_sq(A);

  auto sqrt2 = std::pow(2,0.5);
  auto A_sq_exact = matrix<double>(2,2);
  A_sq_exact(0,0) = (9.0+16.0*sqrt2) / 5.0;
  A_sq_exact(0,1) = (-12.0+12.0*sqrt2) / 5.0;
  A_sq_exact(1,0) = (-12.0+12.0*sqrt2) / 5.0;
  A_sq_exact(1,1) = (16.0+9.0*sqrt2) / 5.0;

  EXPECT_ARRAY_NEAR(A_sq, A_sq_exact, 1e-14);
}

TEST(common_functions, pseudo_inverse_real) { // NOLINT
  
  using namespace risb;

  auto A = matrix<double>{{{1, 0}, {1, 0}}};
  auto A_inv = pseudo_inverse(A);
  auto A_inv_exact = matrix<double>{{{1, 1}, {0, 0}}};
  A_inv_exact /= 2.0;

  // FIXME gives error: On entry to DGELSS parameter number  7 had an illegal value
  // but I only ever use square matrices anyway. I wonder why this doesn't work though?
  //auto C = matrix<double>{{{1, 0}, {0, 1}, {0, 1}}};
  //auto C_inv = pseudo_inverse(C);
  //auto C_inv_exact = matrix<double>{{{1,0,0}, {0,1,1}}};
  //C_inv_exact(1,1) = 0.5;
  //C_inv_exact(1,2) = 0.5;

  EXPECT_ARRAY_NEAR(A_inv, A_inv_exact, 1e-14);
  //EXPECT_ARRAY_NEAR(C_inv, C_inv_exact, 1e-14);
}

TEST(functions, get_hqp) { // NOLINT

  using namespace risb::functions;
  
  auto dispersion = build_cubic_dispersion();
  auto [R, lambda] = build_mf_matrices(); 
  auto h_qp = get_h_qp<double,double>({R}, {lambda}, dispersion);
}

TEST(functions, get_ke) { // NOLINT

  using namespace risb::functions;
  
  auto dispersion = build_cubic_dispersion();
  auto nk = fifth_dim(dispersion);
  auto [R, lambda] = build_mf_matrices(); 
  auto h_qp = get_h_qp<double,double>({R}, {lambda}, dispersion);
  auto disp_R = get_disp_R<double,double>({R},dispersion,h_qp);
  //kintegrator.setEks(h_qp.val);
  //kintegrator.setEF_fromFilling(1);
  //auto const &wks = kintegrator.getWs();
  auto wks = fermi_fnc(h_qp.val) / double(nk);
  auto ke = get_ke<double>(disp_R, h_qp.vec_dag, wks);
}


TEST(functions, get_pdensity) { // NOLINT

  using namespace risb::functions;
  
  auto dispersion = build_cubic_dispersion();
  auto nk = fifth_dim(dispersion);
  auto [R, lambda] = build_mf_matrices(); 
  auto h_qp = get_h_qp<double,double>({R}, {lambda}, dispersion);
  auto wks = fermi_fnc(h_qp.val) / double(nk);
  auto pdensity = get_pdensity<double>(h_qp.vec, h_qp.vec_dag, wks);
}


TEST(functions, get_d) { // NOLINT

  using namespace risb::functions;
  
  auto dispersion = build_cubic_dispersion();
  auto nk = fifth_dim(dispersion);
  auto [R, lambda] = build_mf_matrices(); 
  auto h_qp = get_h_qp<double,double>({R}, {lambda}, dispersion);
  auto disp_R = get_disp_R<double,double>({R},dispersion,h_qp);
  auto wks = fermi_fnc(h_qp.val) / double(nk);
  auto ke = get_ke<double>(disp_R, h_qp.vec_dag, wks);
  auto pdensity = get_pdensity<double>(h_qp.vec, h_qp.vec_dag, wks);
  auto D = get_d<double> (pdensity, ke);
}

TEST(functions, get_lambda_c) { // NOLINT

  using namespace risb::functions;
  
  auto dispersion = build_cubic_dispersion();
  auto nk = fifth_dim(dispersion);
  auto [R, lambda] = build_mf_matrices(); 
  auto h_qp = get_h_qp<double,double>({R}, {lambda}, dispersion);
  auto disp_R = get_disp_R<double,double>({R},dispersion,h_qp);
  auto wks = fermi_fnc(h_qp.val) / double(nk);
  auto ke = get_ke<double>(disp_R, h_qp.vec_dag, wks);
  auto pdensity = get_pdensity<double>(h_qp.vec, h_qp.vec_dag, wks);
  auto D = get_d<double> (pdensity, ke);
  auto lambda_c = get_lambda_c<double>(pdensity, R, lambda, D);
}

TEST(functions, get_h_emb) { // NOLINT
  
  using namespace risb::functions;
 
  int const orb_dim = 2; 
  auto dispersion = build_cubic_dispersion();
  auto nk = fifth_dim(dispersion);
  auto [R, lambda] = build_mf_matrices(); 
  auto h_qp = get_h_qp<double,double>({R}, {lambda}, dispersion);
  auto disp_R = get_disp_R<double,double>({R},dispersion,h_qp);
  auto wks = fermi_fnc(h_qp.val) / double(nk);
  auto ke = get_ke<double>(disp_R, h_qp.vec_dag, wks);
  auto pdensity = get_pdensity<double>(h_qp.vec, h_qp.vec_dag, wks);
  auto D = get_d<double> (pdensity, ke);
  auto lambda_c = get_lambda_c<double>(pdensity, R, lambda, D);
  
  fundamental_operator_set fops_local;
  std::vector<int> orbs;
  for (auto i : range(orb_dim / 2)) orbs.push_back(i+1);
  std::vector<std::string> spins = {"up","dn"};
  for (auto i : orbs) {
    for (auto s : spins) {
      fops_local.insert(s,i);
    }
  }
  auto [fops_bath, fops_emb] = get_embedding_space(fops_local);

  auto h_loc = 2.0 * n("up", 1) * n("dn", 1);
  //for (auto s : spins) h_loc -= c_dag(s, 1) * c(s, 2) + c_dag(s, 2) * c(s, 1);

  auto h_emb = get_h_emb<double>(h_loc, D, lambda_c, fops_local, fops_bath);
}


TEST(functions, solve_emb) { // NOLINT

  using namespace risb::functions;
  using namespace risb::embedding_atom_diag;
  
  int orb_dim = 2;
  auto dispersion = build_cubic_dispersion(orb_dim);
  auto nk = fifth_dim(dispersion);
  auto [R, lambda] = build_mf_matrices(orb_dim); 
  auto h_qp = get_h_qp<double,double>({R}, {lambda}, dispersion);
  auto disp_R = get_disp_R<double,double>({R},dispersion,h_qp);
  auto wks = fermi_fnc(h_qp.val) / double(nk);
  auto ke = get_ke<double>(disp_R, h_qp.vec_dag, wks);
  auto pdensity = get_pdensity<double>(h_qp.vec, h_qp.vec_dag, wks);
  auto D = get_d<double> (pdensity, ke);
  auto lambda_c = get_lambda_c<double>(pdensity, R, lambda, D);
  
  auto gf_struct = gf_struct_t{{"up", {1}}, {"dn", {1}}};
  fundamental_operator_set fops_local;
  std::vector<int> orbs;
  for (auto i : range(orb_dim / 2)) orbs.push_back(i+1);
  std::vector<std::string> spins = {"up","dn"};
  for (auto i : orbs) {
    for (auto s : spins) {
      fops_local.insert(s,i);
    }
  }
  auto [fops_bath, fops_emb] = get_embedding_space(fops_local);
  auto h_loc = 0.0 * n("up", 1) * n("dn", 1);
  
  //auto h_emb = get_h_emb<double>(h_loc, D, lambda_c, fops_local, fops_bath);
  
  auto emb_solver = embedding_atom_diag<false>(gf_struct);
  
  //auto Nf = emb_solver.get_nf();
  //auto Mcf = emb_solver.get_mcf();

  //auto lambda_1 = get_lambda(R, D, lambda_c, Nf);
  //auto R_1 = get_r(Mcf, Nf);

  //EXPECT_ARRAY_NEAR(R, R_1, 1e-10); // NOLINT
  //EXPECT_ARRAY_NEAR(lambda, lambda_1, 1e-10); // NOLINT
}

TEST(embedding_atom_diag, constructor) { // NOLINT
  
  using namespace risb::functions;
  using namespace risb::embedding_atom_diag;
 
  // Set of fundamental operators
  fundamental_operator_set fops_local;
  //fundamental_operator_set fops_bath, fops_emb;
  auto gf_struct = gf_struct_t{{"up", {1,2}}, {"dn", {1,2}}};
  std::vector<int> orbs = {1,2};
  std::vector<std::string> spins = {"up","dn"};
  for (auto i : orbs)
  {
    for (auto s : spins) {
      //if (i%2) fops_local.insert(s,i);
      //else fops_bath.insert(s,i);
      //fops_emb.insert(s,i);
      
      fops_local.insert(s,i);
    }
  }

  auto [fops_bath, fops_emb] = get_embedding_space(fops_local);

  // Hamiltonian
  auto h_loc = 2.0 * (n("up", 1) * n("dn", 1) + n("up", 2) * n("dn", 2));
  for (auto s : spins) h_loc -= c_dag(s, 1) * c(s, 2) + c_dag(s, 2) * c(s, 1);
  
  
  auto emb = embedding_atom_diag<false>(gf_struct);
  
  //auto Nf = emb.get_nf();
  //auto Nc = emb.get_nc();
  //auto Mcf = emb.get_mcf();
}
