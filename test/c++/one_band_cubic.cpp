#include "./common.hpp"

TEST(one_band_cubic, half_filling) { // NOLINT

  using namespace risb::functions;
  using namespace risb::embedding_atom_diag;
  
  double const beta = 10;
  int const num_cycles = 25;

  std::vector<std::string> const spin_names = {"up","dn"};
  auto const gf_struct = gf_struct_t{{"up", {1}}, {"dn", {1}}};
  
  double const U = 1.5;
  auto const h_loc = U * n("up", 1) * n("dn", 1);
  double const mu = U / 2.0; // half-filling
  
  auto const dispersion = build_cubic_dispersion(); // same dispersion for each block
  auto const nk = fifth_dim(dispersion);

  auto [R, lambda] = build_mf_matrices(gf_struct);
  auto [D, lambda_c] = build_mf_matrices(gf_struct);
    
  auto emb_solver = embedding_atom_diag<false>(gf_struct);
  
  double error;
  int total_cycles = 0;

#ifdef __GNUC__
  for (__attribute__ ((unused)) auto const cycle : range(num_cycles)) { 
#else
  for (auto const cycle : range(num_cycles)) { 
#endif

    error = 0;
    auto R_old = R;
    auto lambda_old = lambda;

    for (auto i : range(spin_names.size())) {
      auto block = spin_names[i];
      //matrix<double> const &R_b = R[i];
      //matrix<double> const &lambda_b = lambda[i];
      //matrix<double> &D_b = D[i];
      //matrix<double> &lambda_c_b = lambda_c[i];

      auto const &R_b = R.at(block);
      auto const &lambda_b = lambda.at(block);
      auto &D_b = D.at(block);
      auto &lambda_c_b = lambda_c.at(block);

      auto h_qp = get_h_qp<double,double>({R_b}, {lambda_b}, dispersion, mu);
      auto disp_R = get_disp_R<double,double>({R_b}, dispersion, h_qp);
    
      // Simple k-sums
      auto wks = fermi_fnc(h_qp.val, beta) / double(nk);
    
      auto ke = get_ke<double>(disp_R, h_qp.vec_dag, wks);
      auto pdensity = get_pdensity(h_qp.vec, h_qp.vec_dag, wks);
      
      D_b = get_d(pdensity, ke);
      lambda_c_b = get_lambda_c(pdensity, R_b, lambda_b, D_b);
    }

    emb_solver.set_h_emb(h_loc,lambda_c,D);
    emb_solver.solve();

    for (auto i : range(spin_names.size())) {
      auto block = spin_names[i];
      //matrix<double> &R_b = R[i];
      //matrix<double> &lambda_b = lambda[i];
      //matrix<double> const &D_b = D[i];
      //matrix<double> const &lambda_c_b = lambda_c[i];
      
      auto &R_b = R.at(block);
      auto &lambda_b = lambda.at(block);
      auto const &D_b = D.at(block);
      auto const &lambda_c_b = lambda_c.at(block);

      auto Nf = emb_solver.get_nf(block);
      auto Mcf = emb_solver.get_mcf(block);

      lambda_b =  get_lambda(R_b, D_b, lambda_c_b, Nf);
      R_b = get_r(Mcf, Nf);

      error = frobenius_norm(matrix<double>(R_b - R_old.at(block)));
      error += frobenius_norm(matrix<double>(lambda_b - lambda_old.at(block)));

    }

    if (error < 1e-6) {
      total_cycles = cycle;
      break;
    }
  }

  std::cout << "cycles = " << total_cycles << "  error = " << error << std::endl;
  std::cout << "R = " << R.at("up") << std::endl;
  std::cout << "lambda = " << lambda.at("up") << std::endl;

  double mu_calculated = 0;
  for (auto block : spin_names) mu_calculated += trace(lambda.at(block)) / 2;
  double mu_expected = U / 2.0;
  matrix<double> R_expected = {{{0.861617}}};
  matrix<double> lambda_expected = {{{mu_expected}}};

  EXPECT_NEAR(mu_expected, mu_calculated, 1e-6); // NOLINT
  for (auto block : spin_names) {
    EXPECT_ARRAY_NEAR(R_expected, R.at(block), 1e-6); // NOLINT
    EXPECT_ARRAY_NEAR(lambda_expected, lambda.at(block), 1e-6); // NOLINE
  }
}

