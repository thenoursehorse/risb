#include <triqs/test_tools/gfs.hpp>
#include <triqs/test_tools/arrays.hpp>
#include <nda/nda.hpp>
//#include <nda/block_matrix.hpp>
#include <triqs/lattice/tight_binding.hpp>
#include <triqs/gfs.hpp>
#include <triqs/mesh.hpp>
#include <risb/common_functions.hpp>
#include <risb/functions/functions.hpp>
#include <risb/embedding_atom_diag/embedding_atom_diag.hpp>

using namespace nda;
using namespace triqs::hilbert_space;
using namespace triqs::operators;
using namespace triqs::gfs;
using triqs::hilbert_space::gf_struct_t;
using namespace triqs::lattice;
using dcomplex = std::complex<double>;

auto fermi_fnc(array<double,2> const &eks, double const beta = 10) {
  return inverse(exp(beta * eks) + 1);
}

auto build_mf_matrices(int dim = 1) {
  placeholder<0> i_;
  placeholder<1> j_;
  matrix<double> R(dim,dim);
  matrix<double> lambda(dim,dim);
  R(i_,j_) << 1.0 * (i_ == j_);
  lambda(i_,j_) << 0.0 * (i_ == j_);
  return std::pair<matrix<double>,matrix<double>>(R,lambda);
}

auto build_mf_matrices(gf_struct_t const &gf_struct) {
  using block_matrix_t = std::map<std::string,matrix<double>>;
  
  placeholder<0> i_;
  placeholder<1> j_;

  std::vector<std::string> names;
  std::vector<matrix<double>> R1;
  std::vector<matrix<double>> lambda1;
  for (auto const &block : gf_struct) {  
    names.push_back(block.first);
    R1.emplace_back(block.second.size(), block.second.size());
    lambda1.emplace_back(block.second.size(), block.second.size());
  }
  for (auto const i : range(R1.size())) {
    R1[i](i_,j_) << 1.0 * (i_ == j_);
    lambda1[i](i_,j_) << 0.0 * (i_ == j_);
  }
  //return std::pair<block_matrix<double>,block_matrix<double>> ( block_matrix<double>(names,R1), block_matrix<double>(names,lambda1) );
   
  block_matrix_t R;
  block_matrix_t lambda;

  for (auto i : range(names.size())) {
    R.insert({names[i],R1[i]});
    lambda.insert({names[i],lambda1[i]});
  }
  
  return std::pair<block_matrix_t,block_matrix_t> ( R, lambda );
}

auto build_cubic_dispersion(int const orb_dim = 1, int const nkx = 6, int const spatial_dim = 2) {
  using namespace risb;
  placeholder<0> i_;
  placeholder<1> j_;

  // Setup dispersion
  int na = 1;
  double t = - 0.5 / ((double) (spatial_dim));

  // Cubic lattice
  matrix<double> units(spatial_dim,spatial_dim);
  units(i_,j_) <<  1.0 * (i_ == j_);
  auto bl = bravais_lattice(units);

  auto displ_vec       = std::vector<std::vector<long>>{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  auto overlap_mat_vec = std::vector<matrix<dcomplex>>{{{t}}, {{t}}, {{t}}, {{t}}};
  auto tb = tight_binding{bl, displ_vec, overlap_mat_vec};
  auto energies = energies_on_bz_grid(tb, nkx);
  int nk = energies.shape()[1];

  array<double, 5> dispersion(na, na, orb_dim, orb_dim, nk);
  assign_foreach (dispersion, [&energies] (auto i, auto j, auto a, auto b, auto k) {
    return energies(0,k) * double( (i == j) * (a == b) );
  });

  /*
  matrix<int> real_space_unit_vectors(spatial_dim,spatial_dim);
  real_space_unit_vectors(i_,j_) << (i_ == j_) * 1;
  auto bz = brillouin_zone{bravais_lattice(real_space_unit_vectors)};

  // Construct the k-mesh
  matrix<double> bz_units_matrix(3,3);
  assign_foreach (bz_units_matrix, [spatial_dim,nkx](auto i, auto j) {
      return double(i == j) * ( (i < spatial_dim) ? 2.0 * M_PI / double(nkx) : 2.0 * M_PI);
  });

  matrix<int> bz_periodization_matrix(3,3);
  assign_foreach (bz_periodization_matrix, [spatial_dim,nkx](auto i, auto j) {
      return int(i == j) * ( (i < spatial_dim) ? nkx : 1);
  });

  auto k_mesh = cluster_mesh(bz_units_matrix,
                             bz_periodization_matrix);
  auto k_mesh_pt = [&k_mesh](auto k) { return *(k_mesh.begin() + k); };

  // Cubic dispersion
  array<double, 5> dispersion(na, na, orb_dim, orb_dim, k_mesh.size()); // eigenenergies for each band at each k
  assign_foreach (dispersion, [t,spatial_dim,&k_mesh_pt] (auto i, auto j, auto a, auto b, auto k) {
    double cos_k = 0;
    for (auto const p : range(spatial_dim)) cos_k += cos(k_mesh_pt(k)[p]);
    return -2.0 * t  * cos_k * double( (i == j) * (a == b) );
  });
  */

  return dispersion;
}

auto build_semicircular_dos(double const half_bandwidth = 1, int const epts = 1024, int const orb_dim = 1) {

  placeholder<0> i_;
  placeholder<1> j_;
  placeholder<2> a_;
  placeholder<3> b_;
  placeholder<4> e_;
  auto _ = range::all;

  int na = 1;

  double const de = 2.0 * half_bandwidth / (double(epts-1));
  array<double,1> energies(epts);
  energies(0) = -half_bandwidth;
  for (auto const i : range(1,epts)) energies(i) = energies(i-1) + de;

  array<double,5> dispersion(na, na, orb_dim, orb_dim, epts); // i mean, kind of
  dispersion(i_,j_,a_,b_,e_) << (i_ == j_) * (a_ == b_) * energies(e_);

  array<double,2> dos(orb_dim, epts);
  for (auto const i : range(epts)) {
    auto e = energies(i);
    if (abs(e) < half_bandwidth) dos(_,i) = sqrt( 1.0 - std::pow(e/half_bandwidth,2) )* 2 / M_PI / half_bandwidth;
    else dos(_,i) = 0.0;
  }

  return std::tuple<array<double,5>,array<double,2>,double> (dispersion, dos, de);
}
 
auto build_g_semicircular(double const beta = 200, long const nw = 30, double const half_bandwidth = 1.0, double const mu = 0.0, int const dim = 1)->gf<imfreq> {

  std::complex<double> I(0.0,1.0);
  
  auto g_iw = gf<imfreq> {{beta, Fermion, nw}, {dim, dim}};
  auto const &iw_mesh = g_iw.mesh();

  for (auto iw : iw_mesh) {
    auto om = iw + mu;
    g_iw[iw] = (om - I*std::copysign(1.0,imag(om))*sqrt(half_bandwidth*half_bandwidth - om*om))/half_bandwidth/half_bandwidth*2.0;
  }

  return g_iw;
}

auto build_g0_k_z_cubic(long const nw = 30, int const nkx = 6, double const beta = 10, int const dim = 1) {

  auto dispersion = build_cubic_dispersion(dim, nkx);
  auto bz     = brillouin_zone{bravais_lattice{{{1, 0}, {0, 1}}}};
  auto g_k_iw = gf<prod<brzone, imfreq>> {{{bz, nkx}, {beta, Fermion, nw}}, {dim, dim}};

  auto const &k_mesh = std::get<0>(g_k_iw.mesh());
  auto const &iw_mesh = std::get<1>(g_k_iw.mesh());

  auto _ = range::all;

  for (auto k : k_mesh) {
    matrix_const_view<double> disp_slice = dispersion(0,0,_,_,k.data_index());
    for (auto iw : iw_mesh) {
      g_k_iw[k,iw] = iw - disp_slice;
    }
  }
  g_k_iw = inverse(g_k_iw);

  return g_k_iw;

  //placeholder<0> k_;
  //placeholder<1> iw_;
  //double t = 0.5;
  //int spatial_dim = 2;
  //auto eps_k_ = -(2.0 * t / ((double) (spatial_dim))) * (cos(k_(0)) + cos(k_(1)));
  //g_k_iw(k_,iw_) << 1 / (iw_ - R * eps_k_ * dagger(R) - lambda + mu); // these are slower than using for loops
  //g_iw(iw_) << sum(g_k_iw(k_,iw_), k_ = k_mesh) / k_mesh.size();

}
