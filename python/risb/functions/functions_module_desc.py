from cpp2py.wrap_generator import *

module = module_(full_name = "risb.functions_module", doc = r"The functions in risb/solver python module", app_name = "risb")
module.add_imports(*['triqs.gf', 'triqs.operators'])
module.add_include("risb/functions/functions.hpp")

module.add_preamble("""
#include <cpp2py/converters/pair.hpp>
#include <cpp2py/converters/set.hpp>
#include <cpp2py/converters/std_array.hpp>
#include <cpp2py/converters/vector.hpp>
#include <cpp2py/converters/variant.hpp> // need this for convertering fundamental_operator_set
#include <triqs/cpp2py_converters/arrays.hpp>
#include <triqs/cpp2py_converters/fundamental_operator_set.hpp>
#include <triqs/cpp2py_converters/gf.hpp>
#include <triqs/cpp2py_converters/operators_real_complex.hpp>

using namespace risb::functions;
using dcomplex = std::complex<double>;
""")

for c_py, c_cpp, in (('Real','double'),('Complex','dcomplex')):
    c_type = "risb::functions::EigSystem<%s>" % c_cpp

    c = class_(
            py_type = "EigSystem%s" % c_py,
            c_type = c_type,
            doc = r"""""",
            hdf5 = False,
    )

    c.add_member(c_name = "val",
                c_type = "triqs::array:array<double,2>",
                read_only = True,
                doc = """Eigenvalues""")

    c.add_member(c_name = "vec",
                c_type = "triqs::array::array<%s,3>" % c_cpp,
                read_only = True,
                doc = """Eigenvectors""")

    c.add_member(c_name = "vec_dag",
                c_type = "triqs::array::array<%s,3>" % c_cpp,
                read_only = True,
                doc = """Inverse of eigenvectors""")

    module.add_class(c)


module.add_function ("std::pair<fundamental_operator_set, fundamental_operator_set> risb::functions::get_embedding_space (triqs::hilbert_space::fundamental_operator_set fops_local)", doc = r"""""")


module.add_function ("EigSystem<double> risb::functions::get_h_qp (std::vector<matrix<double> > R, std::vector<matrix<double> > lambda, array<double, 5> dispersion, double mu = 0)", doc = r"""""")
module.add_function ("EigSystem<dcomplex> risb::functions::get_h_qp (std::vector<matrix<double> > R, std::vector<matrix<double> > lambda, array<dcomplex, 5> dispersion, double mu = 0)", doc = r"""""")


module.add_function ("array<double,3> risb::functions::get_disp_R (std::vector<matrix<double> > R, array<double, 5> dispersion, EigSystem<double> h_qp)", doc = r"""""")
module.add_function ("array<dcomplex,3> risb::functions::get_disp_R (std::vector<matrix<double> > R, array<dcomplex, 5> dispersion, EigSystem<dcomplex> h_qp)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_ke (array<double,3> disp_R, array<double,3> vec_dag, array<double, 2> wks)", doc = r"""""")
module.add_function ("matrix<dcomplex> risb::functions::get_ke (array<dcomplex,3> disp_R, array<dcomplex,3> vec_dag, array<double, 2> wks)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_pdensity (array<double,3> vec, array<double,3> vec_dag, array<double, 2> wks)", doc = r"""""")
module.add_function ("matrix<double> risb::functions::get_pdensity (array<dcomplex,3> vec, array<dcomplex,3> vec_dag, array<double, 2> wks)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_d (matrix<double> pdensity, matrix<double> ke)", doc = r"""""")
module.add_function ("matrix<dcomplex> risb::functions::get_d (matrix<double> pdensity, matrix<dcomplex> ke)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_lambda_c (matrix<double> pdensity, matrix<double> R, matrix<double> lambda, matrix<double> D)", doc = r"""""")
module.add_function ("matrix<double> risb::functions::get_lambda_c (matrix<double> pdensity, matrix<dcomplex> R, matrix<double> lambda, matrix<dcomplex> D)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_lambda (matrix<double> R, matrix<double> D, matrix<double> lambda_c, matrix<double> Nf)", doc = r"""""")
module.add_function ("matrix<double> risb::functions::get_lambda (matrix<dcomplex> R, matrix<dcomplex> D, matrix<double> lambda_c, matrix<double> Nf)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_r (matrix<double> Mcf, matrix<double> Nf)", doc = r"""""")
module.add_function ("matrix<dcomplex> risb::functions::get_r (matrix<dcomplex> Mcf, matrix<double> Nf)", doc = r"""""")


module.add_function ("triqs::operators::many_body_operator_generic<double> risb::functions::get_h_emb (triqs::operators::many_body_operator_generic<double> h_loc, matrix<double> D, matrix<double> lambda_c, triqs::hilbert_space::fundamental_operator_set fops_local, triqs::hilbert_space::fundamental_operator_set fops_bath)", doc = r"""""")
module.add_function ("triqs::operators::many_body_operator_generic<dcomplex> risb::functions::get_h_emb (triqs::operators::many_body_operator_generic<dcomplex> h_loc, matrix<dcomplex> D, matrix<double> lambda_c, triqs::hilbert_space::fundamental_operator_set fops_local, triqs::hilbert_space::fundamental_operator_set fops_bath)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_pdensity_gf (gf<triqs::gfs::imfreq> g_z, matrix<double> R)", doc = r"""""")
module.add_function ("matrix<double> risb::functions::get_pdensity_gf (gf<triqs::gfs::imfreq> g_z, matrix<dcomplex> R)", doc = r"""""")


module.add_function ("matrix<double> risb::functions::get_ke_gf (gf<triqs::gfs::imfreq> g_z, gf<triqs::gfs::imfreq> delta_z, matrix<double> R)", doc = r"""""")
module.add_function ("matrix<dcomplex> risb::functions::get_ke_gf (gf<triqs::gfs::imfreq> g_z, gf<triqs::gfs::imfreq> delta_z, matrix<dcomplex> R)", doc = r"""""")


module.add_function ("gf<triqs::gfs::imfreq> risb::functions::get_delta_z (gf<triqs::gfs::imfreq> g0_z)", doc = r"""""")


module.add_function ("gf<triqs::gfs::imfreq> risb::functions::get_sigma_z (gf<triqs::gfs::imfreq> g_z, matrix<double> R, matrix<double> lambda, double mu = 0.0)", doc = r"""""")
module.add_function ("gf<triqs::gfs::imfreq> risb::functions::get_sigma_z (gf<triqs::gfs::imfreq> g_z, matrix<dcomplex> R, matrix<double> lambda, double mu = 0.0)", doc = r"""""")



module.generate_code()
