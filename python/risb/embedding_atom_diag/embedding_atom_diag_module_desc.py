from cpp2py.wrap_generator import *

# The module
module = module_(full_name="risb.embedding_atom_diag_module", 
                 doc=r"The impurity solver of embedding space using atom_diag in TRIQS python module", 
                 app_name="risb")

# Imports

# Add here all includes
module.add_include("risb/embedding_atom_diag/embedding_atom_diag.hpp")

module.add_include("<cpp2py/converters/string.hpp>")
module.add_include("<cpp2py/converters/pair.hpp>")
module.add_include("<cpp2py/converters/vector.hpp>")
module.add_include("<cpp2py/converters/map.hpp>")
module.add_include("<cpp2py/converters/variant.hpp>")

module.add_include("<triqs/cpp2py_converters.hpp>")
module.add_include("<triqs/cpp2py_converters/operators_real_complex.hpp>")

# Add here all namespaces
module.add_using("namespace triqs::arrays")
module.add_using("namespace triqs::operators")
module.add_using("namespace triqs::hilbert_space")
module.add_using("namespace risb::embedding_atom_diag")

# Add here anything to add in the C++ code at the start, e.g., that does not fit into
# the stuff above
module.add_preamble("""
""")

# Wrap class embedding_atom_diag
for c_py, c_cpp, in (('Real','false'),('Complex','true')):
    c_type = "risb::embedding_atom_diag::embedding_atom_diag<%s>" % c_cpp

    c = class_(
            py_type = "EmbeddingAtomDiag%s" %c_py,  # name of the python class
            c_type = c_type,   # name of the C++ class
            doc = r"""Impurity solver of embedding space using atom_diag in TRIQS python module.""",   # doc of the C++ class
            hdf5 = False,
    )
    
    c.add_constructor("(gf_struct_t gf_struct, double beta = 1e6)",
                      doc = "atom_diag autopartition constructor")

    c.add_method("%s::atom_diag_t get_ad()" % c_type, # FIXME this does not convert properly
                 doc = "The atomic_diag object")

    c.add_method("%s::block_matrix_t get_dm()" % c_type,
                 doc = "The density matrix")

    c.add_method("void solve ()",
                 doc = "Solve the embedding Hamiltonian")
    
    c.add_method("void set_h_emb (many_body_operator h_loc, std::map<std::string,matrix<double>> lambda_c, std::map<std::string, matrix<%s::scalar_t>> D, double mu = 0)" % c_type,
                 doc = "Sets the embedding Hamiltonian for the solver")
    
    c.add_method("double overlap (many_body_operator Op)",
                 doc = "Returns <Phi| Op |Phi>")
    
    c.add_method("matrix<double> get_nf (std::string block)",
                 doc = "Returns the density matrix of the f-fermions")
    
    c.add_method("matrix<double> get_nc (std::string block)",
                 doc = "Returns the density matrix of the c-fermions")
    
    c.add_method("matrix<%s::scalar_t> get_mcf (std::string block)" % c_type,
                 doc = "Returns the density matrix of the hybridization of the c and f-fermions")

    module.add_class(c)

module.generate_code()
