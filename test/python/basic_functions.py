#!/usr/bin/env python

from common import *
from risb.embedding_atom_diag import *

beta = 10
nkx = 10
orbitals_dim = 2

class tests(unittest.TestCase):

    def test_get_hqp(self):
        dispersion = build_cubic_dispersion(nkx, orbitals_dim)
        R, Lambda = build_mf_matrices(orbitals_dim)
        eig, vec = sc.get_h_qp(R, Lambda, dispersion)
        
    def test_get_ke(self):
        dispersion = build_cubic_dispersion(nkx, orbitals_dim)
        nk = dispersion.shape[2]
        R, Lambda = build_mf_matrices(orbitals_dim)
        eig,vec = sc.get_h_qp(R, Lambda, dispersion)
        disp_R = sc.get_disp_R(R, dispersion, vec)
        wks = fermi_fnc(eig, beta) / nk
        ke = sc.get_ke(disp_R, vec, wks)
    
    def test_get_pdensity(self):
        dispersion = build_cubic_dispersion(nkx, orbitals_dim)
        nk = dispersion.shape[2]
        R, Lambda = build_mf_matrices(orbitals_dim)
        eig, vec = sc.get_h_qp(R, Lambda, dispersion)
        wks = fermi_fnc(eig, beta) / nk
        pdensity = sc.get_pdensity(vec, wks)

    def test_get_d(self):
        dispersion = build_cubic_dispersion(nkx, orbitals_dim)
        nk = dispersion.shape[2]
        R, Lambda = build_mf_matrices(orbitals_dim)
        eig, vec = sc.get_h_qp(R, Lambda, dispersion)
        disp_R = sc.get_disp_R(R, dispersion, vec)
        wks = fermi_fnc(eig, beta) / nk
        ke = sc.get_ke(disp_R, vec, wks)
        pdensity = sc.get_pdensity(vec, wks)
        D = sc.get_d(pdensity, ke)
    
    def test_get_lambda_c(self):
        dispersion = build_cubic_dispersion(nkx, orbitals_dim)
        nk = dispersion.shape[2]
        R, Lambda = build_mf_matrices(orbitals_dim)
        eig, vec = sc.get_h_qp(R, Lambda, dispersion)
        disp_R = sc.get_disp_R(R, dispersion, vec)
        wks = fermi_fnc(eig, beta) / nk
        ke = sc.get_ke(disp_R, vec, wks)
        pdensity = sc.get_pdensity(vec, wks)
        D = sc.get_d(pdensity, ke)
        Lambda_c = sc.get_lambda_c(pdensity, R, Lambda, D)
    
    def test_get_h_emb(self):
        dispersion = build_cubic_dispersion(nkx, orbitals_dim)
        nk = dispersion.shape[2]
        R, Lambda = build_mf_matrices(orbitals_dim)
        eig, vec = sc.get_h_qp(R, Lambda, dispersion)
        disp_R = sc.get_disp_R(R, dispersion, vec)
        wks = fermi_fnc(eig, beta) / nk
        ke = sc.get_ke(disp_R, vec, wks)
        pdensity = sc.get_pdensity(vec, wks)
        D = sc.get_d(pdensity, ke)
        Lambda_c = sc.get_lambda_c(pdensity, R, Lambda, D)

        fops_local = [(s,o) for s,o in product(('up','dn'),list(range(1,2)))]
        [fops_bath, fops_emb] = get_embedding_space(fops_local)
        h_loc = 2.0 * n("up", 1) * n("dn", 1)
        h_emb = get_h_emb(h_loc, D, Lambda_c, fops_local, fops_bath)
    
    def test_solve_emb(self):
        dispersion = build_cubic_dispersion(nkx, orbitals_dim)
        nk = dispersion.shape[2]
        R, Lambda = build_mf_matrices(orbitals_dim)
        eig, vec = sc.get_h_qp(R, Lambda, dispersion)
        disp_R = sc.get_disp_R(R, dispersion, vec)
        wks = fermi_fnc(eig, beta) / nk
        ke = sc.get_ke(disp_R, vec, wks)
        pdensity = sc.get_pdensity(vec, wks)
        D = sc.get_d(pdensity, ke)
        Lambda_c = sc.get_lambda_c(pdensity, R, Lambda, D)

        fops_local = [(s,o) for s,o in product(('up','dn'),list(range(1,2)))]
        [fops_bath, fops_emb] = get_embedding_space(fops_local)
        h_loc = 0.5 * n("up", 1) * n("dn", 1)
        #h_loc = Operator()
        set_approx_zero(D)
        set_approx_zero(Lambda_c)
        h_emb = get_h_emb(h_loc, D, Lambda_c, fops_local, fops_bath)

        gf_struct = [ ["up", [1]], ["dn", [1]] ]
        emb_solver = EmbeddingAtomDiag(gf_struct)
        
        #Nf = emb_solver.get_nf()
        #Mcf = emb_solver.get_mcf()

        #Lambda_1 = get_lambda(R, D, Lambda_c, Nf);
        #R_1 = get_r(Mcf, Nf)

        #assert_arrays_are_close(R, R_1)
        #assert_arrays_are_close(Lambda, Lambda_1)


if __name__ == '__main__':
    unittest.main()
