#!/usr/bin/env python

from common import *
from risb.embedding_atom_diag import *
from risb.solver import Solver

class tests(unittest.TestCase):
 
    def test_semicircular_gf_half_filling(self):
        t = 0.5
        beta = 10
        n_iw = 10*beta
        n_loops = 100

        gf_struct = [ ['up', [0]], ['dn', [0]] ]
        
        emb_solver = EmbeddingAtomDiag(gf_struct)
        S = Solver(beta = beta, gf_struct = gf_struct, n_iw = n_iw, emb_solver = emb_solver)

        for U in [1.5]: #np.arange(0, 13):
            h_loc = U * n('up',0) * n('dn',0)
            mu = U / 2
            eprint("U =", U, "mu =", mu)
           
            # First guess for Lambda has mu on the diagonal
            for b in S.block_names:
                for inner in S.Lambda[b]:
                    inner = mu

            # Non-interacting part is the semicircular DOS
            for name, g0 in S.G0_iw:
                g0 << inverse( iOmega_n + mu - t**2 * SemiCircular(2*t) )

            # First guess for G
            S.G_iw << S.G0_iw

            for i in range(n_loops):
                
                norm = 0
                R_old = deepcopy(S.R)
                Lambda_old = deepcopy(S.Lambda)

                S.solve(h_int = h_loc, mu = mu)

                for block in S.block_names:
                    norm += np.linalg.norm(S.R[block] - R_old[block])
                    norm += np.linalg.norm(S.Lambda[block] - Lambda_old[block])

                if norm < 1e-6:
                    break
            
            eprint("cycles =", i, "norm =", norm)
            eprint("R =", S.R)
            eprint("Lambda =", S.Lambda)
            eprint("pdensity =", S.pdensity)
            eprint("density =", S.density())
            eprint("total_density =", S.total_density())
            eprint("G0_iw.density() =", S.G0_iw.density())
            eprint("Gqp_iw.density() =", S.Gqp_iw.density())
            eprint("G_iw.density() =", S.G_iw.density())
       
        mu_calculated = 0
        for block in S.block_names:
            mu_calculated += np.trace(S.Lambda[block]) / 2.0
        mu_expected = U/2.
        #R_expected = np.array([[0.987918]])
        #Lambda_expected = np.array([[mu_expected]])
        #
        assert are_close(mu_calculated, mu_expected, 1e-3), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        #for b in S.block_names:
        #    assert_arrays_are_close(R_expected, R[b], 1e-3)
        #    assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-3)


if __name__ == '__main__':
    unittest.main()
