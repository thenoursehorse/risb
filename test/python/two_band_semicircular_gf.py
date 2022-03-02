#!/usr/bin/env python

from common import *
from risb.embedding_atom_diag import *
from risb.solver import Solver

class tests(unittest.TestCase):
 
    def test_seemicircular_gf_half_filling(self):
        t = 0.5
        beta = 10
        n_iw = 10*beta
        n_loops = 100
        n_orbitals = 2
        filling = 'half'

        gf_struct = [('up-0',[0]), ('up-1',[0]), ('down-0',[0]), ('down-1',[0])]
        
        emb_solver = EmbeddingAtomDiag(gf_struct)
        S = Solver(beta = beta, gf_struct = gf_struct, n_iw = n_iw, emb_solver = emb_solver)

        coeff = 0.0
        for U in [1]: #np.arange(0,13):

            J = coeff * U

            # Expression of mu for half and quarter filling
            if filling == 'half':
                mu = 0.5*U + 0.5*(U-2*J) + 0.5*(U-3*J)
            elif filling == 'quarter':
                mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2
            mu_expected = mu

            # Set the interacting Kanamori hamiltonian
            h_int = Operator()
            for o in range(0,n_orbitals):
                h_int += U*n('up-%s'%o,0)*n('down-%s'%o,0)
            for o1,o2 in product(list(range(0,n_orbitals)),list(range(0,n_orbitals))):
                if o1==o2: continue
                h_int += (U-2*J)*n('up-%s'%o1,0)*n('down-%s'%o2,0)
            for o1,o2 in product(list(range(0,n_orbitals)),list(range(0,n_orbitals))):
                if o2>=o1: continue;
                h_int += (U-3*J)*n('up-%s'%o1,0)*n('up-%s'%o2,0)
                h_int += (U-3*J)*n('down-%s'%o1,0)*n('down-%s'%o2,0)
            for o1,o2 in product(list(range(0,n_orbitals)),list(range(0,n_orbitals))):
                if o1==o2: continue
                h_int += -J*c_dag('up-%s'%o1,0)*c_dag('down-%s'%o1,0)*c('up-%s'%o2,0)*c('down-%s'%o2,0)
                h_int += -J*c_dag('up-%s'%o1,0)*c_dag('down-%s'%o2,0)*c('up-%s'%o2,0)*c('down-%s'%o1,0)

            # First guess for Lambda has mu on the diagonal
            for b in S.block_names:
                for inner in S.Lambda[b]:
                    inner = mu

            # Non-interacting part is the semicircular DOS
            for name, g0 in S.G0_iw:
                g0 << inverse( iOmega_n + mu - t**2 * SemiCircular(2*t) )

            # First guess for G
            S.G_iw << S.G0_iw

            eprint("U =", U, "J/U = ", coeff, "mu =", mu)
               
            # DMFT loop
            for i in range(n_loops):
                
                # Symmetrize
                symmetrize(S.R,S.block_names)
                symmetrize(S.Lambda,S.block_names)

                norm = 0
                R_old = deepcopy(S.R)
                Lambda_old = deepcopy(S.Lambda)

                S.solve(h_int = h_int, mu = mu)

                for block in S.block_names:
                    norm += np.linalg.norm(S.R[block] - R_old[block])
                    norm += np.linalg.norm(S.Lambda[block] - Lambda_old[block])

                if norm < 1e-6:
                    break
            
            eprint("cycles =", i, "norm =", norm)
            eprint("R =", S.R)
            eprint("Lambda =", S.Lambda)
            eprint("pdensity =", S.pdensity)
            eprint("G0_iw.density() =", S.G0_iw.density())
            eprint("Gqp_iw.density() =", S.Gqp_iw.density())
            eprint("G_iw.density() =", S.G_iw.density())
       
        mu_calculated = 0
        for block in S.block_names:
            mu_calculated += np.trace(S.Lambda[block]) / len(S.block_names)
        #R_expected = np.array([[0.987918]])
        #Lambda_expected = np.array([[mu_expected]])
        #
        assert are_close(mu_calculated, mu_expected, 1e-3), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        #for b in S.block_names:
        #    assert_arrays_are_close(R_expected, R[b], 1e-3)
        #    assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-3)

if __name__ == '__main__':
    unittest.main()
