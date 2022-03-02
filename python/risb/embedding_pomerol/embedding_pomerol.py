import numpy as np
from copy import deepcopy
from triqs.operators import Operator, c, c_dag
from triqs.utility import mpi
from pomerol2triqs import PomerolED

class EmbeddingPomerol(object):

    def __init__(self, gf_struct, beta = 1e6):

        self.gf_struct = gf_struct
        self.beta = beta

        # Set up the structure
        self.str_to_int = {}
        self.str_to_int.update({self.gf_struct[b][0] : b for b in range(len(self.gf_struct))})

        self.fops_loc = []
        for b in self.gf_struct:
            fops = [(str(b[0]),o) for o in b[1]]
            self.fops_loc.append(fops)

        self.fops_bath = []
        for b in self.gf_struct:
            fops = [("bath_"+str(b[0]),o) for o in b[1]]
            self.fops_bath.append(fops)
       
        self.fops_emb = [item for sublist in self.fops_loc + self.fops_bath for item in sublist]

        # Conversion from TRIQS to Pomerol notation for operator indices
        # TRIQS: block_name, inner_index
        # Pomerol: site_label, orbital_index, spin_name
        self.index_converter = {}
        self.index_converter.update({fops : ("bath" if "bath_" in fops[0] else 'loc', fops[1], "down" if "dn" in fops[0] else "up")
                                    for fops in self.fops_emb})

        # Make PomerolED solver object
        self.ed = PomerolED(self.index_converter, verbose = True)

        # Density matrix blocks with small diagonal elements will be discarded
        self.ed.rho_threshold = 1e-6

    # ------------------------------------------------------------------
    
    def solve(self, ignore_symmetries = False):
        self.ed.diagonalize(self.h_emb, ignore_symmetries)
    
    # ------------------------------------------------------------------

    def set_h_emb(self, h_loc, Lambda_c, D, mu = 0): 
        #self.h_emb = deepcopy(h_loc)
        self.h_emb = h_loc

        for fops_l in self.fops_loc:
            for alpha in range(len(fops_l)):
                s = fops_l[alpha][0]
                o = fops_l[alpha][1]
                self.h_emb -= mu * c_dag(s,o) * c(s,o)

        for b,fops_b in enumerate(self.fops_bath):
            block = self.gf_struct[b][0]
            for a in range(len(fops_b)):
                s = fops_b[a][0]
                o = fops_b[a][1]
                for b in range(len(fops_b)):
                    ss = fops_b[b][0]
                    oo = fops_b[b][1]
                    if a == b:
                        self.h_emb += (Lambda_c[block][a,b] + mu) * c(ss,oo) * c_dag(s,o) # subtract off mu contribution
                    else:
                        self.h_emb += Lambda_c[block][a,b] * c(ss,oo) * c_dag(s,o)

        for b in range(len(self.fops_loc)):
            block = self.gf_struct[b][0]
            fops_l = self.fops_loc[b]
            fops_b = self.fops_bath[b]
            for a in range(len(fops_b)):
                s = fops_b[a][0]
                o = fops_b[a][1]
                for alpha in range(len(fops_l)):
                    ss = fops_l[alpha][0]
                    oo = fops_l[alpha][1]
                    self.h_emb += D[block][a,alpha] * c_dag(ss,oo) * c(s,o)
                    self.h_emb += np.conj(D[block][alpha,a]) * c_dag(s,o) * c(ss,oo)

    # ------------------------------------------------------------------

    def get_nf(self,block):
        b = self.str_to_int[block]
        fops = self.fops_bath[b]
        nf = np.empty([len(fops),len(fops)])
        for idx,i in enumerate(fops):
            for jdx,j in enumerate(fops):
                nf[idx,jdx] = self.ed.ensemble_average(i, j, self.beta).real
        return np.eye(nf.shape[0]) - nf
        #return nf
    
    def get_nc(self,block):
        b = self.str_to_int[block]
        fops = self.fops_loc[b]
        nc = np.empty([len(fops),len(fops)])
        for idx,i in enumerate(fops):
            for jdx,j in enumerate(fops):
                nc[idx,jdx] = self.ed.ensemble_average(i, j, self.beta).real
        return nc

    def get_mcf(self,block):
        b = self.str_to_int[block]
        fops_b = self.fops_bath[b]
        fops_l = self.fops_loc[b]
        mcf = np.empty([len(fops_l),len(fops_b)])
        for idx,i in enumerate(fops_l):
            for jdx,j in enumerate(fops_b):
                mcf[idx,jdx] = self.ed.ensemble_average(i, j, self.beta).real #FIXME for if h_loc is complex
        return mcf
