import numpy as np
from itertools import product
from triqs.atom_diag import AtomDiag, act
from triqs.operators import Operator, c, c_dag

class EmbeddingAtomDiag:
    """
    Impurity solver of embedding space using atom_diag from TRIQS."

    Parameters
    ----------

    h_loc : TRIQS Operator
        Local Hamiltonian including interactions and quadratic terms.

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    """

    def __init__(self, 
                 h_loc, gf_struct):
        self._h_loc = h_loc
        self._gf_struct = gf_struct

        # Set structure of bath and embedding space
        # FIXME if doing ghost bath and loc are not the same size    
        self._gf_struct_bath = [(self._bl_loc_to_bath(bl), bl_size) for bl, bl_size in self._gf_struct]
        self._gf_struct_emb = self._gf_struct + self._gf_struct_bath

        # Set of fundamental operators
        self._fops = self.fops_from_gf_struct(self._gf_struct)
        self._fops_bath = self.fops_from_gf_struct(self._gf_struct_bath)
        self._fops_emb = self.fops_from_gf_struct(self._gf_struct_emb)
        
        # Do gf_struct as a map
        self._gf_struct_dict = self._dict_gf_struct(self._gf_struct)
        self._gf_struct_bath_dict = self._dict_gf_struct(self._gf_struct_bath)
        self._gf_struct_emb_dict = self._dict_gf_struct(self._gf_struct_emb)

    @staticmethod
    def _bl_loc_to_bath(bl):
        return 'bath_'+bl
    
    @staticmethod
    def _bl_bath_to_loc(bl):
        return bl.replace('bath_', '')
    
    @staticmethod
    def fops_from_gf_struct(gf_struct):
        return [(bl,i) for bl, bl_size in gf_struct for i in range(bl_size)]
    
    @staticmethod
    def _dict_gf_struct(gf_struct):
        return {bl: bl_size for bl, bl_size in gf_struct}
    
    def set_h_bath(self, Lambda_c):
        """
        Set the bath terms in the impurity Hamiltonian.

        Parameters
        ----------

        Lambda_c : optional, dict of ndarray
            Bath coupling. Each key in dictionary must follow gf_struct.

        """
        self._h_bath = Operator()
        for bl_bath, bl_bath_size in self._gf_struct_bath:
            bl = self._bl_bath_to_loc(bl_bath)
            for a,b in product(range(bl_bath_size), range(bl_bath_size)):
                self._h_bath += Lambda_c[bl][a,b] * c(bl_bath,b) * c_dag(bl_bath,a)

    def set_h_hybr(self, D):
        """
        Set the hybridization terms in the impurity Hamiltonian.

        Parameters
        ----------

        D : optional, dict of ndarray
            Hybridization coupling. Each key in dictionary must follow gf_struct.

        """
        self._h_hybr = Operator()
        for bl, loc_size in self._gf_struct:
            bl_bath = self._bl_loc_to_bath(bl)
            bath_size = self._gf_struct_bath_dict[bl_bath]
            for a, alpha in product(range(bath_size), range(loc_size)):
                self._h_hybr += D[bl][a,alpha] * c_dag(bl,alpha) * c(bl_bath,a)
                self._h_hybr += np.conj(D[bl][a,alpha]) * c_dag(bl_bath,a) * c(bl,alpha)

    def set_h_emb(self, Lambda_c, D, mu=None):
        """
        Set the terms in the impurity Hamiltonian to solve the embedding 
        problem.

        Parameters
        ----------
        
        Lambda_c : optional, dict of ndarray
            Bath coupling. Each key in dictionary must follow gf_struct.

        D : optional, dict of ndarray
            Hybridization coupling. Each key in dictionary must follow gf_struct.

        """
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        
        # For operators equal is copy, not a view
        self._h_emb = self._h_loc + self._h_bath + self._h_hybr
        
        if mu is not None:
            for bl, bl_size in self._gf_struct:
                for alpha in range(bl_size):
                    self._h_emb -= mu * c_dag(bl,alpha) * c(bl,alpha)

    def solve(self, fixed='half'):
        """
        Solve for the groundstate in the half-filled number sector of the 
        embedding problem.

        Parameters
        ----------

        fixed : optional, string
            How the Hilbert space is restricted. For fixed = 'half' 
            atom_diag will be passed n_min = n_max = half-filled. Options are
            'half'. Default is 'half'.

        todo : other restrictions, like none, for testing.
        """
        if fixed == 'half':
            # FIXME for ghost does this need to be different?
            M = int(len(self._fops_emb) / 2)
            self._ad = AtomDiag(self._h_emb, self._fops_emb, n_min=M, n_max=M)
            self._gs_vec = self._ad.vacuum_state
            self._gs_vec[0] = 1

        else:
            raise ValueError('Unrecognized fixed particle number !')
        
    def get_nf(self, bl):
        """
        Return the  f-electron density matrix from impurity.

        Parameters
        ----------

        bl : string
            Which block in `gf_struct` to return.

        """
        bl_bath = self._bl_loc_to_bath(bl)
        bl_size = self._gf_struct_bath_dict[bl_bath]
        Nf = np.zeros([bl_size, bl_size])
        for a, b in product(range(bl_size), range(bl_size)):
            Op = c(bl_bath, b) * c_dag(bl_bath, a)
            Nf[a,b] = self.overlap(Op, force_real=True)
        return Nf
    
    def get_nc(self, bl):
        """
        Return the  c-electron density matrix from impurity.

        Parameters
        ----------

        bl : string
            Which block in `gf_struct` to return.

        """
        bl_size = self._gf_struct_dict[bl]
        Nc = np.zeros([bl_size, bl_size])
        for alpha, beta in product(range(bl_size), range(bl_size)):
            Op = c_dag(bl, alpha) * c(bl, beta)
            Nc[alpha,beta] = self.overlap(Op, force_real=True)
        return Nc
    
    def get_mcf(self, bl):
        """
        Return the c,f-electron hybridization density matrix from impurity.

        Parameters
        ----------

        bl : string
            Which block in `gf_struct` to return.

        """
        bl_bath = self._bl_loc_to_bath(bl)
        bath_size = self._gf_struct_bath_dict[bl_bath]
        loc_size = self._gf_struct_dict[bl]
        Mcf = np.zeros([loc_size, bath_size], dtype=complex)
        for alpha, a in product(range(loc_size), range(bath_size)):
            Op = c_dag(bl, alpha) * c(bl_bath, a)
            Mcf[alpha,a] = self.overlap(Op, force_real=False)
        return Mcf
    
    def overlap(self, Op, force_real=True):
        """
        Return the expectation value an operator against the ground state of 
        the embedding problem.

        Parameters
        ----------

        Op : TRIQS Operator

        force_real : optional, bool
            Whether the result should be real or complex.

        """
        res = self._gs_vec @ act(Op, self._gs_vec, self._ad)
        if force_real:
            return res.real
        else:
            return res
    
    @property
    def ad(self):
        """
        The TRIQS AtomDiag instance. See atom_diag in the TRIQS manual.
        """
        return self._ad
    
    @property
    def gs_energy(self):
        """
        Ground state energy of the embedding problem.
        """
        return self._ad.gs_energy
    
    @property
    def gs_vector(self):
        """
        Ground state of the embedding problem.
        """
        return self._gs_vec
    
    @property
    def h_emb(self):
        """
        Embedding Hamiltonian as a TRIQS Operator.
        """
        return self._h_emb
    
    @property
    def h_bath(self):
        """
        Bath part of the embedding Hamiltonian as a TRIQS Operator.
        """
        return self._h_bath
    
    @property
    def h_hybr(self):
        """
        Hybridization part of the embedding Hamiltonian as a TRIQS Operator.
        """
        return self._h_hybr
    
    @property
    def h_loc(self):
        """
        Local Hamiltonian as a TRIQS Operator.
        """
        return self._h_loc