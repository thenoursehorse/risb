import pyscf
import types
import numpy as np
numpy = np
import scipy
from copy import deepcopy


from ase.build import bulk
#from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc import gto, scf, dft, df

# Heavily borrows from https://github.com/SebWouters/QC-DMET/
# and https://github.com/hungpham2017/pDMET, as well as the pyscf code itself.

# s1 : The overlap matrix between the non-orthonormal atomic orbitals. See 
# the discussion below Szabo and Ostlund, eq. 3.136. In pyscf this is obtained
# as mf.get_ovlp() or from mol/cell as mol.intor('int1e_ovlp') or 
# cell.pbc_intor('int1e_ovlp', kpts=). It can also be obtained in sph or cubic
# coordinates etc with, e.g.,  'int1e_ovlp_sph'. Note that get_ovlp() does 
# not enforce s1 to be hermitian (but it does check and warn if any elements
# are greater than some threshold). It may be worth making s1 hermitian after
# obtaining it with s1 = np.tril(s1) + np.tril(s1,-1).conj().T
#
# C matrix in Szabo and Ostlund, eq 3.133 is given by ao2mo = mf.mo_coeff
# The row index is the molecular orbital index, and the col is the atomic 
# orbital subscript. It is the transformation matrix from the atomic orbital 
# basis to the molecular orbitals basis.
#
# The density matrix in the non-orthonormal atomic orbital basis for 
# restricted HF is D = 2 * ao2mo_occ * ao2mo_occ.T Szabo and Ostlund, eq 3.145
# Only the indicies with occupied molecular orbitals enter into ao2mo. The 
# occupations of the molecular orbitals are given by mf.mo_occ. In pyscf D 
# is calculated with mf.make_rdm1.
#
# The rotation matrices between bases are named as basis_one2basis_two (ao2lo) 
# where basis_one are the rows and basis_two are the columns. Therefore, each 
# column is a vector in basis_two with each element a coefficient in basis_one.
#
# lo : The local orbital basis. It should be the same size as the atomic 
# orbital basis because we construct them to be unitary.
#
# imp : The impurity orbital basis. This will be the size of the selected 
# correlated space. We in principle can create multiple projections to 
# multiple impurities. The 'rotation' matrices in this case aren't unitary, 
# and instead will have the number of rows the size of the larger basis 
# (atomic orbitals, or local orbitals etc), and the number of columns the 
# size of the correlated space.
#
# R0 : A quantity in the reference cell R=0. That is, a k-space quantity F(k)  
# that has been Fourier transformed to R=0 with exp(i R=0 k) = 1, so that 
# F(R=0) 1/N sum_k F(k)

# TODO 
# allow open shell HF etc. Currently only have restricted HF.

# TODO
# Allow ao2lo etc to be on the IBZ. But need to figure out the correct way to 
# transform from the IBZ to the BZ in the lo basis and symmetrize.

# TODO
# If do not use psuedo potential then have to treat the core electrons 
# correctly and freeze them. But for now we just assume that the core 
# electrons are not in the atomic orbital basis. See pyscf get_roothaan_fock
# for maybe an idea on how to split the space.

# NOTE
# pyscf.lib.tag_array allows to add attributes to a numpy array

def check_pbc(mol):
    #return getattr(mol, 'dimension', 0) > 0
    return isinstance(mol, pyscf.pbc.gto.Cell)

def check_symm(kpts):
    return isinstance(kpts, pyscf.pbc.symm.symmetry.Symmetry)

def get_ref_mol(mol, reference_basis=None):
    '''
    Make a copy of the molecule or cell but in reference_basis. Usually this 
    would be a smaller basis than what mol is.

    Args:
        mol : A cell or molecule object.
        
        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.
    
    Returns:
        ref_mol : Copy of mol in reference_basis.
    '''
    ref_mol = mol.copy()
    if reference_basis is not None:
        old_verbose = deepcopy(ref_mol.verbose)
        ref_mol.verbose = 0
        ref_mol.build(dump_input=False, parse_arg=False, basis=reference_basis)
        ref_mol.verbose = old_verbose
    # else ref_mol.basis but ref_mol is already in this basis
    # Could maybe just use iao.reference_mol but it uses pyscf.gto and I'm not 
    # sure if that will break the cell object (probably not).
    return ref_mol

def get_ref_orbs(mol, reference_basis=None, corr_orbs_labels=None):
    '''
    Gives the indices of the the atomic orbital labels of reference_basis 
    (or corr_orbs_labels) in the original larger atomic basis of mol.

    Args:
        mol : A cell or molecule object.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.

        corr_orbs_labels : (default None) A list of orbitals. Partial 
            matches are OK, such that "Ni" will be all Ni orbitals, and 
            "Ni 3d" will be all 3d orbitals on Ni.

        One of reference_basis or corr_orbs_labels must not be None.
    
    Returns:
        ref_orbs : A dictionary where the key is the index in the atomic basis 
            and the value is the orbital label.

        ref_orbs_idx : The indicies of reference_basis in the original atomic 
            orbital basis.

        ref_orbs_com_idx : The compliment indices of reference_basis in the 
            original atomic orbital basis.
    '''
    if (reference_basis is not None) and (corr_orbs_labels is not None):
        raise ValueError(f"One of reference basis or corr_orbs_labels must be None !")

    # This could be replaced with ref_mol.search_ao_label(labels) 
    if corr_orbs_labels is not None:
        ref_orbs = {idx: s for idx,s in enumerate(mol.ao_labels()) if any(xs in s for xs in corr_orbs_labels)}
    elif reference_basis is not None:
        ref_mol = get_ref_mol(mol=mol, reference_basis=reference_basis)
        ref_orbs = {idx: s for idx,s in enumerate(mol.ao_labels()) if any(xs in s for xs in ref_mol.ao_labels())}
    else:
        raise ValueError(f"Both reference basis and corr_orbs_labels can not be None !")
    
    ref_orbs_idx = [*ref_orbs]
    ref_orbs_com_idx = [i for i in range(len(mol.ao_labels())) if i not in ref_orbs_idx]
    assert ( len(ref_orbs_idx) + len(ref_orbs_com_idx) ) == mol.nao_nr()
    return ref_orbs, ref_orbs_idx, ref_orbs_com_idx
    
def orthogonalize(c, s1, has_pbc, otype='lowdin'):

    '''
    Orthongonalize the basis C.

    Args:
        c : The coefficients that define the basis c, written in the atomic 
            orbital basis.
        
        s1 : The overlap matrix between the atomic orbitals.

        has_pbc : A boolean where True is if c and s1 are on a lattice (are 
            3-dim array if lattice, 2-dim array if molecule).

        otype : (Default lowdin) The orthogonalization algorithm. Choices are 
            lowdin, knizia.
    '''
    # doi:10.1021/ct400687b appendix C
    def knizia(c, s):
        # Orthogonalize metric c.T s c
        return c @ scipy.linalg.fractional_matrix_power(c.conj().T @ s @ c, -0.5)
    
    if otype == 'lowdin':
        from pyscf.lo.orth import vec_lowdin
        orth = vec_lowdin
    elif otype == 'schmidt':
        from pyscf.lo.orth import vec_schmidt
        orth = vec_schmidt
    elif otype == 'knizia':
        orth = knizia
    else:
        raise ValueError(f"{otype} not implemented for orthogonalize !")

    nkpts = c.shape[0]
    if has_pbc:
        for k in range(nkpts):
            c[k] = orth(c[k], s1[k])
    else:
        c = orth(c, s1)
    return c

def get_nested_function(outer, inner, **freevars):
    '''
    Creates a function from the nested function within another function.
    This will only go 1 nest deep.
    From https://stackoverflow.com/a/40511805

    Args:
        outer : The outer function.

        inner : The name, as a string, of the inner nested function.

        free_vars : The variables and their values of any free variables 
            defined in the outer function that the inner function requires.

    Return:
        A types.FunctionType function of the inner function.

    Example:
        inner_function = get_nested_function(outer_function, 'inner', a=1, b=2, c=3)
    '''
    # Deals with the free variables (not inside nested funciton scope)
    def freeVar(val):
        def nested():
            return val
        return nested.__closure__[0]

    if isinstance(outer, (types.FunctionType, types.MethodType)):
        outer = outer.__code__
    for const in outer.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == inner:
            return types.FunctionType(const, globals(), None, None, tuple(
                freeVar(freevars[name]) for name in const.co_freevars))
    raise ValueError(f"{inner} is not a nested function of {outer} !")

def make_iaos(s1, s2, s12, mo, lindep_threshold=1e-8):
    '''
    Make the unorthonormalized intrisic atomic orbitals. This is a wrapper to 
    expose the nested function make_iaos that is inside pyscf.lo.iao.iao.
    [Ref. JCTC, 9, 4834]

    Args:
        s1 : The 1 electron overlap matrix of the atomic basis orbitals.

        s2 : The 1 electron overlap matrix in a different (smaller) reference 
            basis. The reference basis should be a subset of the basis for s1.

        s12 : The cross overlap between s1 and s2.

        mo : The rotation matrix from the molecular orbitals to the atomic 
            basis orbitals of s1. It should be a square matrix that only 
            contains the occupied orbitals (or unoccupied for holes/virtual).
    
    Returns:
        ao2iao : The rotation matrix from the atomic orbitals to the intrinsic 
            atomic orbitals.
    '''
    # HACK
    # To get access to make_iaos(s1, s2, s12, mo) defined inside iao
    global vec_lowdin
    from pyscf.lo.orth import vec_lowdin
    _make_iaos = get_nested_function(pyscf.lo.iao.iao, 'make_iaos', lindep_threshold=lindep_threshold)
    return _make_iaos(s1=s1, s2=s2, s12=s12, mo=mo)
  
# Adapted from https://github.com/SebWouters/QC-DMET iao_helper.py
def get_ao2pao(mf, ao2lo, reference_basis=None):
    '''
    The unorthonormalized transformation matrix from the atomic orbital basis 
    to the projected atomic orbital basis (the compliment of the reference
    basis).

    Args:
        mf : A mean-field object from HF or DFT.

        ao2lo : 2-dim (molecule) or 3-dim (lattice) rotation matrix from the 
            atomic orbital basis to the reference basis. If on a lattice it 
            must be on the full BZ.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is the unitary 
        transformation into the local orbital basis.
    '''
    
    if reference_basis is None:
        reference_basis = mf.mol.basis
    
    num_ao = mf.mol.nao_nr()
    num_lo = ao2lo.shape[-1]
    
    # Check if we are a lattice or molecule 
    has_pbc = check_pbc(mf.mol)

    s1 = mf.get_ovlp()
    if has_pbc:
        has_symm = check_symm(mf.kpts)
        if has_symm:
            s1 = np.asarray(mf.kpts.transform_1e_operator(s1))
            kpts = mf.kpts.kpts
        else:
            kpts = mf.kpts
        nkpts = len(kpts)

    if len(ao2lo) != nkpts:
        raise ValueError(f'ao2lo must be on the full BZ with {nkpts} points, but got {len(ao2lo)} !')

    # Get the compliment of iao
    if has_pbc: 
        dm_lo = pyscf.lib.einsum('kij,kjl->kil', ao2lo, ao2lo.conj().transpose((0,2,1)) )
        mx = pyscf.lib.einsum('kij,kjl,klm->kim', s1, dm_lo, s1)
        ao2pao = np.empty(shape=(nkpts, num_ao, num_ao-num_lo), dtype=ao2lo.dtype)

        for k in range(nkpts):
            # Projection info:
            # Projector P onto a subapce C of H maps every vector v in C to itself
            # with eigenvalue 1. Every vector v not in C maps to a vector in C. Hence, 
            # for any vector v = Pv + (1-P)v, Pv is in C and v-Pv is in the nullspace
            # of P. We can see this by takig Pv = P^2v + Pv - P^2v = (Pv as expected 
            # because P^2 = P) or can say that P(1-P)v = 0. Hence, every vector can be 
            # written as a sum of a vector in C with eigenvalue 1, and a vector in the 
            # nullspace of P with eigenvalue 0. Hence, the eigs below are the union of
            # the the eigenbasis of C and a basis in the null space of C.

            # s1 dm s1 vecs = eigs s1 vecs    (1)    P s1 vecs = eigs s1 vecs
            #
            # vec+ s1 dm s1 vec = eigs        (2)
            #
            # vecs+ s1 vecs = 1               (3)
            #
            # eigs are 0 or 1, where 0 is a compliment orbital in the ao basis
            #
            # Hence, P = s1 dm is the projector onto the local orbital space from eq 1.
            # The vectors in C (local space) will be eigenvectors s1 vecs (unchanged) and
            # the rest will be nullspace vectors of C with eigenvalue 0.

            # Below is diagonalizing the projector P = s1 dm but taking into account
            # the non-orthogonality of the atomic orbitals (s1 is not the identity).
            # The rotation into the local space is orthogonal as ao2lo+ s1 ao2lo = 1,
            # but ao2lo+ ao2lo != 1. From Szabo and Ostlund this is Eq. 3.174 with
            # symmetric orthogonalization X=s1^1/2 and C=ao2lo. Only C'=XC forms
            # an orthogonal space.

            eigs, vecs = scipy.linalg.eigh(a=mx[k], b=s1[k])
            # eigs are sorted from lowest to highest
            ao2pao[k,...] = vecs[..., :num_ao-num_lo]

            # pg. 223 Saebo and Pulay, Annu. Rev. Phys. Chem 1993 44:213-36
            # doi: 10.1146/annurev.pc/44.100193.001241 
            # Projector onto the compliment of the local space in the atomic orbital basis
            # P = np.eye(s1[k].shape[0]) - s1[k] @ dm_lo[k]
            # So isn't the compliment P s1^1/2 mo mo+ s1^1/2 P ? 
            # But then it is in the ao basis and we want it to be in the basis where it is an
            # eigenvector of P. I can't think of a way around doing the diagonalization, either on
            # the compliment projector taking eigenvalues 1 or the iao projector as above.
    else:
        dm_lo = ao2lo @ ao2lo.conj().T
        mx = s1 @ dm_lo @ s1
        eigs, vecs = scipy.linalg.eigh(a=mx, b=s1)
        ao2pao = vecs[..., :num_ao - num_lo]

    # NOTE 
    # It isn't clear to me if you need to redo the iao for the complement, 
    # I just do because Seb does it in their code. Surely the projected atomic
    # orbitals are already obtained at this line and all that has to be done is to
    # orthogonalize? Doing iao on the projected part will just make polarized localized 
    # orbitals in the compliment. Since we never actually use them, I don't think it matters, 
    # except that it might change the shape of the correlated atomic orbitals when we 
    # orthgonalize (but if it does not in any appreciable way?, and it doesn't change for NiO).

    # Note that I do think this is required for DMET that uses an interacting bath, because 
    # then the compliment space will be important. I keep this here because it doesn't matter
    # for slave-bosons/DMFT projection methods, but it might matter for DMET if anyone else 
    # ever wants to use this.

    # Get the overlap matrices at the indices of the complement of the reference basis
    ref_orbs, ref_orbs_idx, ref_orbs_com_idx = get_ref_orbs(mol=mf.mol, reference_basis=reference_basis)
    s13 = s1[..., :, ref_orbs_com_idx]
    s3 = s13[..., ref_orbs_com_idx, :]
    
    # Construct iaos in the compliment
    if has_pbc:
        for k in range(nkpts):
            ao2pao[k] = make_iaos(s1=s1[k], s2=s3[k], s12=s13[k], mo=ao2pao[k])
    else:
        ao2pao = make_iaos(s1=s1, s2=s3, s12=s13, mo=ao2pao)

    return ao2pao

# Adapted from https://github.com/SebWouters/QC-DMET iao_helper.py
def get_ao2lo(mf, reference_basis=None):
    '''
    The transformation matrix at each k-point from the atomic orbital basis 
    to the local orbital basis.

    Args:
        mf : A mean-field object from HF or DFT. It can be an object that 
            uses symmetry. 

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is the unitary 
        transformation into the local orbital basis. If on a lattice it 
        returns the transformation on the full BZ.
    '''

    if reference_basis is None:
        reference_basis = mf.mol.basis
    
    # Check if we are a lattice or molecule 
    has_pbc = check_pbc(mf.mol)
    
    ao2mo = mf.mo_coeff
    mo_occ = mf.mo_occ

    # Get overlap, the transform from mo to ao, and the mo occupations
    # If it is a symmetry object we put these onto the entire BZ 
    # instead of the IBZ.
    s1 = mf.get_ovlp()
    if has_pbc:
        has_symm = check_symm(mf.kpts)
        if has_symm:
            s1 = np.asarray(mf.kpts.transform_1e_operator(s1))
            ao2mo = np.asarray(mf.kpts.transform_mo_coeff(ao2mo))
            mo_occ = np.asarray(mf.kpts.transform_mo_occ(mo_occ))
            kpts = mf.kpts.kpts
        else:
            kpts = mf.kpts
        nkpts = len(kpts)

    # Get only the mo indices that are occupied
    if has_pbc:
        ao2mo_occ = []
        for k in range(nkpts):
            ao2mo_occ.append( ao2mo[k][:,mo_occ[k]>0.01] )
    else:
        ao2mo_occ = ao2mo[:,mo_occ[k]>0.01]

    # Get the intrinsic atomic orbitals (IAO)
    otype = 'lowdin'
    #otype = 'knizia'
    ao2iao = pyscf.lo.iao.iao(mf.mol, ao2mo_occ, minao=reference_basis, kpts=kpts)
    ao2iao = orthogonalize(c=ao2iao, s1=s1, has_pbc=has_pbc, otype=otype)

    num_ao = mf.mol.nao_nr()
    num_lo = ao2iao.shape[-1]
    
    # Get the compliment, projected atomic orbitals
    if num_ao != num_lo:
        ao2pao = get_ao2pao(mf=mf, ao2lo=ao2iao, reference_basis=reference_basis)
        ao2pao = orthogonalize(c=ao2pao, s1=s1, has_pbc=has_pbc, otype=otype)

        # Contrusct the localized orbitals
        ref_orbs, ref_orbs_idx, ref_orbs_com_idx = get_ref_orbs(mol=mf.mol, reference_basis=reference_basis)
        if (ao2iao.dtype == np.complex_) or (ao2pao.dtype == np.complex_) or (s1.dtype == np.complex_):
            lo_dtype = np.complex_
        else:
            lo_dtype = np.float_
        ao2lo = np.empty(shape=s1.shape, dtype=lo_dtype)
        ao2lo[...,:,ref_orbs_idx] = ao2iao
        ao2lo[...,:,ref_orbs_com_idx] = ao2pao
    
        # FIXME reorder here, I don't think we have to because the occupied orbitals are 
        # in the order of reference_basis, while the virtual pao are not in whatever order 
        # that eigh returns. But I don't think we really care about them?
        #for k in range(nkpts):
        #    ao2lo[k] = resort_orbitals(mol=mf.mol, ao2lo=ao2lo[k], k=k)

        ao2lo = orthogonalize(c=ao2lo, s1=s1, has_pbc=has_pbc, otype=otype)
    else:
        ao2lo = ao2iao
    
    # Check is actually orthogonal
    if has_pbc:
        should_be_1 = pyscf.lib.einsum('kij,kjl,klm->kim', ao2lo.conj().transpose((0,2,1)), s1, ao2lo)
        for k in range(nkpts):
            #print(np.linalg.norm(should_be_1[k] - np.eye( should_be_1[k].shape[0])))
            error = np.linalg.norm(should_be_1[k] - np.eye( should_be_1[k].shape[0]))
            if error > 1e-10:
                raise ValueError(f"ao2lo at k={k} is not orthogonalized with norm={error} !")
    else:
        should_be_1 = ao2lo.conj().T @ s1 @ ao2lo
        error = np.linalg.norm(should_be_1 - np.eye( should_be_1.shape[0]))
        if error > 1e-10:
            raise ValueError(f"ao2lo is not orthogonalized with norm={error} !")
    
    # Can only use this if can figure out how to transform ao2lo IBZ to full BZ, probably
    # something like kpts.transform(mo_coeff @ ao2lo @ ao2lo.conj()) or something might work
    #if has_symm:
    #    ao2lo = ao2lo[mf.kpts.ibz2bz,...]

    return ao2lo

def get_ao2corr(mf, corr_orbs_labels, ao2lo=None, reference_basis=None):
    '''
    The projection matrix at each k-point from the atomic orbital basis 
    to the correlated orbitals. It can be a non-kpoint object as well 
    for molecules.

    Args:
        mf : A mean-field object from HF or DFT. It can be an object that 
            uses symmetry. 
        
        corr_orbs_labels : (default None) A list of orbitals. Partial 
            matches are OK, such that "Ni" will be all Ni orbitals, and 
            "Ni 3d" will be all 3d orbitals on Ni.
        
        ao2lo : (default None) The rotation matrices at each k-point from 
            the atomic orbital basis to the local basis. If None then 
            it will construct ao2lo.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.

    Returns:
        ao2corr : A 2-dim (molecule) or 3-dim (lattice) array that is the unitary 
            transformation into the local orbital basis. If on a lattice it 
            returns the transformation on the full BZ.

        ao2corr_com : The compliment of ao2corr.
    '''
    if ao2lo is None:
        if reference_basis is None:
            raise ValueError("There must be a reference basis to construct ao2lo !")
        ao2lo = get_ao2lo(mf=mf, reference_basis=reference_basis)

    corr_orbs, corr_orbs_idx, corr_orbs_com_idx = get_ref_orbs(mol=mf.mol, corr_orbs_labels=corr_orbs_labels)
    return ao2lo[..., corr_orbs_idx], ao2lo[..., corr_orbs_com_idx]
    
def get_lo2corr(mf, corr_orbs_labels):
    '''
    The projection matrix at each k-point from the local orbital basis 
    to the correlated orbitals. Since the lo basis is the computational basis 
    these are just identity matrices.
    
    Args:
        mf : A mean-field object from HF or DFT. It can be an object that 
            uses symmetry. 
        
        corr_orbs_labels : (default None) A list of orbitals. Partial 
            matches are OK, such that "Ni" will be all Ni orbitals, and 
            "Ni 3d" will be all 3d orbitals on Ni.

    Returns:
        lo2corr : A 2-dim (molecule) or 3-dim (lattice) array that is the unitary
            transformation into the local orbital basis. If on a lattice it 
            returns the transformation on the full BZ.

        lo2corr_com : The compliment of lo2corr.
    '''
    corr_orbs, corr_orbs_idx, corr_orbs_com_idx = get_ref_orbs(mol=mf.mol, corr_orbs_labels=corr_orbs_labels)

    lo2corr = np.zeros(shape=(len(mf.kpts.kpts), mf.mol.nao_nr(), len(corr_orbs_idx)))
    lo2corr[...,corr_orbs_idx,:] = np.eye(len(corr_orbs_idx))
    
    lo2corr_com = np.zeros(shape=(len(mf.kpts.kpts), mf.mol.nao_nr(), len(corr_orbs_com_idx)))
    lo2corr_com[...,corr_orbs_com_idx,:] = np.eye(len(corr_orbs_com_idx))
    
    # Will give the same as get_ao2corr, could maybe add a test for this
    #ao2corr = pyscf.lib.einsum('kij,kjl->kil', ao2lo, lo2corr)
    #ao2corr_com = pyscf.lib.einsum('kij,kjl->kil', ao2lo, lo2corr_com)

    return lo2corr, lo2corr_com    

def get_dm_rotated(mf, ao2b):
    '''
    Get the density matrix rotated into the basis of b from the atomic orbital 
    basis.

    Args:
        mf : A mean-field object from HF or DFT.

        ao2b : The transformation matrix from the atomic orbital basis to the 
            basis b. If it is on a lattice it must be on the full BZ. It can 
            be square (project onto some orbitals).

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is the density matrix 
        in the basis b.
    '''
    has_pbc = check_pbc(mf.mol)
    
    s1 = mf.get_ovlp()
    
    dm_ao = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    
    if has_pbc:
        has_symm = check_symm(mf.kpts)
        if has_symm:
            s1 = np.asarray(mf.kpts.transform_1e_operator(s1))
            dm_ao = np.asarray(mf.kpts.transform_dm(dm_ao))
        dm = np.zeros(shape=(dm_ao.shape[0], ao2b.shape[-1], ao2b.shape[-1]), dtype=np.complex_)

    # einsum is very slow for large k and # of orbs
    #dm = lib.einsum('...ij,...jk,...kl,...lm,...mn->...in', np.transpose(ao2b.conj(), (0,2,1)), s1, dm_ao, s1, ao2b)
    if has_pbc:
        for k in range(dm_ao.shape[0]):
            dm[k] = ao2b[k].conj().T @ s1[k] @ dm_ao[k] @ s1[k] @ ao2b[k]
    else:
        dm = ao2b.conj().T @ s1 @ dm_ao @ s1 @ ao2b

    return dm

def get_1e_operator_rotated(mf, ao2b, mat_1e):
    '''
    Get the one-electron (1e) matrix rotated into the basis of b from the 
    atomic orbital basis.
    
    Args:
        mf : A mean-field object from HF or DFT.

        ao2b : The transformation matrix from the atomic orbital basis to the 
            basis b. If it is on a lattice it must be on the full BZ. It can 
            be square (project onto some orbitals).

        mat_1e : The 1e matrix in the atomic orbital basis.

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is a 1e matrix 
        in the basis b.
    '''
    has_pbc = check_pbc(mf.mol)
    
    if has_pbc:
        has_symm = check_symm(mf.kpts)
        if has_symm:
            mat_1e = np.asarray(mf.kpts.transform_1e_operator(mat_1e))
    
    return pyscf.lib.einsum('...ij,...jk,...kl->...il', np.transpose(ao2b.conj(), (0,2,1)), mat_1e, ao2b)
    
def get_mat_R0(mf, mat):
    '''
    Get a matrix on the k-grid at R=0 (Fourier phase factor exp(iRk) = 1).

    Args:
        mf : A mean-field object from HF or DFT. It must be on the full BZ.

        mat : A density matrix at each k-point on the k-grid.

    Returns:
        An matrix at R=0.
    '''
    has_pbc = check_pbc(mf.mol)
    assert (has_pbc), f"mat should be 3-dim (lattice), got: {mat.shape}"
    
    has_symm = check_symm(mf.kpts)
    if has_symm:
        #dm_loc = mf.kpts.dm_at_ref_cell(dm_ibz=dm)
        if len(mat) != len(mf.kpts.kpts):
            raise ValueError(f"mat must be on the full BZ with {len(mf.kpts.kpts)} points, but got {len(mat)} !")
    
    # FIXME should we be using cell.get_lattice_Ls instead?    
    return pyscf.lib.einsum('kij->ij', mat) / len(mat)

def save_orbs_molden(mol, ao2b, filename='local_orbitals.molden', occ=None, energy=None, symm_labels=None, for_iboview=True):
    '''
    Output orbitals in the molden format.

    Args:
        mol : A cell or molecule object. If symm is None, then mol cannot 
            have mol.symmetry = True unless ao2b is the matrix to the 
            molecular orbitals because then the symmetry labels will make no 
            sense.

        ao2b : The rotation matrix from the atomic orbital basis to another 
            basis (not sure if this can be rectangular).

        filename : (default local_orbitals.molden) The output filename.

        occ : (default None) A list of the occupation of each orbital. Only 
            makes sense if the density matrix in the basis of the orbitals is 
            diagonal. If None then assumes are molecular orbitals and will 
            give that occupation.

        energy : (default None) A list of the energy of each orbital. Same 
            discussion as occ applies here.

        symm_labels : (default None) A list of the symmetry labels of each 
            orbital. 
    '''
    if len(ao2b.shape) > 2:
        raise ValueError('Can only output orbitals to molden at a single R point, but got coefficients on a grid !')

    # HACK
    # For iboview:
    # Even though molden outputs the core electrons when using psuedo 
    # potentials (ECP enabled), iboview does not recognize this.
    # Since mol.atom_charge has the core electrons screened out (subtracted), 
    # the atom symbol should then be charge(atom_charge)+core_charge.
    mol_copy = mol.copy()
    if for_iboview:
        for i in range(len(mol_copy._atm)):
            mol_copy._atm[i][0] += mol_copy.atom_nelec_core(i)

    # ignore_h prevents mol from getting rebuilt and truncating l>5 coefficients.
    from pyscf.tools import molden
    with open(filename,'w') as f:
            molden.header(mol=mol_copy, fout=f, ignore_h=False)
            molden.orbital_coeff(mol=mol, fout=f, mo_coeff=ao2b, ene=energy, occ=occ, symm=symm_labels)
    
def fermi_smearing(eks, sigma=0.01, mu=0):
    beta = 1.0/sigma
    return 1.0 / (np.exp(beta * (eks - mu)) + 1.0)
    
def gaussian_smearing(eks, sigma=0.01, mu=0):
    beta = 1.0/sigma
    return 0.5 * scipy.special.erfc(beta * (eks - mu))
    
def get_mu(eks, nocc, smearing_fnc=fermi_smearing, sigma=0.01, mu0=0):
    def nocc_cost_fnc(mu, eks, nocc, sigma):
        occ = smearing_fnc(eks, sigma, mu)
        return (occ.sum() - nocc)**2
    res = scipy.optimize.minimize(nocc_cost_fnc, mu0, args=(eks,nocc,sigma), method='Powell',
                                  options={'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 10000})
    return res.x

# TODO The self-consistency part. I will have to rotate from lo2ao and then probably ao2mo.
# TODO check whether we are on a k-grid or not
# TODO allow unrestricted/openshell HF
# TODO save out to hdf5
class LocalOrbitals(object):
    def __init__(self, mf, reference_basis, corr_orbs_labels, filename=None):
        self.mf = mf
        self.reference_basis = reference_basis
        self.corr_orbs_labels = corr_orbs_labels
        self.filename = filename
        if self.filename is None:
            self.filename = ''.join(list(dict(mf.mol.atom).keys()))

        # FIXME check for restricted/unrestricted 
        # and set a flag accordingly to handle the cases properly.

        self.nelec = mf.mol.nelec
        #self.nelec = self.mf.mol.tot_electrons()

        # Atomic orbital basis rotated into local orbitals 
        self.ao2lo = None
        self.ao2lo_R0 = None
        self.dm_lo = None
        self.dm_lo_R0 = None
        self.nelec_lo = None

        # Subset of local orbitals identified as correlated
        self.ao2corr = None
        self.ao2corr_com = None
        self.ao2corr_R0 = None
        self.dm_corr = None
        self.dm_corr_R0 = None
        self.nelec_corr = None
        self.lo2corr = None
        self.lo2corr_com = None
        self.lo2corr_R0 = None

        # The molecular orbitals returned by the scf
        self.ao2mo = self.mf.kpts.transform_mo_coeff(self.mf.mo_coeff)
        self.ao2mo_R0 = get_mat_R0(mf=self.mf, mat=self.ao2mo)

        # The Hamiltonian quantities in the lo basis
        self.fock = None
        self.eri = None
        self.mu = None
    
    def kernel(self):
        self.make_corr_proj()
        self.save_orbs()
        self.make_fock()
        self.check_fock_density()
        self.make_eri_local()
        self.print_out()

    def make_local_rotation(self):
        """
        Unitary rotation from atomic orbital basis to the local orbital basis.

        Sets:
            ao2lo : The rotation into the local orbital basis (at each k-point 
                if on a k-grid).

            ao2lo_R0 : The rotation into the local orbitals at the reference 
                cell R=0 if on a k-grid.
            
            dm_lo : The one-electron density matrix in the local orbitals (at 
                each k-point if on a k-grid).)

            dm_lo_R0 : The one-electron density matrix at the reference cell 
                R=0 if on a k-grid.

            nelec_lo : The total number of electrons in the local orbitals.

        """
        self.ao2lo = get_ao2lo(mf=self.mf, reference_basis=self.reference_basis)
        self.ao2lo_R0 = get_mat_R0(mf=self.mf, mat=self.ao2lo)

        # FIXME checks for imaginary density components 
        self.dm_lo = get_dm_rotated(mf=self.mf, ao2b=self.ao2lo)
        self.dm_lo_R0 = get_mat_R0(mf=self.mf, mat=self.dm_lo)

        self.nelec_lo = np.trace(self.dm_lo_R0).real
        nelec_diff = self.nelec_lo - np.sum(self.nelec)
        if np.abs(nelec_diff) > 1e-12:
            raise ValueError(f"Total electrons in local orbitals differs from molecular orbitals by {nelec_diff} !")

    # FIXME Do I need the projectors for dft_tools to be orthogonal in the sense that P+ P = 1? In the local
    # basis they satisfy P+ P = 1 (they are just the identity). But if I am in a different basis, say, the molecular
    # orbital basis then mo_coeff+ s1 mo_coeff = 1 (they are not stored in their orthgonal basis). I can rotate into
    # C' = s1^(1/2) mo_coeff which will be an orthogonal basis. But then I just have to be careful about what
    # I feed back into pyscf for self-consistency.
    def make_corr_proj(self):
        """
        Projectors onto the correlated orbital subspace and its complement.

        Sets:
            ao2corr : The projector from the atomic orbital basis onto the 
                correlated orbitals (at each k-point if on a grid).

            ao2corr_com : The complement of ao2corr.

            ao2corr_R0 : ao2corr but at the reference cell R=0 if on a 
                k-grid.

            lo2corr : The projector from the local orbital basis onto the 
                correlated orbitals (at each k-point if on a grid). These 
                are just identity matrices of the selected orbitals.

            lo2corr_com : The complement of lo2corr.

            lo2corr_R0 : lo2corr but at the reference cell R=0 if on a 
                k-grid.

            dm_corr : The density matrix in the correlated subspace (at each 
                k-point if on a grid).

            dm_corr_R0 : As above but at the reference cell R=0 if on a 
                k-grid.

            nelec_corr : The total number of electrons in the correlated 
                subspace.
        """
        if self.ao2lo is None:
            self.make_local_rotation()

        self.ao2corr, self.ao2corr_com = get_ao2corr(mf=self.mf, corr_orbs_labels=self.corr_orbs_labels, ao2lo=self.ao2lo)
        self.ao2corr_R0 = get_mat_R0(mf=self.mf, mat=self.ao2corr)

        self.dm_corr = get_dm_rotated(mf=self.mf, ao2b=self.ao2corr)
        self.dm_corr_R0 = get_mat_R0(mf=self.mf, mat=self.dm_corr)
        self.nelec_corr = np.trace(self.dm_corr_R0).real

        self.lo2corr, self.lo2corr_com = get_lo2corr(mf=self.mf, corr_orbs_labels=self.corr_orbs_labels)
        self.lo2corr_R0 = get_mat_R0(mf=self.mf, mat=self.lo2corr) # This will just be the identity
    
    # TODO
    # the eri algorithm
    # If I just do this onto the correlated space it will probably be faster
    # but how long does this really take?
    def make_eri_local(self):
        '''
        Two-electron repulsion integral in the local basis.
        '''
        # Here I have to rotate the ERI into the local basis
        #ao2eo = lib.einsum('...ij,...jl->...il', ao2lo, lo2eo)
        
        # eri in embedding orbital basis (impurity + bath + core/virtual)
        #eri_eo = mf.mol.ao2mo(ao2corr, intor='int2e', compact=False)

        # eri in embedding orbital basis (impurity + bath)
        #eri_corr = mf.mol.ao2mo(ao2corr, intor='int2e', compact=False)

        #from pyscf.pbc import df
        #eri = df.DF(mf.mol).get_eri()

        # https://pyscf.org/develop/ao2mo_developer.html
        return None

    def make_fock(self):
        '''
        The Fock operator (1e integrals + Hartree-Fock contribution) in the 
        local basis. This will be the non-interacting Hamiltonian for 
        correlated methods.

        Sets:
            fock : Fock matrix (at each k-point if on a grid).

            fock_R0 : Fock matrix in reference cell R0. The non-interacting 
                local Hamiltonian which includes the Hartree-Fock component.
        '''
        if self.ao2lo is None:
            self.make_local_rotation()

        fock_ao = self.mf.get_fock()
        self.fock = get_1e_operator_rotated(mf=self.mf, ao2b=self.ao2lo, mat_1e=fock_ao)
        self.fock_R0 = get_mat_R0(mf=self.mf, mat=self.fock)

    # TODO
    # Needs the ERI algorithm
    def make_hatree_fock_eri(self):
        '''
        The Hartree-Fock contribution of the two-electron integral to the Fock 
        operator.
        # Note that this will have to be updated at each dmft step so 
        '''
        # The subtrat off ERI contribution has to be a separate part becuase it only is
        # subtracted off the impurity. So for dft-tools I could store like a 'double counting' matrix.)
        # I'm going to be storing the ERI anyway. Or, I could calculate it at the mf level and keep
        # the double counting fixed, and only update at each charge update. But I think this will be
        # very wrong for single-shot!
        return None

    def check_fock_density(self):
        '''
        Check that the Fock matrix in the local orbital basis gives the 
        correct electron density in the reference cell R=0.
        '''
        if self.fock is None:
            self.make_fock()
        
        # FIXME Check that smearing and sigma are in mf

        # If mf uses smearing then mf.get_fermi() does not return
        # the correct mu. So calculate it ourselves.
        if self.mf.smearing_method == 'fermi':
            smearing_fnc = fermi_smearing
        elif mf.smearing_method == 'gaussian':
            smearing_fnc = gaussian_smearing
        else:
            raise ValueError('Smearing must be fermi or gaussian !')
        sigma = self.mf.sigma
        nelec = self.mf.mol.tot_electrons(self.mf.kpts.nkpts)
        nocc = nelec // 2
        
        eks, vks = np.linalg.eigh(self.fock)
        mu0 = np.sort(eks.flatten())[nocc-1]
        self.mu = get_mu(eks=eks, nocc=nocc, smearing_fnc=smearing_fnc, sigma=sigma, mu0=mu0)

        # *2 because RHF
        dm = 2.0 * pyscf.lib.einsum('kij,kj,kjl->il', vks, smearing_fnc(eks=eks, sigma=sigma, mu=self.mu), np.transpose(vks, (0,2,1)))
        #dm = self.lo2corr_R0.conj().T @ dm @ self.lo2corr_R0 / eks.shape[0]
        dm /= eks.shape[0]

        if np.any( np.abs(dm - lo.dm_lo_R0).real > 1e-6):
            print("WARNING: Elements in local orbital basis density does not match the the density calculated from the Fock matrix within 1e-6 !")

        return dm
    
    def print_out(self):
        print(f"Total number of k-points: {len(mf.kpts)}")
        print(f"Impurity density: {self.nelec_corr}")
        print(f"Total density in local space: {self.nelec_lo}")

        fock_R0_corr = self.lo2corr_R0.conj().T @ self.fock_R0 @ self.lo2corr_R0
        print(f"Local Hamiltonian:")
        print(fock_R0_corr)

    def save_orbs(self, for_iboview=True):
        save_orbs_molden(mol=self.mf.mol, ao2b=self.ao2mo_R0, filename=self.filename+'_molecular_orbitals.molden', for_iboview=for_iboview)
        save_orbs_molden(mol=self.mf.mol, ao2b=self.ao2lo_R0, filename=self.filename+'_local_orbitals.molden', occ=np.diagonal(self.dm_lo_R0), for_iboview=for_iboview)
        save_orbs_molden(mol=self.mf.mol, ao2b=self.ao2corr_R0, filename=self.filename+'_correlated_orbitals.molden', occ=np.diagonal(self.dm_corr_R0), for_iboview=for_iboview)

    def save(self):
        return None

class PyscfConverter(object):
    """
    Conversion from pyscf output to an hdf5 file that can be used as input for the SumkDFT class.
    """
    def __init__(self, filename, hdf_filename=None,
                       dft_subgrp = 'dft_input', misc_subgrp = 'dft_misc_input',
                       repacking = False):
        """
        Args:
            filename : Base name of DFT files.

            hdf_filename : Name of hdf5 file to be created.

            dft_subgrp : Name of subgroup storing necessary DFT data.

            misc_subgrp : Name of subgroup storing miscellaneous DFT data.
        
            repacking : Does the hdf5 archive need to be repacked to save space?
        """
        assert isinstance(filename, str), "Please provide the DFT files' base name as a string."
        if hdf_filename is None: hdf_filename = filename+'.h5'
        self.filename = filename
        self.hdf_file = hdf_filename
        self.dft_subgrp = dft_subgrp
        self.misc_subgrp = misc_subgrp


if __name__ == "__main__":
    nio = False
    #nio = True
    if nio:
        # Make in ase
        atoms = bulk(name='NiO', crystalstructure='rocksalt', a=4.17)
        # Make in pyscf
        cell = gto.Cell()
        cell.from_ase(atoms)
        cell.basis = 'gth-dzvp-molopt-sr',
        cell.pseudo = 'gth-pade'
        cell.verbose = 5
        cell.exp_to_discard=0.1 # maybe remove this?
        #cell.output = './log_dmet_test.txt'
        cell.build()
        reference_basis = 'gth-szv-molopt-sr' # Used in garnet chan paper for iao
        
        corr_orbs_labels = ["Ni 3d"]
    else:
        # Make in pyscf
        atoms = bulk(name='Si', crystalstructure='fcc', a=5.43053)
        cell = gto.Cell()
        cell.from_ase(atoms)
        cell.basis = 'gth-dzvp',
        cell.pseudo = 'gth-pade'
        cell.verbose = 5
        cell.exp_to_discard=0.1 # maybe remove this? # nah I think I need it to stop libcint having errors
        #cell.output = './log_dmet_test.txt'
        cell.build()
        reference_basis = 'gth-szv'
        
        corr_orbs_labels = ["Si 3p"]

    nkx = 2
    kmesh = [nkx,nkx,nkx]
    #symm = False
    symm = True
    if symm:
        kpts = cell.make_kpts(kmesh, 
                              space_group_symmetry=True, 
                              time_reversal_symmetry=True,
                              symmorphic=True)
    else:
        kpts = cell.make_kpts(kmesh)

    mf = dft.KRKS(cell, kpts=kpts, exxdiv='ewald').density_fit() # restricted spin S = 0
    #mf = dft.KUKS(cell, kpts=kpts, exxdiv='ewald').density_fit()
    #mf.with_df.auxbasis = "weigend" # same as #def2-universal-jfit" recommended for dft
    mf.with_df.auxbasis = df.aug_etb(cell, beta=2.2) # used in dmet paper (even-tempered basis, beta = 2.2 for NiO)
    mf = scf.addons.smearing_(mf, sigma=0.01, method='fermi')
    mf.kernel()


    lo = LocalOrbitals(mf=mf, reference_basis=reference_basis, corr_orbs_labels=corr_orbs_labels)
    lo.kernel()