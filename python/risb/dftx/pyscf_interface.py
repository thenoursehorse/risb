import pyscf
from pyscf.pbc import gto, scf, dft, df
from pyscf import lo
from pyscf import lib
from pyscf.pbc.tools import pbc as pbctools
import types
import numpy as np
numpy = np
import scipy

from copy import deepcopy

from ase.build import bulk

# Heavily borrows from https://github.com/SebWouters/QC-DMET/
# and https://github.com/hungpham2017/pDMET, as well as the pyscf code itself.

# s1 : The overlap matrix between the non-orthonormal atomic orbitals. See 
# the discussion below Szabo and Ostlund, eq. 3.136. In pyscf this is obtained
# as mf.get_ovlp() or from mol/cell as mol.intor('int1e_ovlp') or 
# cell.pbc_into('int1e_ovlp', kpts=). It can also be obtained in sph or cubic
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
# loc : A quantity in the reference cell R=0. That is, a k-space quantity F(k)  
# that has been Fourier transformed to R=0 with exp(i R=0 k) = 1, so that 
# F(R=0) 1/N sum_k F(k)

# TODO 
# allow open shell HF etc. Currently only have restricted HF.

# TODO
# Allow ao2lo etc to be on the IBZ. But need to figure out the correct way to 
# transform from the IBZ to the BZ in the lo basis.

# TODO
# If do not use psuedo potential then have to treat the core electrons 
# correctly and freeze them. But for now we just assume that the core 
# electrons are not in the atomic orbital basis

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

def get_corr_orbs(mol, corr_orbs_labels=["Ni 3d"], reference_basis=None):
    '''
    Get the indices in the reference_basis of the correlated orbitals.
    If corr_orbs_labels are atomic orbital labels in a larger space than
    reference_basis, then this returns the indices of the reference basis
    in the larger basis.

    Args:
        mol : A cell or molecule object.
        
        corr_orbs_labels : A list of orbitals. Partial matches are OK, such
            that "Ni" will be all Ni orbitals, and "Ni 3d" will be all
            3d orbitals on Ni.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.
    
    Returns:
        corr_orbs_idx : A list of the orbital indices in the reference basis.
        
        corr_orbs : A dictionary where the key is the index in the reference
            basis and the value is the orbital label.
    '''    
    ref_mol = get_ref_mol(mol=mol, reference_basis=reference_basis)

    # This could be replaced with ref_mol.search_ao_label(labels) 
    corr_orbs = {idx: s for idx,s in enumerate(ref_mol.ao_labels()) if any(xs in s for xs in corr_orbs_labels)}
    corr_orbs_idx = list(corr_orbs.keys())
    return corr_orbs_idx, corr_orbs

# FIXME ref_orbs and corr_orbs should just be the other way around to be honest
# Adapted from https://github.com/SebWouters/QC-DMET iao_helper.py
def get_ref_orbs(mol, reference_basis=None):
    '''
    Gives the indices of the the orbitals of reference_basis in the original 
    larger original atomic basis of mol.

    Args:
        mol : A cell or molecule object.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.
    
    Returns:
        ref_orbs_idx : The indicies of reference_basis in the original atomic 
            orbital basis.

        ref_orbs_com_idx : The compliment indices of reference_basis in the 
            original atomic orbital basis.
    '''
    #ref_mol = get_ref_mol(mol=mol, reference_basis=reference_basis)
    #ref_orbs_idx = [1 if any(xs in s for xs in ref_mol.ao_labels()) else 0 for s in mol.ao_labels()]
    #assert(np.sum(p_list) == ref_mol.nao_nr())
    
    ref_orbs_idx, _ = get_corr_orbs(mol=mol, corr_orbs_labels=mol.ao_labels(), reference_basis=reference_basis)
    ref_orbs_com_idx = [i for i in range(len(mol.ao_labels())) if i not in ref_orbs_idx]
    assert ( len(ref_orbs_idx) + len(ref_orbs_com_idx)) == len(mol.ao_labels() )
    
    # Check that ref_mol is actually a subset of mol
    ref_mol = get_ref_mol(mol=mol, reference_basis=reference_basis)
    if len(ref_orbs_idx) != ref_mol.nao_nr():
        raise ValueError(f"reference_basis {ref_mol.basis} must be a subset of {mol.basis}")
    return ref_orbs_idx, ref_orbs_com_idx
    
def orthogonalize(C, s1, has_pbc, otype='lowdin'):
    '''
    Orthongonalize the basis C.

    Args:
        C : The coefficients that define the basis C, written in the atomic 
            orbital basis.
        
        s1 : The overlap matrix between the atomic orbitals.

        has_pbc : A boolean where True is if C and s1 are on a lattice (are 
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

    nkpts = C.shape[0]
    if has_pbc:
        for k in range(nkpts):
            C[k] = orth(C[k], s1[k])
    else:
        C = orth(C, s1)
    return C

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
    _make_iaos = get_nested_function(lo.iao.iao, 'make_iaos', lindep_threshold=lindep_threshold)
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
        dm_lo = lib.einsum('kij,kjl->kil', ao2lo, ao2lo.conj().transpose((0,2,1)) )
        mx = lib.einsum('kij,kjl,klm->kim', s1, dm_lo, s1)
        ao2pao = np.empty(shape=(nkpts, num_ao, num_ao-num_lo), dtype=ao2lo.dtype)
        for k in range(nkpts):
            # s1 ao2lo ao2lo* s1 = vecs eigs vecs*
            # ao2lo ao2lo* = s1^-1 vecs eigs vecs s1^-1
            # vecs* s1 vecs = 1
            # eigs are 0 or 1, where 0 is a compliment orbital in the ao basis
            eigs, vecs = scipy.linalg.eigh(a=mx[k], b=s1[k])
            # eigs are sorted from lowest to highest
            ao2pao[k,...] = vecs[..., :num_ao-num_lo]
    else:
        dm_lo = ao2lo @ ao2lo.conj().T
        mx = s1 @ dm_lo @ s1
        eigs, vecs = scipy.linalg.eigh(a=mx, b=s1)
        ao2pao = vecs[..., :num_ao - num_lo]
    
    # FIXME It isn't clear to me if you need redo the iao for the complement, 
    # I just do because Seb does it in their code. Surely the projected atomic
    # orbitals are already obtained at this line and all that has to be done is to
    # orthogonalize?
    # Doing below causes orthogonalization issues for NiO at some k-points

    # Get the overlap matrices at the indices of the complement of the reference basis
    #ref_orbs_idx, ref_orbs_com_idx = get_ref_orbs(mol=mf.mol, reference_basis=reference_basis)
    #s13 = s1[..., :, ref_orbs_com_idx]
    #s3 = s13[..., ref_orbs_com_idx, :]
    #
    ## Construct iaos in the compliment
    #if has_pbc:
    #    for k in range(nkpts):
    #        ao2pao[k] = make_iaos(s1=s1[k], s2=s3[k], s12=s13[k], mo=ao2pao[k])
    #else:
    #    ao2pao = make_iaos(s1=s1, s2=s3, s12=s13, mo=ao2pao)

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
    ao2iao = lo.iao.iao(mf.mol, ao2mo_occ, minao=reference_basis, kpts=kpts)
    ao2iao = orthogonalize(C=ao2iao, s1=s1, has_pbc=has_pbc, otype=otype)

    num_ao = mf.mol.nao_nr()
    num_lo = ao2iao.shape[-1]
    
    # Get the compliment, projected atomic orbitals
    if num_ao != num_lo:
        ao2pao = get_ao2pao(mf=mf, ao2lo=ao2iao, reference_basis=reference_basis)
        ao2pao = orthogonalize(C=ao2pao, s1=s1, has_pbc=has_pbc, otype=otype)

        # Contrusct the localized orbitals
        ref_orbs_idx, ref_orbs_com_idx = get_ref_orbs(mol=mf.mol, reference_basis=reference_basis)
        if (ao2iao.dtype == np.complex_) or (ao2pao.dtype == np.complex_) or (s1.dtype == np.complex_):
            lo_dtype = np.complex_
        else:
            lo_dtype = np.float_
        ao2lo = np.empty(shape=s1.shape, dtype=lo_dtype)
        ao2lo[...,:,ref_orbs_idx] = ao2iao
        ao2lo[...,:,ref_orbs_com_idx] = ao2pao
    
        # FIXME reorder here, I don't think we have to because the occupied orbitals are 
        # in the order of reference_basis, while the virtual pao are not in whatever order. 
        # that eigh returns. But I don't think we really care about them?
        #for k in range(nkpts):
        #    ao2lo[k] = resort_orbitals(mol=mf.mol, ao2lo=ao2lo[k], k=k)

        ao2lo = orthogonalize(C=ao2lo, s1=s1, has_pbc=has_pbc, otype=otype)
    else:
        ao2lo = ao2iao
    
    # Check is actually orthogonal
    if has_pbc:
        should_be_1 = lib.einsum('kij,kjl,klm->kim', ao2lo.conj().transpose((0,2,1)), s1, ao2lo)
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

def get_ao2corr(mf, corr_orbs_labels, reference_basis=None, ao2lo=None):
    '''
    The projection matrix at each k-point from the atomic orbital basis 
    to the correlated orbitals.

    Args:
        mf : A mean-field object from HF or DFT. It can be an object that 
            uses symmetry. 
        
        corr_orbs_labels : The

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.


    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is the unitary 
        transformation into the local orbital basis. If on a lattice it 
        returns the transformation on the full BZ.
    '''
    if reference_basis is None:
        reference_basis = mf.mol.basis
    
    if ao2lo is None:
        ao2lo = get_ao2lo(mf=mf, reference_basis=reference_basis)

    corr_orbs_idx, corr_orbs = get_corr_orbs(mol=mf.mol, corr_orbs_labels=corr_orbs_labels, reference_basis=reference_basis)
    return ao2lo[..., corr_orbs_idx]
    
    #lo2corr = np.ones(shape=(ao2lo[0], ao2lo[1], len(corr_orbs_idx)))
    #lo2corr = lo2corr[..., corr_orbs]


def get_dm(mf, ao2b):
    '''
    Get the density matrix rotated into the basis of b.

    Args:
        mf : A mean-field object from HF or DFT.

        ao2b : The transformation matrix from the atomic orbital basis to the 
            basis b. If it is on a lattice it must be on the full BZ. It can 
            be square (project onto some orbitals).

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is density matrix 
        in the local orbital basis
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

def get_mat_loc(mf, mat):
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
    return lib.einsum('kij->ij', mat) / len(mat)

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

# FIXME This is from pdmet, but I don't think we ever need it
def get_supercell(cell, kmesh): # Get supercell and phase
    a = cell.lattice_vectors()
    Ts = lib.cartesian_prod((np.arange(kmesh[0]), np.arange(kmesh[1]), np.arange(kmesh[2])))
    Rs = Ts @ a
    NRs = Rs.shape[0]
    # FIXME this is wrong for when symmetry
    if isinstance(mf.kpts, pyscf.pbc.symm.symmetry.Symmetry):
        #phase = 1/np.sqrt(NRs) * np.exp(1j*Rs.dot(mf.kpts.kpts_ibz.T))
        phase = 1/np.sqrt(NRs) * np.exp(1j*Rs.dot(mf.kpts.kpts.T))
    else:
        phase = 1/np.sqrt(NRs) * np.exp(1j*Rs.dot(mf.kpts.T))
    scell = pbctools.super_cell(cell, kmesh)
    return scell, phase

nio = False
nio = True
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
    molden_filename = 'Ni'
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
    molden_filename = 'Si'

nkx = 7
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

ao2mo = mf.kpts.transform_mo_coeff(mf.mo_coeff)
ao2mo_loc = get_mat_loc(mf=mf, mat=ao2mo)

# Unitary to local orbital basis
ao2lo = get_ao2lo(mf=mf, reference_basis=reference_basis)
ao2lo_loc = get_mat_loc(mf=mf, mat=ao2lo)

# Projector to correlated basis
ao2corr = get_ao2corr(mf=mf, corr_orbs_labels=corr_orbs_labels, reference_basis=reference_basis)
ao2corr_loc = get_mat_loc(mf=mf, mat=ao2corr)

# Density matrix in local orbital basis
dm_lo = get_dm(mf=mf, ao2b=ao2lo)
dm_lo_loc = get_mat_loc(mf=mf, mat=dm_lo)
nelec_lo = np.trace(dm_lo_loc)
nelec_diff = nelec_lo - np.sum(mf.mol.nelec)
print(f"Density within window: {nelec_lo}")
if np.abs(nelec_diff) > 1e-12:
    raise ValueError(f"Total electrons in local orbitals differs from molecular orbitals by {nelec_diff} !")

# Density matrix in correlated basis 
dm_corr = get_dm(mf=mf, ao2b=ao2corr)
dm_corr_loc = get_mat_loc(mf=mf, mat=dm_corr)
nelec_corr = np.trace(dm_corr_loc)
print(f"Impurity density: {nelec_corr}")

save_orbs_molden(mol=mf.mol, ao2b=ao2mo_loc, filename=molden_filename+'_molecular_orbitals.molden')
save_orbs_molden(mol=mf.mol, ao2b=ao2lo_loc, filename=molden_filename+'_local_orbitals.molden', occ=np.diagonal(dm_lo_loc))
save_orbs_molden(mol=mf.mol, ao2b=ao2corr_loc, filename=molden_filename+'_correlated_orbitals.molden', occ=np.diagonal(dm_corr_loc))

# ERI in correlated basis

#ao2eo = lib.einsum('...ij,...jl->...il', ao2lo, lo2eo)

# eri in embedding orbital basis (impurity + bath + core/virtual)
#eri_eo = mf.mol.ao2mo(ao2corr, intor='int2e', compact=False)

# eri in embedding orbital basis (impurity + bath)
#eri_corr = mf.mol.ao2mo(ao2corr, intor='int2e', compact=False)

#from pyscf.pbc import df
#eri = df.DF(mf.mol).get_eri()