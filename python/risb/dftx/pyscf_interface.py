import pyscf
from pyscf.pbc import gto, scf, dft, df
from pyscf import lo
from pyscf import lib
from pyscf.lo.orth import vec_lowdin
from pyscf.pbc.tools import pbc as pbctools
import types
import numpy as np
numpy = np
import scipy

from copy import deepcopy

from ase.build import bulk

# TODO
# Move some of the explaining physics comments to a general comment section at start

# FIXME none of the transform_mo and transform_dm work in any basis other than
# the atomic orbital basis (I guess that makes sense). So ao2iao, ao2pao, ao2lo has
# to be on the full grid. Then when calculating density matrices etc have to transform
# the ao dm etc and then transform. For the interface will need to store everything on the 
# full BZ. But that's fine because as long as the DFT part can be done with symmetries
# the rest of it being on the full BZ is not a huge deal.

def get_pmol(mol, reference_basis=None):
    '''
    TODO: docstring here
    '''
    pmol = mol.copy()
    if reference_basis is not None:
        old_verbose = deepcopy(pmol.verbose)
        pmol.verbose = 0
        pmol.build(dump_input=False, parse_arg=False, basis=reference_basis)
        pmol.verbose = old_verbose
    # else pmol.basis but pmol is already in this basis
    return pmol

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
    pmol = get_pmol(mol=mol, reference_basis=reference_basis)

    # This could be replaced with pmol.search_ao_label(labels) 
    corr_orbs = {idx: s for idx,s in enumerate(pmol.ao_labels()) if any(xs in s for xs in corr_orbs_labels)}
    corr_orbs_idx = list(corr_orbs.keys())
    return corr_orbs_idx, corr_orbs

# Adapted from https://github.com/SebWouters/QC-DMET iao_helper.py
def get_porbs(mol, reference_basis=None):
    '''
    Gives the indices of the the orbitals of reference_basis in the original 
    larger original atomic basis of mol.

    Args:
        mol : A cell or molecule object.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as mol.
    
    Returns:
        porbs_idx : The indicies of reference_basis in the original atomic 
            orbital basis.

        porbs_com_idx : The compliment indices of reference_basis in the 
            original atomic orbital basis.
    '''
    #pmol = get_pmol(mol=mol, reference_basis=reference_basis)
    #p_list = [1 if any(xs in s for xs in pmol.ao_labels()) else 0 for s in mol.ao_labels()]
    #assert(np.sum(p_list) == pmol.nao_nr())
    
    porbs_idx, _ = get_corr_orbs(mol=mol, corr_orbs_labels=mol.ao_labels(), reference_basis=reference_basis)
    porbs_com_idx = [i for i in range(len(mol.ao_labels())) if i not in porbs_idx]
    assert ( len(porbs_idx) + len(porbs_com_idx)) == len(mol.ao_labels() )
    
    # Check that pmol is actually a subset of mol
    pmol = get_pmol(mol=mol, reference_basis=reference_basis)
    if len(porbs_idx) != pmol.nao_nr():
        raise ValueError(f"reference_basis {pmol.basis} must be a subset of {mol.basis}")
    return porbs_idx, porbs_com_idx
    
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
    def knizia(C, s1):
        return C @ scipy.linalg.fractional_matrix_power(C.conj().T @ s1 @ C, -0.5)
    
    if otype == 'lowdin':
        orth = vec_lowdin
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
    has_pbc = getattr(mf.mol, 'dimension', 0) > 0

    s1 = mf.get_ovlp()
    if has_pbc:
        has_symm = isinstance(mf.kpts, pyscf.pbc.symm.symmetry.Symmetry)
        if has_symm:
            s1 = np.asarray(mf.kpts.transform_1e_operator(s1))
            kpts = mf.kpts.kpts
        else:
            kpts = mf.kpts
        nkpts = len(kpts)

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
    #porbs_idx, porbs_com_idx = get_porbs(mol=mf.mol, reference_basis=reference_basis)
    #s13 = s1[..., :, porbs_com_idx]
    #s3 = s13[..., porbs_com_idx, :]
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
    has_pbc = getattr(mf.mol, 'dimension', 0) > 0
    
    # C matrix in Szabo and Ostlund, eq 3.133
    # the row index is the atomic orbital index, and 
    # the col is the molecular orbital subscript
    # It is the transformation from the molecular orbitals to the basis functions (atomic orbitals)
    mo2ao = mf.mo_coeff

    # The occupation of each molecular orbital
    mo_occ = mf.mo_occ

    # Get the 1 elec overlap integrals in ao basis
    # S matrix in Szabo and Ostlund, eq. 3.136
    # (copied from lo.iao.iao code)
    # Surely this is the same as mf.get_ovlp() ?
    # I'm just concerned about it not being in the same basis as the int1e_ovlp
    # Because int1e_ovlp_spherical, for example, is different
    #s1 = np.asarray(mf.mol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts.kpts_ibz))
    s1 = mf.get_ovlp()
    if has_pbc:
        has_symm = isinstance(mf.kpts, pyscf.pbc.symm.symmetry.Symmetry)
        if has_symm:
            s1 = np.asarray(mf.kpts.transform_1e_operator(s1))
            mo2ao = np.asarray(mf.kpts.transform_mo_coeff(mo2ao))
            mo_occ = np.asarray(mf.kpts.transform_mo_occ(mo_occ))
            kpts = mf.kpts.kpts
        else:
            kpts = mf.kpts
        nkpts = len(kpts)

    # Get only the occupied molecular orbitals basis transformation from molecular to atomic
    # P = 2 * mo2ao_occ * mo2ao_occ.T Szabo and Ostlund, eq 3.145
    # P essentially gives the restricted density matrix in the atomic orbital basis
    if has_pbc:
        mo2ao_occ = []
        for k in range(nkpts):
            mo2ao_occ.append( mo2ao[k][:,mo_occ[k]>0.01] )
    else:
        mo2ao_occ = mo2ao[:,mo_occ[k]>0.01]

    # Get the intrinsic atomic orbitals (IAO)
    otype = 'lowdin'
    #otype = 'knizia'
    ao2iao = lo.iao.iao(mf.mol, mo2ao_occ, minao=reference_basis, kpts=kpts)
    ao2iao = orthogonalize(C=ao2iao, s1=s1, has_pbc=has_pbc, otype=otype)

    num_ao = mf.mol.nao_nr()
    num_lo = ao2iao.shape[-1]
    
    # Get the compliment, projected atomic orbitals
    if num_ao != num_lo:
        ao2pao = get_ao2pao(mf=mf, ao2lo=ao2iao, reference_basis=reference_basis)
        ao2pao = orthogonalize(C=ao2pao, s1=s1, has_pbc=has_pbc, otype=otype)

        # Contrusct the localized orbitals
        porbs_idx, porbs_com_idx = get_porbs(mol=mf.mol, reference_basis=reference_basis)
        if (ao2iao.dtype == np.complex_) or (ao2pao.dtype == np.complex_) or (s1.dtype == np.complex_):
            lo_dtype = np.complex_
        else:
            lo_dtype = np.float_
        ao2lo = np.empty(shape=s1.shape, dtype=lo_dtype)
        ao2lo[...,:,porbs_idx] = ao2iao
        ao2lo[...,:,porbs_com_idx] = ao2pao
    
        # FIXME reorder here, I don't think we have to because the valence orbitals are 
        # in the order of reference_basis, while the pao are not in any order. But I don't
        # think we really care about them?
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

def get_dm_lo(mf, ao2lo):
    '''
    Get the density matrix rotated into the basis of ao2lo.

    Args:
        mf : A mean-field object from HF or DFT.

        ao2lo : The transformation matrix from the atomic orbital basis to the 
            local basis. If it is on a lattice it must be on the full BZ.

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is density matrix 
        in the local orbital basis
    '''
    has_pbc = getattr(mf.mol, 'dimension', 0) > 0
    
    s1 = mf.get_ovlp()
    
    # The density matrix in the non-orthogonal atomic orbital basis
    # P = 2 * mo2ao_occ * mo2ao_occ.T Szabo and Ostlund, eq 3.145
    dm_ao = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    
    if has_pbc:
        has_symm = isinstance(mf.kpts, pyscf.pbc.symm.symmetry.Symmetry)
        if has_symm:
            s1 = np.asarray(mf.kpts.transform_1e_operator(s1))
            dm_ao = np.asarray(mf.kpts.transform_dm(dm_ao))

    # einsum is very slow for large k and # of orbs
    #dm_lo = lib.einsum('...ij,...jk,...kl,...lm,...mn->...in', np.transpose(ao2lo.conj(), (0,2,1)), s1, dm_ao, s1, ao2lo)
    dm_lo = np.zeros(shape=dm_ao.shape, dtype=dm_ao.dtype)
    if has_pbc:
        for k in range(dm_lo.shape[0]):
            dm_lo[k] = ao2lo[k].conj().T @ s1[k] @ dm_ao[k] @ s1[k] @ ao2lo[k]
    else:
        dm_lo = ao2lo.conj().T @ s1 @ dm_ao @ s1 @ ao2lo

    return dm_lo

def get_dm_loc(mf, dm):
    '''
    Get the local density matrix at R=0 (Fourier phase factor exp(iRk) = 1).

    Args:
        mf : A mean-field object from HF or DFT. It must be on the full BZ.

        dm : A density matrix at each k-point on the k-grid.

    Returns:
        An array that is the local density matrix.
    '''
    has_pbc = getattr(mf.mol, 'dimension', 0) > 0
    assert (has_pbc), f"dm should be 3-dim (lattice), got: {dm.shape}"
    
    has_symm = isinstance(mf.kpts, pyscf.pbc.symm.symmetry.Symmetry)
    if has_symm:
        #dm_loc = mf.kpts.dm_at_ref_cell(dm_ibz=dm)
        if len(dm) != len(mf.kpts.kpts):
            raise ValueError(f"dm must be on the full BZ with {len(mf.kpts.kpts)} points, but got {len(shape)} !")
    
    dm_loc = lib.einsum('kij->ij', dm) / len(dm)

    return dm_loc

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

#nio = False
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

nkx = 4
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

ao2lo = get_ao2lo(mf=mf, reference_basis=reference_basis)

dm_lo = get_dm_lo(mf=mf, ao2lo=ao2lo)
dm_lo_loc = get_dm_loc(mf=mf, dm=dm_lo)
dm_lo_nelec = np.trace(dm_lo_loc)

corr_orb_labels = ["Ni 3d"]
corr_orbs_idx, corr_orbs = get_corr_orbs(mol=mf.mol, corr_orbs_labels=corr_orb_labels, reference_basis=reference_basis)

nkpts = len(mf.kpts.kpts)

impurityOrbs = np.zeros( [ ao2lo.shape[-1] ], dtype=int )
impurityOrbs[corr_orbs_idx] = 1
impurityOrbs_mat = np.matrix( impurityOrbs )
if (impurityOrbs_mat.shape[0] > 1):
    impurityOrbs_mat = impurityOrbs_mat.T
isImpurity = np.dot( impurityOrbs_mat.T, impurityOrbs_mat) == 1
numImpOrbs = np.sum(impurityOrbs)
impurity1RDM = np.reshape( dm_lo[ ...,isImpurity ], (nkpts, numImpOrbs, numImpOrbs) )
impurity1RDM_loc = np.reshape( dm_lo_loc[ isImpurity ], (numImpOrbs, numImpOrbs) )

embeddingOrbs = 1 - impurityOrbs
embeddingOrbs = np.matrix( embeddingOrbs )
if (embeddingOrbs.shape[0] > 1):
    embeddingOrbs = embeddingOrbs.T
isEmbedding = np.dot( embeddingOrbs.T, embeddingOrbs) == 1
numEmbedOrbs = np.sum( embeddingOrbs )
embedding1RDM = np.reshape( dm_lo[ ...,isEmbedding ], (nkpts, numEmbedOrbs, numEmbedOrbs) )
embedding1RDM_loc = np.reshape( dm_lo_loc[ isEmbedding ], (numEmbedOrbs, numEmbedOrbs) )

numTotalOrbs = len( impurityOrbs )

threshold = 1e-13
eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort(axis=-1) # occupation numbers closest to 1 come first
eigenvals = np.take_along_axis(eigenvals, idx, axis=-1)
for k in range(eigenvecs.shape[0]):
    eigenvecs[k] = eigenvecs[k][:,idx[k]]
# Number of bath orbitals to keep at each k-point (should take the max of this so each k-point has the same size)
# but in principle we don't actually need that
tokeep = np.sum( -np.maximum( -eigenvals, eigenvals - 2.0 ) > threshold, axis=-1)

numBathOrbs = numImpOrbs
numBathOrbs = min(np.max(tokeep), numBathOrbs)

pureEnvironEigVals = -eigenvals[...,numBathOrbs:]
pureEnvironEigVecs = eigenvecs[...,:,numBathOrbs:]
idx = pureEnvironEigVals.argsort(axis=-1)
for k in range(eigenvecs.shape[0]):
    eigenvecs[k,:,numBathOrbs:] = pureEnvironEigVecs[k][:,idx[k]]
pureEnvironEigVals = np.take_along_axis(-pureEnvironEigVals, idx, axis=-1)
coreOccupations = np.empty([pureEnvironEigVals.shape[0], pureEnvironEigVals.shape[1]+numImpOrbs+numBathOrbs])
for k in range(pureEnvironEigVals.shape[0]):
    coreOccupations[k] = np.hstack(( np.zeros([ numImpOrbs + numBathOrbs ]), pureEnvironEigVals[k] ))

# eigenvecs indexed as k,row,col
for counter in range(0, numImpOrbs):
    eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=2) #Stack columns with zeros in the beginning
counter = 0
for counter2 in range(0, numTotalOrbs):
    if ( impurityOrbs[counter2] ):
        eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=1) #Stack rows with zeros on locations of the impurity orbitals
        eigenvecs[:, counter2, counter] = 1.0
        counter += 1
assert( counter == numImpOrbs )

# Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
assert( np.linalg.norm( lib.einsum('kij,kjl->kil', np.transpose(eigenvecs.conj(), (0,2,1)), eigenvecs) - np.eye(numTotalOrbs) ) < 1e-12 )

# eigenvecs[ : , 0:numImpOrbs ]                      = impurity orbitals
# eigenvecs[ : , numImpOrbs:numImpOrbs+numBathOrbs ] = bath orbitals
# eigenvecs[ : , numImpOrbs+numBathOrbs: ]           = pure environment orbitals in decreasing order of occupation number

# eigenvecs is the operator that goes from local orbitals to DMET orbital basis
# coreOccupations is the reduced density matrix of of the core orbitals in the DMET basis

core_cutoff = 0.01
#core_cutoff = 0.5
coreOccupations[ np.where(coreOccupations < core_cutoff) ] = 0.0
coreOccupations[ np.where(coreOccupations > (2.0 - core_cutoff)) ] = 2.0

for cnt in range(len(coreOccupations)):
    if ( coreOccupations[ cnt ] < core_cutoff ):
        coreOccupations[ cnt ] = 0.0
    elif ( coreOccupations[ cnt ] > 2.0 - core_cutoff ):
        coreOccupations[ cnt ] = 2.0
    else:
        print("Bad DMET bath orbital selection: trying to put a bath orbital with occupation", coreOccupations[ cnt ], "into the environment :-(.")

lo2eo = eigenvecs
Norb_in_imp  = numImpOrbs + numBathOrbs

ao2eo = lib.einsum('...ij,...jl->...il', ao2lo, lo2eo)
core1RDM_eo = lib.einsum('...ij,...j,...jl->...il', ao2eo, coreOccupations, np.transpose(ao2eo.conj(), (0,2,1)))

# eri in embedding orbital basis (impurity + bath + core/virtual)
#eri_eo = mf.mol.ao2mo(ao2eo, intor='int2e', compact=False)

# eri in embedding orbital basis (impurity + bath)
eri_eo = mf.mol.ao2mo(ao2eo[:,:Norb_in_imp], intor='int2e', compact=False)

from pyscf.pbc import df
eri = df.DF(mf.mol).get_eri()