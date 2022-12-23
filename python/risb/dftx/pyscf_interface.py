import pyscf
from pyscf.pbc import gto, scf, dft, df
from pyscf import lo
from pyscf import lib
from pyscf.pbc.tools import pbc as pbctools
import numpy as np
import scipy

from ase.build import bulk

def get_pcell(cell, reference_basis=None):
    pcell = cell.copy()
    if reference_basis is not None:
        pcell.build(dump_input=False, parse_arg=False, basis=reference_basis)
    # else pcell.basis but pcell is already in this basis
    return pcell

def get_corr_orbs(cell, corr_orbs_labels=["Ni 3d"], reference_basis=None):
    '''
    Get the indices in the reference_basis of the correlated orbitals.
    If corr_orbs_labels are atomic orbital labels in a larger space than
    reference_basis, then this returns the indices of the reference basis
    in the larger basis.

    Args:
        cell : A cell or molecule object.
        
        corr_orbs_labels : A list of orbitals. Partial matches are OK, such
            that "Ni" will be all Ni orbitals, and "Ni 3d" will be all
            3d orbitals on Ni.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as cell.
    
    Returns:
        corr_orbs_idx : A list of the orbital indices in the reference basis.
        
        corr_orbs : A dictionary where the key is the index in the reference
            basis and the value is the orbital label.
    '''    
    pcell = get_pcell(cell=cell, reference_basis=reference_basis)

    # This could be replaced with pcell.search_ao_label(labels) 
    corr_orbs = {idx: s for idx,s in enumerate(pcell.ao_labels()) if any(xs in s for xs in corr_orbs_labels)}
    corr_orbs_idx = list(corr_orbs.keys())
    return corr_orbs_idx, corr_orbs

# Adapted from https://github.com/SebWouters/QC-DMET iao_helper.py
def construct_p_list(cell, reference_basis=None):
    #pcell = get_pcell(cell=cell, reference_basis=reference_basis)
    #p_list = [1 if any(xs in s for xs in pcell.ao_labels()) else 0 for s in cell.ao_labels()]
    #assert(np.sum(p_list) == pcell.nao_nr())
    
    p_orbs_idx, _ = get_corr_orbs(cell, corr_orbs_labels=cell.ao_labels(), reference_basis=reference_basis)
    p_orbs_idx_com = [i for i in range(len(cell.ao_labels())) if i not in p_orbs_idx]
    assert ( len(p_orbs_idx) + len(p_orbs_idx_com)) == len(cell.ao_labels() )
    return p_orbs_idx, p_orbs_idx_com
    
# Adapted from https://github.com/SebWouters/QC-DMET iao_helper.py
def get_ao2po(mf, ao2lo, reference_basis=None):
    
    if reference_basis is None:
        reference_basis = mf.cell.basis
    
    nkpts = len(mf.kpts)

    # hack to get access to make_iaos(s1, s2, s12, mo) defined inside iao
    # using the codebyte and making it a function
    make_iaos = lo.iao.iao.__code__.co_consts[10]
    
    # Copy pasted from iao line 69
    pmoll = get_pcelc(cell=mf.cell, reference_basis=reference_basis)
    has_pbc = getattr(mol, 'dimension', 0) > 0
    if has_pbc:
        from pyscf.pbc import gto as pbcgto
        s1 = mol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        s2 = pmol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        s12 = pbcgto.cell.intor_cross('int1e_ovlp', mol, pmol, kpts=kpts)
    else:
        #s1 is the one electron overlap integrals (coulomb integrals)
        s1 = mol.intor_symmetric('int1e_ovlp')
        #s2 is the same as s1 except in minao
        s2 = pmol.intor_symmetric('int1e_ovlp')
        #overlap integrals of the two molecules
        s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)
    
    # Get the compliment of iao
    dm_lo = lib.einsum('kij,kjl->kil', ao2lo, ao2lo.conj().transpose((0,2,1)) )
    mx = lib.einsum('kij,kjl,klm->kim', s1, dm_lo, s1)
    ao2lo_com = np.empty(shape=(nkpts, num_ao, num_ao-num_lo), dtype=ao2lo.dtype)
    for k in range(nkpts):
    eigs, vecs = scipy.linalg.eigh(a=mx[k],b=s1e_ovlp[k])
    ao2lo_com[k,...] = vecs[...,:num_ao - num_lo]

# Adapted from https://github.com/SebWouters/QC-DMET iao_helper.py
def get_ao2lo(mf, reference_basis=None):
    '''
    The transformation matrix at each k-point from the atomic orbital basis 
    to the local orbital basis.

    Args:
        mf : A mean-field object from HF or DFT.

        reference_basis : (default None) The reference basis in the local
            orbital space. If None, it is the same reference basis as cell.

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is the unitary transformation 
        into the local orbital basis.
    '''

    # FIXME check cases for cell or mol, currently only does a lattice

    if reference_basis is None:
        reference_basis = mf.cell.basis
    
    nkpts = len(mf.kpts)

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
    #s1 = np.asarray(mf.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts.kpts_ibz))
    s1 = mf.get_ovlp()

    # Get only the occupied molecular orbitals basis transformation from molecular to atomic
    # P = 2 * mo2ao_occ * mo2ao_occ.T Szabo and Ostlund, eq 3.145
    # P essentially gives the restricted density matrix in the atomic orbital basis
    mo2ao_occ = []
    for k in range(nkpts):
	    mo2ao_occ.append( mo2ao[k][:,mo_occ[k]>0] )

    # Get the intrinsic atomic orbitals (IAO)
    if isinstance(mf.kpts, pyscf.pbc.symm.symmetry.Symmetry):
        ao2lo = lo.iao.iao(mf.cell, mo2ao_occ, minao=reference_basis, kpts=mf.kpts.kpts_ibz)
    else:
        ao2lo = lo.iao.iao(mf.cell, mo2ao_occ, minao=reference_basis, kpts=mf.kpts)

    # Orthogonalize IAO
    for k in range(nkpts):
	    ao2lo[k] = lo.vec_lowdin(ao2lo[k], s1[k])

    num_ao = mf.mol.nao_nr()
    num_lo = ao2lo.shape[-1]
    print(f"num_ao = {num_ao}, num_lo = {num_lo}")
    
    # Get the compliment, projected atomic orbitals
    if num_ao != num_lo:
        get_ao2po(mf=mf, ao2lo=ao2lo)

    # Quick check
    should_be_1 = lib.einsum('kij,kjl,klm->kim', ao2lo.conj().transpose((0,2,1)), s1, ao2lo)
    for k in range(nkpts):
	    print(f"norm(I - ao2lo_full.T * S * ao2lo_full) for k->{k} =", \
                np.linalg.norm(should_be_1[k] - np.eye( should_be_1[k].shape[0])))

    return ao2lo, ao2lo_com

def get_dm_lo(mf, ao2lo):
    '''
    Get the density matrix rotated into the basis of ao2lo.

    Args:
        mf : A mean-field object from HF or DFT.

        ao2lo : The transformation matrix from the atomic orbital basis to the 
            local basis.

    Returns:
        A 2-dim (molecule) or 3-dim (lattice) array that is density matrix 
        in the local orbital basis
    '''
    s1 = mf.get_ovlp()
    
    # The density matrix in the non-orthogonal atomic orbital basis
    # P = 2 * mo2ao_occ * mo2ao_occ.T Szabo and Ostlund, eq 3.145
    dm_ao = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)

    # einsum is very slow for large k and # of orbs
    #dm_lo = lib.einsum('...ij,...jk,...kl,...lm,...mn->...in', np.transpose(ao2lo.conj(), (0,2,1)), s1, dm_ao, s1, ao2lo)
    dm_lo = np.zeros(shape=dm_ao.shape, dtype=dm_ao.dtype)
    if len(dm_lo.shape) == 3:
        for k in range(dm_lo.shape[0]):
            dm_lo[k] = ao2lo[k].conj().T @ s1[k] @ dm_ao[k] @ s1[k] @ ao2lo[k]
    
    elif len(dm_lo.shape) == 2:
        dm_lo = ao2lo.conj().T @ s1 @ dm_ao @ s1 @ ao2lo

    else:
        raise ValueError("The density matrix must be a 2-dim (molecule) or 3-dim (lattice) array !")

    return dm_lo

def get_dm_loc(mf, dm):
    '''
    Get the local density matrix at R=0 (Fourier phase factor exp(iRk) = 1).

    Args:
        mf : A mean-field object from HF or DFT. It is fine if the k-grid is 
            in the IBZ.

        dm : A density matrix at each k-point on the k-grid.

    Returns:
        An array that is the local density matrix.
    '''
    assert (dm.shape == 3), f"dm should be 3-dim (lattice), got: {dm.shape}"
    
    if isinstance(mf.kpts, pyscf.pbc.symm.symmetry.Symmetry):
        #dm_loc = lib.einsum('kij,k->ij', dm, mf.kpts.weights_ibz)
        dm_loc = mf.kpts.dm_at_ref_cell(dm_ibz=dm)
    else:
        dm_loc = lib.einsum('kij->ij', dm) / dm.shape[0]

    return dm_loc

def get_supercell(cell, kmesh): # Get supercell and phase?
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

## Make in ase
#atoms = bulk(name='NiO', crystalstructure='rocksalt', a=4.17)
#
## Make in pyscf
#cell = gto.Cell()
#cell.from_ase(atoms)
#cell.basis = 'gth-dzvp-molopt-sr',
#cell.pseudo = 'gth-pade'
#cell.verbose = 5
#cell.exp_to_discard=0.1 # maybe remove this? # nah I think I need it to stop libcint having errors
##cell.output = './log_dmet_test.txt'
#cell.build()

atoms = bulk(name='Si', crystalstructure='fcc', a=5.43053)

# Make in pyscf
cell = gto.Cell()
cell.from_ase(atoms)
cell.basis = 'gth-dzvp',
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.exp_to_discard=0.1 # maybe remove this? # nah I think I need it to stop libcint having errors
#cell.output = './log_dmet_test.txt'
cell.build()

nkx = 4
kmesh = [nkx,nkx,nkx]
kpts = cell.make_kpts(kmesh, 
                      space_group_symmetry=True, 
                      time_reversal_symmetry=True,
                      symmorphic=True)
#kpts = cell.make_kpts(kmesh)

mf = dft.KRKS(cell, kpts=kpts, exxdiv='ewald').density_fit() # restricted spin S = 0
#mf = dft.KUKS(cell, kpts=kpts, exxdiv='ewald').density_fit()
#mf.with_df.auxbasis = "weigend" # same as #def2-universal-jfit" recommended for dft
mf.with_df.auxbasis = df.aug_etb(cell, beta=2.2) # used in dmet paper (even-tempered basis, beta = 2.2 for NiO)
#mf = scf.addons.smearing_(mf, sigma=0.01, method='fermi')
mf.kernel()

#reference_basis = 'gth-szv-molopt-sr' # Used in garnet chan paper for iao
reference_basis = 'gth-szv'
ao2lo, ao2lo_com = get_ao2lo(mf=mf, reference_basis=reference_basis)

dm_lo = get_dm_lo(mf=mf, ao2lo=ao2lo)
dm_lo_loc = get_dm_loc(mf=mf, dm=dm_lo)
dm_lo_nelec = np.trace(dm_lo_loc)

corr_orb_labels = ["Ni 3d"]
corr_orbs_idx, corr_orbs = get_corr_orbs(cell=mf.cell, corr_orbs_labels=corr_orb_labels, reference_basis=reference_basis)

impurityOrbs = np.zeros( [ num_orbs_C ], dtype=int )
impurityOrbs[corr_orbs_idx] = 1
impurityOrbs_mat = np.matrix( impurityOrbs )
if (impurityOrbs_mat.shape[0] > 1):
    impurityOrbs_mat = impurityOrbs_mat.T
isImpurity = np.dot( impurityOrbs_mat.T, impurityOrbs_mat) == 1
numImpOrbs = np.sum(impurityOrbs)
impurity1RDM = np.reshape( dm_iao[ ...,isImpurity ], (nkpts, numImpOrbs, numImpOrbs) )

embeddingOrbs = 1 - impurityOrbs
embeddingOrbs = np.matrix( embeddingOrbs )
if (embeddingOrbs.shape[0] > 1):
    embeddingOrbs = embeddingOrbs.T
isEmbedding = np.dot( embeddingOrbs.T, embeddingOrbs) == 1
numEmbedOrbs = np.sum( embeddingOrbs )
embedding1RDM = np.reshape( dm_iao[ ...,isEmbedding ], (nkpts, numEmbedOrbs, numEmbedOrbs) )

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

ao2lo = C
lo2eo = eigenvecs
Norb_in_imp  = numImpOrbs + numBathOrbs

ao2eo = lib.einsum('...ij,...jl->...il', ao2lo, lo2eo)
core1RDM_eo = lib.einsum('...ij,...j,...jl->...il', ao2eo, coreOccupations, np.transpose(ao2eo.conj(), (0,2,1)))

# eri in embedding orbital basis (impurity + bath + core/virtual)
#eri_eo = mf.mol.ao2mo(ao2eo, intor='int2e', compact=False)

# eri in embedding orbital basis (impurity + bath)
eri_eo = mf.mol.ao2mo(ao2eo[:,:Norb_in_imp], intor='int2e', compact=False)

from pyscf.pbc import df
eri = df.DF(cell).get_eri()