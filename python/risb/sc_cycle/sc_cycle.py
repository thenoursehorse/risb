import numpy as np
from itertools import product
from scipy.linalg import sqrtm
from scipy.linalg import pinv
from scipy.special import binom
from triqs.gf import Gf, iOmega_n, inverse, dyson

def block_mat_to_full(A):
    total_size = 0
    for block in A:
        if len(A[block].shape) != 2:
            raise ValueError("Blocks in matrix must be a matrix !")
        if A[block].shape[0] != A[block].shape[1]:
            raise ValueError("Block in matrix must have square blocks !")
        total_size += A[block].shape[0]
    
    A_full = np.zeros(shape=(total_size, total_size))
    
    stride = 0
    for block in A:
        size = A[block].shape[0]
        A_full[stride:stride+size,stride:stride+size] = A[block]
        stride += size

    return A_full

def full_mat_to_block(A, gf_struct):
    if len(A.shape) != 2:
        raise ValueError("Must be a matrix !")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Must be a square matrix !")
    
    A_blocked = dict()
    
    stride = 0
    for bname,ind in gf_struct:
        size = len(ind)
        A_blocked[bname] = A[stride:stride+size, stride:stride+size]
        stride += size
    return A_blocked

def get_mat_block(A, gf_struct, bname_out, kgrid=False):
    have_bname = False
    for bname,ind in gf_struct:
        if bname == bname_out:
            have_bname = True

    if not have_bname:
        raise ValueError("block must be in gf_struct !")
    
    stride = 0
    for bname,ind in gf_struct:
        size = len(ind)
        if bname == bname_out:
            if kgrid:
                if len(A.shape) == 3:
                    return A[:, stride:stride+size, stride:stride+size]
                elif len(A.shape) == 2:
                    return A[:, stride:stride+size]
                else:
                    raise ValueError("For A a matrix on the kgrid the shape must be (N,a,b) or (N,a) !")
            else:
                if len(A.shape) == 2:
                    if A.shape[0] == A.shape[1]:
                        return A[stride:stride+size, stride:stride+size]
                    raise ValueError("Must be a square matrix !")
                else:
                    raise ValueError("Can only project onto a matrix !")

        stride += size
    
    raise RuntimeError("How did we never hit block? Something catastrophically went wrong !")

# Formula is (1-A)^{-1/2} = sum_r=0^{infty} (-1)^r * 1/2 choose r * A^r
def one_sqrtm_inv(A, tol=np.finfo(float).eps, N=10000):
    # Do r = 0 manually (it is just the identity)
    A_r = np.eye(A.shape[0])
    out = np.eye(A.shape[0])
    for r in range(1,N+1):
        old = out.copy()
        A_r = np.dot(A_r,A)
        out += (-1)**r * binom(-1/2., r) * A_r
        err = np.linalg.norm(out - old)
        if err < tol:
            break
    print(r,err)
    return out

def get_K_sq_inv(pdensity, hdensity, tol=np.finfo(float).eps, N=10000):
    return np.dot( one_sqrtm_inv(A=pdensity, tol=tol, N=N), 
                   one_sqrtm_inv(A=hdensity, tol=tol, N=N) )

def get_d(pdensity, ke):
    K = pdensity - np.dot(pdensity,pdensity)
    K_sq = sqrtm(K)
    #K_sq_inv = get_K_sq_inv(pdensity, np.eye(pdensity.shape[0])-pdensity)
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    return np.dot(K_sq_inv,ke.T)

def get_lambda_c(pdensity, R, Lambda, D):
    P = np.eye(pdensity.shape[0]) - 2.0*pdensity
    K = pdensity - np.dot(pdensity,pdensity)
    K_sq = sqrtm(K)
    #K_sq_inv = get_K_sq_inv(pdensity, np.eye(pdensity.shape[0])-pdensity)
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    #return -np.real(np.dot(np.dot(R,D).T, np.dot(K_sq_inv,P))).T - Lambda
    lhs = np.dot( np.dot(R,D).T, np.dot(K_sq_inv,P) ).T
    return -0.5*np.real(lhs + np.conj(lhs)) - Lambda

def get_lambda(R, D, Lambda_c, Nf):
    P = np.eye(Nf.shape[0]) - 2.0*Nf
    K = Nf - np.dot(Nf,Nf)
    K_sq = sqrtm(K)
    #K_sq_inv = get_K_sq_inv(Nf, np.eye(Nf.shape[0])-Nf)
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    #return -np.real(np.dot(np.dot(R,D).T, np.dot(K_sq_inv,P))).T - Lambda_c
    lhs = np.dot( np.dot(R,D).T, np.dot(K_sq_inv,P) )
    return -0.5*np.real(lhs + np.conj(lhs)) - Lambda_c

def get_r(Mcf, Nf):
    K = Nf - np.dot(Nf,Nf)
    K_sq = sqrtm(K)
    #K_sq_inv = get_K_sq_inv(Nf, np.eye(Nf.shape[0])-Nf)
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    return np.dot(Mcf,K_sq_inv).T

def get_f1(Mcf,pdensity,R):
    K = pdensity - np.dot(pdensity,pdensity)
    K_sq = sqrtm(K)
    return Mcf - np.dot(R.T, K_sq)

def get_f2(Nf, pdensity):
    return Nf - pdensity.T

def get_h_qp(R, Lambda, h0_k, mu=0):
    #h_qp = np.einsum('ac,cdk,db->kab', R, h0_k, R.conj().T, optimize='optimal') + (Lambda - mu*np.eye(Lambda.shape[0]))
    h_qp = np.einsum('ac,kcd,db->kab', R, h0_k, R.conj().T) + (Lambda - mu*np.eye(Lambda.shape[0]))
    eig, vec = np.linalg.eigh(h_qp)
    return (eig, vec)

def get_h0_R(R, h0_k, vec):
    #return np.einsum('ack,cd,kdb->kab', h0_k, R.conj().T, vec, optimize='optimal')
    return np.einsum('kac,cd,kdb->kab', h0_k, R.conj().T, vec)

# FIXME add possible rotation
def get_h_qp2(R, Lambda, h0_k, mu=0):
    mesh_num = h0_k.shape[0]
    na = h0_k.shape[1]
    orb_dim = h0_k.shape[3]

    # FIXME what if each inequivalent cluster is not the same internal dimension?
    h_qp = np.zeros(shape=(mesh_num,na*orb_dim,na*orb_dim), dtype=complex)

    for i,j in product(range(na),range(na)):
        the_slice = np.index_exp[:, i*orb_dim:(i+1)*orb_dim, j*orb_dim:(j+1)*orb_dim]
        h_qp[the_slice] = np.matmul( R[i], np.matmul(h0_k[:,i,j,...], R[j].conj().T) )
        #h_qp[the_slice] = np.einsum('ac,kcd,db->kab', R[i], h0_k[:,i,j,...], R[j].conj().T)
        if i == j:
            mu_mat = mu * np.eye(Lambda[i].shape[0])
            h_qp[the_slice] += Lambda[i] - mu_mat

    eig, vec = np.linalg.eigh(h_qp)
    return (eig, vec)

def get_h0_R2(R, h0_k, vec):
    mesh_num = h0_k.shape[0]
    na = h0_k.shape[1]
    orb_dim = h0_k.shape[3]

    h0_R = np.zeros(shape=(mesh_num,na*orb_dim,na*orb_dim), dtype=complex)
    for i,j in product(range(na),range(na)):
        the_slice = np.index_exp[:, i*orb_dim:(i+1)*orb_dim, j*orb_dim:(j+1)*orb_dim]
        h0_R[the_slice] = np.matmul(h0_k[:,i,j,...], R[j].conj().T)
        #h0_R[the_slice] = np.einsum('kac,cb->kab', h0_k[:,i,j,...], R[j].conj().T)
    #A = np.einsum('kac,kcb->kab', h0_R, vec)
    return np.matmul(h0_R, vec) # Right multiply into eigenbasis of quasiparticles


# FIXME add projectors
#\sum_n \sum_k [A_k P_k]_{an} [D_k]_n  [P_k^+ B_k]_{nb}
def get_pdensity(vec, wks):
    vec_dag = np.transpose(vec.conj(), (0,2,1))
    #return np.real( np.einsum('kan,kn,knb->ab', vec, wks, vec_dag, optimize='optimal').T )
    return np.real( np.einsum('kan,kn,knb->ab', vec, wks, vec_dag).T )

# FIXME add projectors
def get_ke(h0_R, vec, wks):
    vec_dag = np.transpose(vec.conj(), (0,2,1))
    #return np.einsum('kan,kn,knb->ab', h0_R, wks, vec_dag, optimize='optimal')
    return np.einsum('kan,kn,knb->ab', h0_R, wks, vec_dag)

def get_sigma_z(mesh_z, R, Lambda, mu = 0.0, e0 = 0.0):
    sigma_z = Gf(mesh = mesh_z, target_shape = R.shape, name = "$\Sigma(z)$")

    #IZ_inv = np.eye(R.shape[0]) - pinv( np.dot(R.conj().T, R) )
    IZ_inv = np.eye(R.shape[0]) - np.linalg.inv( np.dot(R.conj().T, R) )

    #mid = np.dot( np.dot(pinv(R), Lambda), pinv(R.conj().T))
    mid = np.dot( np.dot(np.linalg.inv(R), Lambda), np.linalg.inv(R.conj().T))

    sigma_z << iOmega_n*IZ_inv + mid + mu*IZ_inv - e0
    return sigma_z

def get_gqp_z(g_z, R):
    gp_z = Gf(mesh = g_z.mesh, target_shape = g_z.target_shape, name = "$G^\mathrm{qp}(z)$")
    #gp_z.from_L_G_R( pinv(R.conj().T), g_z, pinv(R) )
    gp_z.from_L_G_R( np.linalg.inv(R.conj().T), g_z, np.linalg.inv(R) )
    return gp_z

def get_gqp_k_z(g_k_z, R, Lambda, h0_k, mu = 0):
    gqp_k_z = g_k_z.copy() # name = "$G^\mathrm{qp}(k,z)$"
    mu_matrix = mu * np.eye(R.shape[0])

    gqp_z = Gf(mesh = gqp_k_z.mesh[1], target_shape = R.shape)

    for k,kay in enumerate(gqp_k_z.mesh.components[0]):
        gqp_z << inverse( iOmega_n  - np.dot( np.dot(R, h0_k[...,k]), R.conj().T ) - Lambda + mu_matrix )
        gqp_k_z[kay,:].data[:] = gqp_z.data
    return gqp_k_z

def get_g_k_z2(gqp_k_z, R):
    g_k_z = gqp_k_z.copy() # name = "$G(k,z)$"
    for k,iw in product(gqp_k_z.mesh.components[0], gqp_k_z.mesh.components[1]):
        g_k_z[k,iw] = np.dot(np.dot(R.conj().T, gqp_k_z[k,iw]), R)
    return g_k_z

def get_g_k_z(g0_k_z, sigma_z):
    g_k_z = g0_k_z.copy() # name = "$G(k,z)$"
    for k in g_k_z.mesh.components[0]:
        g_k_z[k,:].data[:] = dyson(G0_iw = g0_k_z[k,:], Sigma_iw = sigma_z).data
    return g_k_z

def get_pdensity_gf(g_z, R):
    return np.real( get_gqp_z(g_z, R).density() )
    #gp_z = get_gqp_z(g_z, R)
    #return np.real( 0.5*np.eye(R.shape[0]) + np.sum(gp_z.data, axis=0) / gp_z.mesh.beta )

# FIXME I think this needs to have hloc*R^+ subtracted off?
def get_ke_gf(g_z, delta_z, R, mu=0):
    gke_z = Gf(mesh = g_z.mesh, target_shape = g_z.target_shape, name = "$\Delta(z) G(z)$")
    gke_z << delta_z * g_z
    #gke_z << (delta_z + mu) * g_z
    ke = np.sum(gke_z.data, axis=0) / gke_z.mesh.beta
    #ke = gke_z.density()
   
    #return np.dot(ke, pinv(R))
    return np.dot(ke, np.linalg.inv(R)) 
