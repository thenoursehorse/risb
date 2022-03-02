import numpy as np
from itertools import product
from scipy.linalg import sqrtm
from scipy.linalg import pinv
from triqs.gf import Gf, iOmega_n, inverse, dyson

def get_d(pdensity, ke):
    K = pdensity - np.dot(pdensity,pdensity)
    K_sq = sqrtm(K);
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    return np.dot(K_sq_inv,ke.T)

def get_lambda_c(pdensity, R, Lambda, D):
    P = np.eye(pdensity.shape[0]) - 2.0*pdensity
    K = pdensity - np.dot(pdensity,pdensity)
    K_sq = sqrtm(K);
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    #return -np.real(np.dot(np.dot(R,D).T, np.dot(K_sq_inv,P))).T - Lambda
    lhs = np.dot( np.dot(R,D).T, np.dot(K_sq_inv,P) ).T
    return -0.5*np.real(lhs + np.conj(lhs)) - Lambda

def get_lambda(R, D, Lambda_c, Nf):
    P = np.eye(Nf.shape[0]) - 2.0*Nf
    K = Nf - np.dot(Nf,Nf)
    K_sq = sqrtm(K);
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    #return -np.real(np.dot(np.dot(R,D).T, np.dot(K_sq_inv,P))).T - Lambda_c
    lhs = np.dot( np.dot(R,D).T, np.dot(K_sq_inv,P) )
    return -0.5*np.real(lhs + np.conj(lhs)) - Lambda_c

def get_r(Mcf, Nf):
    K = Nf - np.dot(Nf,Nf)
    K_sq = sqrtm(K);
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = np.linalg.inv(K_sq)
    return np.dot(Mcf,K_sq_inv).T

def get_h_qp(R, Lambda, dispersion, mu=0):
    #h_qp = np.einsum('ac,cdk,db->kab', R, dispersion, R.conj().T, optimize='optimal') + (Lambda - mu*np.eye(Lambda.shape[0]))
    h_qp = np.einsum('ac,cdk,db->kab', R, dispersion, R.conj().T) + (Lambda - mu*np.eye(Lambda.shape[0]))
    eig, vec = np.linalg.eigh(h_qp)
    return (eig, vec)

def get_disp_R(R, dispersion, vec):
    #return np.einsum('ack,cd,kdb->kab', dispersion, R.conj().T, vec, optimize='optimal')
    return np.einsum('ack,cd,kdb->kab', dispersion, R.conj().T, vec)


#\sum_n \sum_k [A_k P_k]_{an} [D_k]_n  [P_k^+ B_k]_{nb}
def get_pdensity(vec, wks):
    vec_dag = np.transpose(vec.conj(), (0,2,1))
    #return np.real( np.einsum('kan,kn,knb->ab', vec, wks, vec_dag, optimize='optimal').T )
    return np.real( np.einsum('kan,kn,knb->ab', vec, wks, vec_dag).T )

def get_ke(disp_R, vec, wks):
    vec_dag = np.transpose(vec.conj(), (0,2,1))
    #return np.einsum('kan,kn,knb->ab', disp_R, wks, vec_dag, optimize='optimal')
    return np.einsum('kan,kn,knb->ab', disp_R, wks, vec_dag)

# FIXME add possible rotation
def get_h_qp2(R, Lambda, dispersion, mu=0):
    mesh_num = dispersion.shape[0]
    na = dispersion.shape[1]
    orb_dim = dispersion.shape[3]

    # FIXME what if each inequivalent cluster is not the same internal dimension?
    h_qp = np.zeros(shape=(mesh_num,na*orb_dim,na*orb_dim), dtype=complex)

    for i,j in product(range(na),range(na)):
        mu_mat = mu * np.eye(Lambda[i].shape[0])
        the_slice = np.index_exp[:, i*orb_dim:(i+1)*orb_dim, j*orb_dim:(j+1)*orb_dim]
        h_qp[the_slice] = np.matmul( R[i], np.matmul(dispersion[:,i,j,...], R[j].conj().T) )
        #h_qp[the_slice] = np.einsum('ac,kcd,db->kab', R[i], dispersion[:,i,j,...], R[j].conj().T)
        if i == j:
            h_qp[the_slice] += Lambda[i] - mu_mat

    eig, vec = np.linalg.eigh(h_qp)
    return (eig, vec)

def get_disp_R2(R, dispersion, vec):
    mesh_num = dispersion.shape[0]
    na = dispersion.shape[1]
    orb_dim = dispersion.shape[3]

    disp_R = np.zeros(shape=(mesh_num,na*orb_dim,na*orb_dim), dtype=complex)
    for i,j in product(range(na),range(na)):
        the_slice = np.index_exp[:, i*orb_dim:(i+1)*orb_dim, j*orb_dim:(j+1)*orb_dim]
        disp_R[the_slice] = np.matmul(dispersion[:,i,j,...], R[j].conj().T)
        #disp_R[the_slice] = np.einsum('kac,cb->kab',dispersion[:,i,j,...], R[j].conj().T)
    #A = np.einsum('kac,kcb->kab', disp_R, vec)
    return np.matmul(disp_R, vec) # Right multiply into eigenbasis of quasiparticles

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

def get_gqp_k_z(g_k_z, R, Lambda, dispersion, mu = 0):
    gqp_k_z = g_k_z.copy() # name = "$G^\mathrm{qp}(k,z)$"
    mu_matrix = mu * np.eye(R.shape[0])

    gqp_z = Gf(mesh = gqp_k_z.mesh[1], target_shape = R.shape)

    for k,kay in enumerate(gqp_k_z.mesh.components[0]):
        gqp_z << inverse( iOmega_n  - np.dot( np.dot(R, dispersion[...,k]), R.conj().T ) - Lambda + mu_matrix )
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
