# Copyright (c) 2016 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

import numpy as np
from itertools import product
from triqs.gf import Gf, iOmega_n, inverse, dyson

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
