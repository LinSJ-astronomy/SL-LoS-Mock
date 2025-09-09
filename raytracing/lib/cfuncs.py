# from astropy.cosmology import Planck15
# cosmo = Planck15


import numpy as np
from astropy.cosmology import FlatLambdaCDM
#cosmo = FlatLambdaCDM(H0=72, Om0=0.26)
cosmo = FlatLambdaCDM(H0=71, Om0=0.264, Ob0=0.044792699861138666)

vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2
#apr = 206269.43
apr = 180.0/np.pi*3600.0
Mpc_h = 3.085677e22/cosmo.h       # m
Day = 31536000.0/365.0  # second per day


def Dc(z):
    res = cosmo.comoving_distance(z).value*cosmo.h
    return res

def Dc2(z1,z2):
    Dcz1 = (cosmo.comoving_distance(z1).value*cosmo.h)
    Dcz2 = (cosmo.comoving_distance(z2).value*cosmo.h)
    res = (Dcz2-Dcz1+1e-8)
    return res

def Da(z):
    res = cosmo.comoving_distance(z).value*cosmo.h/(1+z)
    return res

def Da2(z1,z2):
    Dcz1 = (cosmo.comoving_distance(z1).value*cosmo.h)
    Dcz2 = (cosmo.comoving_distance(z2).value*cosmo.h)
    res = (Dcz2-Dcz1+1e-8)/(1+z2)
    return res

def Dl(z):
    res = cosmo.luminosity_distance(z).value*cosmo.h
    return res

def sigma_crit(z1, z2):
    res = vc*vc/(4.0*np.pi*G)*Da(z2)/(Da(z1)*Da2(z1,z2))
    return res

#---------------------------------------------------------------------------------
import ctypes as ct

lib_path = "lib/"
#---------------------------------------------------------------------------------
sps = ct.CDLL(lib_path+"lib_so_sph_w_omp/libsphsdens.so")

sps.cal_sph_sdens_weight.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    ct.c_float,ct.c_long,ct.c_float,ct.c_long,ct.c_long, \
                                    ct.c_float,ct.c_float,ct.c_float, \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float)]

sps.cal_sph_sdens_weight.restype  = ct.c_int

def call_sph_sdens_weight(x1,x2,x3,mp,Bsz,Ncc):

    x1_in = np.array(x1,dtype=ct.c_float)
    x2_in = np.array(x2,dtype=ct.c_float)
    x3_in = np.array(x3,dtype=ct.c_float)
    mp_in = np.array(mp,dtype=ct.c_float)
    dcl = ct.c_float(Bsz/Ncc)
    Ngb = ct.c_long(32)
    xc1 = ct.c_float(0.0)
    xc2 = ct.c_float(0.0)
    xc3 = ct.c_float(0.0)
    Np  = len(mp)
    posx1 = np.zeros((Ncc,Ncc),dtype=ct.c_float)
    posx2 = np.zeros((Ncc,Ncc),dtype=ct.c_float)
    sdens = np.zeros((Ncc,Ncc),dtype=ct.c_float)

    sps.cal_sph_sdens_weight(x1_in,x2_in,x3_in,mp_in,ct.c_float(Bsz),ct.c_long(Ncc),dcl,Ngb,ct.c_long(Np),xc1,xc2,xc3,posx1,posx2,sdens);
    return sdens
#---------------------------------------------------------------------------------


sps.cal_sph_sdens_weight_omp.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        ct.c_float,ct.c_long,ct.c_float,ct.c_long,ct.c_long, \
                                        ct.c_float,ct.c_float,ct.c_float, \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float)]

sps.cal_sph_sdens_weight_omp.restype  = ct.c_int

def call_sph_sdens_weight_omp(x1,x2,x3,mp,Bsz,Nc):

    x1_in = np.array(x1,dtype=ct.c_float)
    x2_in = np.array(x2,dtype=ct.c_float)
    x3_in = np.array(x3,dtype=ct.c_float)
    mp_in = np.array(mp,dtype=ct.c_float)
    dsx = ct.c_float(Bsz/Nc)
    Ngb = ct.c_long(32)
    xc1 = ct.c_float(0.0)
    xc2 = ct.c_float(0.0)
    xc3 = ct.c_float(0.0)
    Np  = len(mp)
    posx1 = np.zeros((Nc,Nc),dtype=ct.c_float)
    posx2 = np.zeros((Nc,Nc),dtype=ct.c_float)
    sdens = np.zeros((Nc,Nc),dtype=ct.c_float)

    sps.cal_sph_sdens_weight_omp(x1_in,x2_in,x3_in,mp_in,ct.c_float(Bsz),ct.c_long(Nc),dsx,Ngb,ct.c_long(Np),xc1,xc2,xc3,posx1,posx2,sdens);
    return sdens

#---------------------------------------------------------------------------------
gls = ct.CDLL(lib_path+"lib_so_cgls/libglsg.so")
gls.kappa0_to_alphas.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                 ct.c_int,ct.c_double,\
                                 np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                 np.ctypeslib.ndpointer(dtype = ct.c_double)]
gls.kappa0_to_alphas.restype  = ct.c_void_p

def call_cal_alphas(Kappa, Bsz, Ncc):
    kappa0 = np.array(Kappa,dtype=ct.c_double)
    alpha1 = np.array(np.zeros((Ncc,Ncc)),dtype=ct.c_double)
    alpha2 = np.array(np.zeros((Ncc,Ncc)),dtype=ct.c_double)
    gls.kappa0_to_alphas(kappa0,Ncc,Bsz,alpha1,alpha2)
    return alpha1,alpha2

gls.kappa0_to_phi.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                              ct.c_int,ct.c_double,\
                              np.ctypeslib.ndpointer(dtype = ct.c_double)]
gls.kappa0_to_phi.restype  = ct.c_void_p

def call_cal_phi(Kappa, Bsz, Ncc):

	kappa0 = np.array(Kappa,dtype=ct.c_double)
	phi = np.array(np.zeros((Ncc,Ncc)),dtype=ct.c_double)
	gls.kappa0_to_phi(kappa0,Ncc,Bsz,phi)

	return phi

#--------------------------------------------------------------------
lzos = ct.CDLL(lib_path+"lib_so_lzos/liblzos.so")
lzos.lanczos_diff_2_tag.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    ct.c_double,ct.c_int,ct.c_int]
lzos.lanczos_diff_2_tag.restype  = ct.c_void_p

def call_lanczos_derivative(alpha1,alpha2,Bsz,Ncc):

    dif_tag = 2

    dcl = Bsz/Ncc

    m1 = np.array(alpha1,dtype=ct.c_double)
    m2 = np.array(alpha2,dtype=ct.c_double)

    m11 = np.zeros((Ncc, Ncc))
    m12 = np.zeros((Ncc, Ncc))
    m21 = np.zeros((Ncc, Ncc))
    m22 = np.zeros((Ncc, Ncc))

    lzos.lanczos_diff_2_tag(m1,m2,m11,m12,m21,m22,ct.c_double(dcl),ct.c_int(Ncc),ct.c_int(dif_tag))

    return m11,m12,m21,m22


#一阶导
lzos.lanczos_diff_1_tag.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    ct.c_double,ct.c_int,ct.c_int]
lzos.lanczos_diff_1_tag.restype  = ct.c_void_p

def call_lanczos_derivative_phi(phi, Bsz, Ncc):
    dif_tag = 2
    dcl = Bsz/Ncc
    mi = np.array(phi, dtype=ct.c_double)
    m1 = np.zeros((Ncc, Ncc))
    m2 = np.zeros((Ncc, Ncc))
    lzos.lanczos_diff_1_tag(mi, m1,m2,ct.c_double(dcl),ct.c_int(Ncc),ct.c_int(dif_tag))
    return m1,m2


#---------------------------------------------------------------------------------
rtf = ct.CDLL(lib_path + "lib_so_icic/librtf.so")

rtf.inverse_cic.argtypes = [np.ctypeslib.ndpointer(dtype =  ct.c_double),\
                            np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                            np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                            ct.c_double,ct.c_double,ct.c_double, \
                            ct.c_int,ct.c_int,ct.c_int,ct.c_int,\
                            np.ctypeslib.ndpointer(dtype = ct.c_double)]
rtf.inverse_cic.restype  = ct.c_void_p

def call_inverse_cic(img_in, yc1, yc2, yi1, yi2, dsi):
    ny1,ny2 = np.shape(img_in)
    nx1,nx2 = np.shape(yi1)

    img_in = np.array(img_in,dtype=ct.c_double)

    yi1 = np.array(yi1,dtype=ct.c_double)
    yi2 = np.array(yi2,dtype=ct.c_double)

    img_out = np.zeros((nx1,nx2))

    rtf.inverse_cic(img_in,yi1,yi2,ct.c_double(yc1),ct.c_double(yc2),ct.c_double(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nx1),ct.c_int(nx2),img_out)
    return img_out.reshape((nx1,nx2))

#--------------------------------------------------------------------
rtf.inverse_cic_omp.argtypes = [np.ctypeslib.ndpointer(dtype =  ct.c_double),\
                                np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                                np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                                ct.c_double,ct.c_double,ct.c_double, \
                                ct.c_int,ct.c_int,ct.c_int,ct.c_int,\
                                np.ctypeslib.ndpointer(dtype = ct.c_double)]
rtf.inverse_cic_omp.restype  = ct.c_void_p

def call_inverse_cic_omp(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    nx1,nx2 = np.shape(yi1)
    img_in = np.array(img_in,dtype=ct.c_double)
    yi1 = np.array(yi1,dtype=ct.c_double)
    yi2 = np.array(yi2,dtype=ct.c_double)
    img_out = np.zeros((nx1,nx2))

    rtf.inverse_cic_omp(img_in,yi1,yi2,ct.c_double(yc1),ct.c_double(yc2),ct.c_double(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nx1),ct.c_int(nx2),img_out)
    return img_out.reshape((nx1,nx2))

#--------------------------------------------------------------------
rtf.inverse_cic_single.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   ct.c_float,ct.c_float,ct.c_float,ct.c_int,ct.c_int,ct.c_int, \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float)]
rtf.inverse_cic_single.restype  = ct.c_void_p

def call_inverse_cic_single(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    img_in = np.array(img_in,dtype=ct.c_float)
    yi1 = np.array(yi1,dtype=ct.c_float)
    yi2 = np.array(yi2,dtype=ct.c_float)
    nlimgs = len(yi1)
    img_out = np.zeros((nlimgs),dtype=ct.c_float)

    rtf.inverse_cic_single(img_in,yi1,yi2,ct.c_float(yc1),ct.c_float(yc2),ct.c_float(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nlimgs),img_out)
    return img_out

#--------------------------------------------------------------------
rtf.inverse_cic_omp_single.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       ct.c_float,ct.c_float,ct.c_float,ct.c_int,ct.c_int,ct.c_int, \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float)]
rtf.inverse_cic_omp_single.restype  = ct.c_void_p

def call_inverse_cic_single_omp(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    img_in = np.array(img_in,dtype=ct.c_float)
    yi1 = np.array(yi1,dtype=ct.c_float)
    yi2 = np.array(yi2,dtype=ct.c_float)
    nlimgs = len(yi1)
    img_out = np.zeros((nlimgs),dtype=ct.c_float)

    rtf.inverse_cic_omp_single(img_in,yi1,yi2,ct.c_float(yc1),ct.c_float(yc2),ct.c_float(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nlimgs),img_out)
    return img_out

#--------------------------------------------------------------------
tri = ct.CDLL(lib_path+"lib_so_tri_roots/libtri.so")
tri.mapping_triangles.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  ct.c_int, \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.mapping_triangles.restype  = ct.c_void_p

def call_mapping_triangles(pys,xi1,xi2,yi1,yi2):

    pys_in = np.array(pys,dtype=ct.c_double)
    xi1_in = np.array(xi1,dtype=ct.c_double)
    xi2_in = np.array(xi2,dtype=ct.c_double)
    yi1_in = np.array(yi1,dtype=ct.c_double)
    yi2_in = np.array(yi2,dtype=ct.c_double)
    nc_in = ct.c_int(np.shape(xi1)[0])

    xroots_out = np.array(np.ones(20)*-99999, dtype=ct.c_double)

    tri.mapping_triangles(pys_in,xi1_in,xi2_in,yi1_in,yi2_in,nc_in,xroots_out)

    aroots = len(xroots_out[xroots_out!=-99999])

    xroot1 = xroots_out[:aroots:2]
    xroot2 = xroots_out[1:aroots:2]

    return xroot1, xroot2
#--------------------------------------------------------------------

def make_r_coor(bs, nc):
    ds = bs/nc
    x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2,x1 = np.meshgrid(x1,x2)
    return x1,x2

def make_c_coor(bs, nc):
    ds = bs/nc
    x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x1,x2 = np.meshgrid(x1,x2)
    return x1,x2


def mags_to_vd(mg,mr,zz):

    Dlum=Dl(zz)
    Mabsr=mr-5.0*np.log10(Dlum/cosmo.h)-25.0;

    ## Get the SDSS magnitude
    mrsdss=Mabsr+0.024*(mg-mr)/0.871;

    ## Transfer to r' = ^{0.1}r Hubble type E, Frei and Gunn
    mrsdss=mrsdss-0.11;

    ## Assume that the luminosity function evolves such that Mr* decline by 1.5
    ## magnitudes from 0.0 to 1.0, this is similar to the evolution in the B band
    ## found by Faber et al. 2007 from DEEP2 and COMBO-17. This is really an adhoc
    ## prescription but nothing better known to me right now.

    mrstar=(-20.44)+(zz-0.1)*1.5;
    LbyLstar=10.0**(-0.4*(mrsdss-mrstar));

    ## Parker et al. 2007, Table 1 - Bright sample
    return 142.0*LbyLstar**(1./3.);
