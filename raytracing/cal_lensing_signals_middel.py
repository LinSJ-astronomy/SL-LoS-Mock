import os,sys
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 配置使用Agg后端
import matplotlib.pyplot as plt
import healpy
import astropy.io.fits as fitsio
from astropy.table import Table
import struct

sys.path.append('/lib')
import cfuncs as cf
from astropy.cosmology import Planck15,FlatLambdaCDM,z_at_value
import astropy.units as u
import time
import multiprocessing
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve_fft
from tqdm import tqdm
import argparse

#########################################################################################
# 解析命令行参数
parser = argparse.ArgumentParser(description="运行指标 i")
parser.add_argument('--i', type=int, help='指标 i 的值')
args = parser.parse_args()
case = args.i

pixel_scale    = 0.05                     # in arcsec
Ngrid_particle = 14
Npix_particle  = 2**Ngrid_particle

Ngrid_density  = 11
Npix_density   = 2**Ngrid_density
middel_size    = 200
aver_size      = 100

bsz_arcsec     = pixel_scale*Npix_density # in arcsec
bsz_arcm       = bsz_arcsec/60            # in arcmin
bsz            = bsz_arcm/60              # in deg

maxComving = 3800                         # in Mpc/h
slicenum = 76
binL = maxComving/slicenum
# NumberDensity = 1.671643130068089e-11
NumberDensity = 3.569992510282642e-12
smooth_radius = 3.5                       # in kpc/h
# smooth_radius = 1                         # in arcsec

path_mapcells='../data/lensplane/LensplanesData_flat_005arcsec'

path_out = '../data0/high_dens/case{:04}/wlmap_middel'.format(case)
with open('../data/high_dens/case{:04}/center_position.txt'.format(case)) as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('num_lensplane'):
            num_lensplane = int(line.split('=')[1])
        if line.startswith('max_row'):
            max_row = int(line.split('=')[1])
        if line.startswith('max_col'):
            max_col = int(line.split('=')[1])

if not os.path.exists(path_out):
    os.makedirs(path_out)
if not os.path.exists(path_out+'/fig_alpha'):
    os.makedirs(path_out+'/fig_alpha')
if not os.path.exists(path_out+'/fig_kappa'):
    os.makedirs(path_out+'/fig_kappa')
if not os.path.exists(path_out+'/fig_phi'):
    os.makedirs(path_out+'/fig_phi')
if not os.path.exists(path_out+'/data'):
    os.makedirs(path_out+'/data')

with open(path_out+'/parameters.txt', 'w') as file:
    # 写入参数
    file.write(f'bsz = {bsz} deg\n')  # deg
    # file.write(f'bsz_arcm = {bsz_arcm} arcmin\n')  # arcmin
    # file.write(f'Ngrid = {Ngrid}\n')
    file.write(f'Npix = {Npix_density}\n')
    file.write(f'Npix_mid = {middel_size}\n')
    file.write(f'maxComving = {maxComving} Mpc/h\n')  # Mpc/h
    file.write(f'slicenum = {slicenum}\n')
    file.write(f'binL = {binL} Mpc/h\n')  # Mpc/h
    file.write(f'smooth_radius = {smooth_radius} kpc/h\n')
    file.write(f'NumberDensity = {NumberDensity}\n')
    file.write(f'path_mapcells = {path_mapcells}\n')
#########################################################################################

cosmo = FlatLambdaCDM(H0=69.7, Om0=0.282, Ob0=0.044792699861138666)
vc = 2.998e5 #km/s

def Dc2redshift(Dc):
    return z_at_value(cosmo.comoving_distance,Dc/cosmo.h*u.Mpc).value

def load_mapcells(iplane, path_mapcells):
        
        buf = path_mapcells +"/lensplane{:04}".format(iplane) +".dat"
        with open(buf, "rb") as fp:
            npix_bytes = fp.read(8)  # 读取8个字节的Npix
            npix = int.from_bytes(npix_bytes, byteorder="little")  # 转换为整数

            totmap_bytes = fp.read(npix * 4)  # 读取Npix个float值，每个值占用4个字节

            # Npix_check_bytes = fp.read(8)  # 读取8个字节的Npix，检查结束

        # 解析读取的totmap字节数据为浮点数数组
        totmap_floats = struct.unpack(f"{npix}f", totmap_bytes)
        map = np.array(totmap_floats)
        nside = int(npix**(1/2))
        mapcells=np.reshape(map,(nside,nside))
        return mapcells

def mesh_theta(ngx, ngy, dtheta):
    theta_x = np.linspace(0, ngx-1, ngx)
    theta_y = np.linspace(0, ngy-1, ngy)
    mgrid = np.meshgrid(theta_x, theta_y)
    theta_x = dtheta*(mgrid[0] - ngx/2 +0.5)
    theta_y = dtheta*(mgrid[1] - ngy/2 +0.5)
    return theta_x, theta_y

def zero_padding(arr, npadding):
    ng = np.shape(arr)
    arrPadding = np.zeros([ng[0]+npadding*2, ng[1]+npadding*2])
    arrPadding[npadding:-npadding, npadding:-npadding] = arr
    return arrPadding

def map_cropping(arr, npadding):
    ng = np.shape(arr)
    arr  = arr[npadding:-npadding, npadding:-npadding]
    return arr


def kernel_phi(theta_x, theta_y):
    theta2 = theta_x*theta_x + theta_y*theta_y +1e-12
    theta = np.sqrt(theta2)
    return np.log(theta)/np.pi

def potential_calculator(kappa, dtheta=1, kern_phi=None, convolve=True):
    if kern_phi is not None:
        if convolve == True:
            print('1. Using kappa astropy.conv kern to calculate potential.', flush=True)
            phi = convolve_fft(kappa, kern_phi, normalize_kernel=False, nan_treatment='fill', fft_pad=False, boundary='wrap',allow_huge=True)
            phi = phi*dtheta*dtheta  #dtheta*dtheta - normalization
        else:
            print('2. Using kappa fft.conv kern to calculate potential.', flush=True)
            kappa_fft = np.fft.fft2(kappa) #F(kappa)
            kern_phi_fft = np.fft.fft2(kern_phi) #F(kappa)
            phi_fft = kappa_fft*kern_phi_fft
            phi_fft[0,0] = 0.0 #为了跟-3做比较，可以手动将0频置零
            phi = np.fft.fftshift(np.fft.ifft2(phi_fft)).real
            phi = phi*dtheta*dtheta  #dtheta*dtheta - normalization
    else:
        '''\nabla**2 phi = 2*kappa'''
        print('3. Using Poisson equ. to calculate potential.', flush=True)
        kappa_fft = np.fft.fft2(kappa) #F(kappa)

        ng = np.shape(kappa)[0]
        k     = 2.*np.pi*np.fft.fftfreq(ng, dtheta) #角频率k=ig*2*pi/(T=ng*dtheta)
        kmesh = np.meshgrid(k,k) #2D角频率
        kk2   = kmesh[0]**2 + kmesh[1]**2 #k**2
        kk2[0,0] = 1. #预设0频,数组操作避免除0
        phi_fft = 2.*kappa_fft/(-1.*kk2) #F(phi) = 2*F(kappa)/(-k**2)
        phi_fft[0,0] = 0.0 #0频置零
        phi = np.fft.ifft2(phi_fft).real #iFFT
    return phi

def potential_calculator_intergral(kappa, dtheta=1):
    ng = np.shape(kappa)[0]
    theta_x, theta_y, theta = mesh_theta(ng, ng, dtheta)
    phi = np.zeros([ng, ng])
    for i in tqdm(range(ng)):
        for j in range(ng):
            xi = theta_x[i,j]
            yi = theta_y[i,j]
            theta2 = (theta_x-xi)**2 + (theta_y-yi)**2 +1e-12
            kernel_phi = np.log(np.sqrt(theta2))/np.pi
            phi[i,j] = np.sum(kappa*kernel_phi)*dtheta*dtheta
        
    return phi

def cal_lensing_signals(sds, bzz, ncc, Dl):
    #dcc = bzz/ncc
    zl = Dc2redshift(Dl)
    kappa = 3*cosmo.H0.value**2*cosmo.Om0/(2*vc*vc) * binL * (1+zl) * (Dl) / (cosmo.h)**2\
        *((sds-NumberDensity*((Dl+1/2*binL)**3-(Dl-1/2*binL)**3))/(NumberDensity*((Dl+1/2*binL)**3-(Dl-1/2*binL)**3)))
    
    smooth_radius_arcsec = (smooth_radius/1000/Dl)/np.pi*180*3600
    print(Dl,np.round(smooth_radius_arcsec,3))
    # filtering with a gaussian kernel
    kappa = gaussian_filter(kappa,sigma=smooth_radius_arcsec/pixel_scale)
    # deflection maps
    alpha2, alpha1 = cf.call_cal_alphas(kappa, bzz, ncc)
    al12, al11, al22, al21 = cf.call_lanczos_derivative(alpha1, alpha2, bzz, ncc)
    alpha1_mean = np.mean(alpha1[int(ncc/2)-int(aver_size/2):int(ncc/2)+int(aver_size/2),\
                                 int(ncc/2)-int(aver_size/2):int(ncc/2)+int(aver_size/2)])
    alpha2_mean = np.mean(alpha2[int(ncc/2)-int(aver_size/2):int(ncc/2)+int(aver_size/2),\
                                 int(ncc/2)-int(aver_size/2):int(ncc/2)+int(aver_size/2)])
    # select the middel part of the maps
    alpha1 = alpha1[int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2),int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2)]
    alpha2 = alpha2[int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2),int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2)]
    al11 = al11[int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2),int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2)]
    al12 = al12[int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2),int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2)]
    al21 = al21[int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2),int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2)]
    al22 = al22[int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2),int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2)]

    # calculate the potential
    npadding = int(ncc/2)
    kappa0 = zero_padding(kappa, npadding)
    ng = np.shape(kappa0)[0]
    theta_x, theta_y = mesh_theta(ng, ng, pixel_scale/3600*np.pi/180)
    kern_phi = kernel_phi(theta_x, theta_y)
    phi = potential_calculator(kappa0, dtheta=pixel_scale/3600*np.pi/180, kern_phi=kern_phi, convolve=True)
    phi = map_cropping(phi, npadding)
    phi = phi[int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2),int(ncc/2)-int(middel_size/2):int(ncc/2)+int(middel_size/2)]
    phi = phi - phi[int(middel_size/2),int(middel_size/2)]
    theta_x_middel, theta_y_middel = mesh_theta(middel_size, middel_size, pixel_scale/3600*np.pi/180)
    phi = phi - theta_x_middel*alpha1_mean - theta_y_middel*alpha2_mean

    return alpha1-alpha1_mean, alpha2-alpha2_mean, kappa, al11, al12, al21, al22, phi

def set_lensmap(bsz, Npix, CurrentPlaneNum, path_mapcells, maxComvDistance, NumLensPlanes):        
    # print('load lensmap:',CurrentPlaneNum)
    sds = load_mapcells(CurrentPlaneNum, path_mapcells)
    sds = sds[max_row-int(Npix/2):max_row+int(Npix/2),max_col-int(Npix/2):max_col+int(Npix/2)]

    binL = maxComvDistance/NumLensPlanes
    Dl = maxComvDistance/NumLensPlanes*CurrentPlaneNum+binL/2

    # print('lensing calculator:',CurrentPlaneNum)
    alpha1, alpha2, kappa, al11, al12, al21, al22, phi =\
             cal_lensing_signals(sds, bsz, Npix, Dl)
    
    fig,axs = plt.subplots(1,3,figsize=(12,4))
    # # Plot alpha
    im1 = axs[0].imshow(np.rad2deg(alpha1)*3600,vmin = -0.1,vmax=0.1)
    axs[0].set_title('alpha1')
    # Plot alpha2_now
    im2 = axs[1].imshow(np.rad2deg(alpha2)*3600,vmin = -0.1,vmax=0.1)
    axs[1].set_title('alpha2')
    # Plot alpha2_tolens
    im3 = axs[2].imshow(np.rad2deg(np.sqrt(alpha1**2+alpha2**2))*3600,vmin = 0,vmax=0.5)
    axs[2].set_title('alpha')
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])
    fig.colorbar(im3, ax=axs[2])
    plt.savefig(path_out+'/fig_alpha/fig{:04}.png'.format(CurrentPlaneNum))
    plt.close()

    plt.figure(figsize=(8,10))
    plt.imshow(kappa*(maxComvDistance-Dl)/maxComvDistance,vmin = 0, vmax = 0.5)
    plt.colorbar()
    plt.savefig(path_out+'/fig_kappa/fig{:04}.png'.format(CurrentPlaneNum))
    plt.close()

    plt.figure(figsize=(8,10))
    plt.imshow(phi)
    plt.colorbar()
    plt.savefig(path_out+'/fig_phi/fig{:04}.png'.format(CurrentPlaneNum))
    plt.close()

    np.savez(path_out+'/data/plane{:04}'.format(CurrentPlaneNum),\
             alpha1 = alpha1, alpha2 = alpha2,\
                  U0 = al11, U1 = al12, U2 = al21, U3 = al22, kappa=kappa, phi=phi)
    del alpha1, alpha2, al11, al12, al21, al22, kappa, phi
    # del kappa
    return None

def process_slice(i):
    time_begin = time.time()
    set_lensmap(np.deg2rad(bsz), Npix_density, i, path_mapcells, maxComving, slicenum)
    time_end = time.time()
    print("Slice", i, "completed in", time_end - time_begin, "seconds")

if __name__ == "__main__":
    with multiprocessing.Pool(processes=16) as pool:
        pool.map(process_slice, range(slicenum))