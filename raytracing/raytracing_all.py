import os,sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
import struct
# import treecorr
import pymaster as nmt
import healpy as hp
import pandas as pd
sys.path.append('/huawei/osv1/linshijie_BNU/code/lib')
import cfuncs as cf
from astropy.cosmology import Planck15,FlatLambdaCDM,z_at_value
from astropy.table import Table
import astropy.units as u

import argparse

parser = argparse.ArgumentParser(description="运行指标 i")
parser.add_argument('--i', type=int, help='指标 i 的值')
args = parser.parse_args()


icase         = args.i
centerposcase = args.i
mlcase = np.random.randint(0, 30)


path_mainplane = '/huawei/osv1/linshijie_BNU/simu_data/mainlens/case{:04}'.format(mlcase)
path_centerpos = '../../data/high_dens/case{:04}'.format(centerposcase)
path_mapcells  = path_centerpos+'/wlmap_middel'

###### The cosmological parameters and the basic constant. 
cosmo = FlatLambdaCDM(H0=69.7, Om0=0.282, Ob0=0.044792699861138666)
Mpc_h = 3.085677e22/cosmo.h     # m
Day   = 31536000.0/365.0        # second per day
vc    = 2.998e5                 # km/s
def Dc(z):                      # Mpc/h
    res = cosmo.comoving_distance(z).value*cosmo.h
    return res
def Dc2redshift(Dc):            # Mpc/h
    return z_at_value(cosmo.comoving_distance,Dc/cosmo.h*u.Mpc).value

# 定义一个函数来读取参数并返回它们作为字典
def read_parameters(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 移除行尾的换行符并分割键和值
            line = line.strip()
            if line:  # 确保行不为空
                key, value_with_unit = line.split(' = ')  # 分割并去除等号两边的空格
                key = key.strip()  # 移除键的首尾空格
                value_with_unit = value_with_unit.strip()  # 移除值和单位的首尾空格
                
                # 提取值部分，并去除单位
                value = value_with_unit.split()[0]  # 分割并取值的部分
                
                # 将值转换为适当的数据类型
                if value.isdigit():  # 如果值是整数
                    params[key] = int(value)
                else:
                    # 假设值是浮点数
                    try:
                        params[key] = float(value)  # 尝试转换为float
                    except ValueError:
                        params[key] = value  # 如果转换失败，保持原样    
    return params

params_wl       = read_parameters(path_mapcells+'/parameters.txt')
params_center   = read_parameters(path_centerpos+'/center_position.txt')
params_mainlens = read_parameters(path_mainplane+'/parameters.txt')

###### The information of the nbody simulation.

center_ra, center_dec = (0,0) # in degree
maxComving = params_wl['maxComving']               # in Mpc/h
slicenum   = params_wl['slicenum']
bsz_deg    = params_wl['bsz']   # in deg
bsz_rad    = np.deg2rad(bsz_deg)
bsz_arcm   = bsz_deg*60

Npix       = params_wl['Npix']
pixel_reso_rad = bsz_rad/Npix
size_mid   = params_wl['Npix_mid']
bsz_mid_rad = pixel_reso_rad*size_mid

binL       = maxComving/slicenum
# main_plane_num = params_center['num_lensplane']
main_plane_num = 38
Dl     = binL*main_plane_num+binL/2
z_lens = Dc2redshift(Dl)


###统一为“弧度”计算
RayDataDemo = np.dtype({'names':['id',      'x',        'xprev',    'alpha',    'A',        'Aprev',    'U',        'phi'],
                        'formats':['int32', '2float64', '2float64', '2float64', '4float64', '4float64', '4float64', 'float64']})
SrcGalDemo  = np.dtype({'names':['id', 'x', 'r'], 'formats':['int32', '2float64', 'float64']})
ImgGalDemo  = np.dtype({'names':['index',          'x',        'A',      'td'],
                        'formats':['int32', '2float64', '4float64', 'float64']})

class RTSimBase(object):
    def __init__(self, bsz, Ngrid, nc_ray=None, OutputRays=True, GridSearch=False):
        self.bsz  = bsz  #In this class, everything is in radians!!!!
        self.Ngrid= Ngrid
        if nc_ray is None:
            self.nc_ray=Ngrid
        self.theta_0 = np.deg2rad(center_ra)
        self.phi_0 = np.deg2rad(center_dec)
        self.init_rays()

        # icase = 0
        self.PathOutput = '../../data/image_rays/case{:04}'.format(icase)+'/all'
        
        if not os.path.exists(self.PathOutput):
            os.makedirs(self.PathOutput)
        # write a txt file to record the parameters
        with open(self.PathOutput+'/../Case.txt', 'w') as f:
            f.write('mlcase = {}\n'.format(mlcase))
            f.write('highDens = {}\n'.format(centerposcase))
        f.close()

            
        self.OutputRays = OutputRays
        self.GridSearch = GridSearch

        
    def init_rays(self):
        #define rays
        self.rays  = np.zeros(self.nc_ray*self.nc_ray, dtype=RayDataDemo)
        #set rays position
        rays_xx1, rays_xx2 = cf.make_r_coor(self.bsz, self.nc_ray)
        print('rays_xx1:', np.rad2deg(rays_xx1)*3600)
        rays_theta= rays_xx1.flatten() +self.theta_0 ##coord-x ###单位是弧度
        rays_phi  = rays_xx2.flatten() +self.phi_0   ##coord-y
        rays_id = np.linspace(0, self.nc_ray*self.nc_ray-1, self.nc_ray*self.nc_ray, dtype=np.int64)

        self.rays['id']      = rays_id
        self.rays['x'][:, 0] = rays_theta
        self.rays['x'][:, 1] = rays_phi
        self.rays['xprev'][:, 0]= 0.0 +self.theta_0
        self.rays['xprev'][:, 1]= 0.0 +self.phi_0

        self.rays['A'][:, 0]    = 1.0
        self.rays['A'][:, 3]    = 1.0
        self.rays['Aprev'][:, 0]= 1.0
        self.rays['Aprev'][:, 3]= 1.0
        del rays_xx1, rays_xx2
        del rays_id, rays_phi, rays_theta
        
        self.rays_xx1 = [] #光线在每个透镜面的位置
        self.rays_xx2 = []
        self.phia_map = []
        self.Dlens    = [] #透镜面的comoving distance
        return None

    def raytracing(self):
        for self.CurrentPlaneNum in range(self.NumLensPlanes):
            # print('CurrentPlaneNum:', self.CurrentPlaneNum)
            
            self.set_lensmap_CurrentPlane()
            self.Dlens.append(self.planeRad)  #recording lens comoving Distance

            #reset ratys alpha & U for CurrentPlane
            self.rays['alpha'] = 0.0
            self.rays['U']     = 0.0
            
            self.poisson_slover() ###calculate alpha & U for rays

            if self.OutputRays:
                self.write_rays_CurrentPlane() ###write rays to fits
            
            self.prop_rays(self.planeRadPlus1, self.planeRad, self.planeRadMinus1)  ###propagate rays to NEXT plane

        if self.OutputRays:
            self.CurrentPlaneNum += 1
            print('CurrentPlaneNum:', self.CurrentPlaneNum)
            self.write_rays_CurrentPlane() ###write rays to fits on maxComovingDistance
        return None


    def prop_rays(self, wp, wpm1, wpm2):
        w1ij= wpm1*(wp-wpm2)/wp/(wpm1-wpm2)
        w2ij= (wp-wpm1)/wp
        
        ###beta_ij
        xp = np.ones_like(self.rays['x'])*0.0
        xp[:, 0] = (1.-w1ij)*self.rays['xprev'][:, 0] + w1ij*self.rays['x'][:, 0] - w2ij*self.rays['alpha'][:, 0]
        xp[:, 1] = (1.-w1ij)*self.rays['xprev'][:, 1] + w1ij*self.rays['x'][:, 1] - w2ij*self.rays['alpha'][:, 1]

        self.rays['xprev'][:, 0] = self.rays['x'][:, 0]
        self.rays['xprev'][:, 1] = self.rays['x'][:, 1]
        self.rays['x'][:, 0] = xp[:, 0]
        self.rays['x'][:, 1] = xp[:, 1]
        
        ###A_ij
        Ap = np.ones_like(self.rays['A'])*0.0
        Ap[:, 0]=(1.-w1ij)*self.rays['Aprev'][:, 0] + w1ij*self.rays['A'][:, 0] - w2ij*(self.rays['U'][:, 0]*self.rays['A'][:, 0] + self.rays['U'][:, 1]*self.rays['A'][:, 2])
        Ap[:, 1]=(1.-w1ij)*self.rays['Aprev'][:, 1] + w1ij*self.rays['A'][:, 1] - w2ij*(self.rays['U'][:, 0]*self.rays['A'][:, 1] + self.rays['U'][:, 1]*self.rays['A'][:, 3])
        Ap[:, 2]=(1.-w1ij)*self.rays['Aprev'][:, 2] + w1ij*self.rays['A'][:, 2] - w2ij*(self.rays['U'][:, 2]*self.rays['A'][:, 0] + self.rays['U'][:, 3]*self.rays['A'][:, 2])
        Ap[:, 3]=(1.-w1ij)*self.rays['Aprev'][:, 3] + w1ij*self.rays['A'][:, 3] - w2ij*(self.rays['U'][:, 2]*self.rays['A'][:, 1] + self.rays['U'][:, 3]*self.rays['A'][:, 3])
    
        self.rays['Aprev'][:, 0] = self.rays['A'][:, 0]
        self.rays['Aprev'][:, 1] = self.rays['A'][:, 1]
        self.rays['Aprev'][:, 2] = self.rays['A'][:, 2]
        self.rays['Aprev'][:, 3] = self.rays['A'][:, 3]
        self.rays['A'][:, 0] = Ap[:, 0]
        self.rays['A'][:, 1] = Ap[:, 1]
        self.rays['A'][:, 2] = Ap[:, 2]
        self.rays['A'][:, 3] = Ap[:, 3]
        del xp, Ap
        return None    
    
    def write_rays_CurrentPlane(self):
        fn = self.PathOutput+'/testrays{:04}.fits'.format(self.CurrentPlaneNum)
        if os.path.exists(fn):
            os.remove(fn)
        hdu = fitsio.PrimaryHDU() #primary image hdu
        #data hdu
        col0= fitsio.Column(name='ID',     format='J', array=self.rays['id'])
        col1= fitsio.Column(name='x1',     format='D', array=self.rays['x'][:, 0])
        col2= fitsio.Column(name='x2',     format='D', array=self.rays['x'][:, 1])
        col3= fitsio.Column(name='A0',     format='D', array=self.rays['A'][:, 0])
        col4= fitsio.Column(name='A1',     format='D', array=self.rays['A'][:, 1])
        col5= fitsio.Column(name='A2',     format='D', array=self.rays['A'][:, 2])
        col6= fitsio.Column(name='A3',     format='D', array=self.rays['A'][:, 3])
        col7= fitsio.Column(name='alpha1', format='D', array=self.rays['alpha'][:, 0])
        col8= fitsio.Column(name='alpha2', format='D', array=self.rays['alpha'][:, 1])
        col9= fitsio.Column(name='phi',    format='D', array=self.rays['phi'])
        
        ehdu= fitsio.BinTableHDU.from_columns([col0, col1, col2, col3, col4, col5, col6, col7, col8, col9])
        ehdu.name = 'LensPlane{:04}'.format(self.CurrentPlaneNum)
        ehdr= ehdu.header
        ehdr.set('iPlane', self.CurrentPlaneNum)
        ehdr.set('nc_ray', self.nc_ray)

        hdulist = fitsio.HDUList([hdu, ehdu])
        hdulist.writeto(fn)
        return None

class pointRTSim(RTSimBase):
    def __init__(self,bsz, Ngrid, nc_ray=None, OutputRays=True, GridSearch=False):
        super().__init__(bsz, Ngrid, nc_ray=nc_ray, OutputRays=OutputRays, GridSearch=GridSearch)

        self.path_mapcells    = path_mapcells
        self.maxComvDistance  = maxComving
        self.NumLensPlanes    = slicenum
        self.NumMainLensPlane = main_plane_num

    def set_lensmap(self):        
        if self.CurrentPlaneNum == self.NumMainLensPlane:

            self.mainlens_path = path_mainplane
            mapcells_data = np.load(self.mainlens_path + '/data.npz')
            print('load mainlens:', self.mainlens_path)
            size_mainlens = np.shape(mapcells_data['alpha1'])[0]
            print('size_mainlens:', size_mainlens)
            print('size_mid:', size_mid)
            self.alpha1_array = mapcells_data['alpha1'][int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2),\
                                                        int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2)]
            self.alpha2_array = mapcells_data['alpha2'][int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2),\
                                                        int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2)]
            self.U0 = mapcells_data['U0'][int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2),\
                                          int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2)]
            self.U1 = mapcells_data['U1'][int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2),\
                                          int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2)]
            self.U2 = mapcells_data['U2'][int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2),\
                                          int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2)]
            self.U3 = mapcells_data['U3'][int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2),\
                                          int(size_mainlens/2)-int(size_mid/2):int(size_mainlens/2)+int(size_mid/2)]
            del mapcells_data.f
            mapcells_data.close()
            
        else:
            
            mapcells_data = np.load(self.path_mapcells + '/data/plane{:04}'.format(self.CurrentPlaneNum)+'.npz')           
            # self.phia_array   = mapcells_data['phi']
            self.alpha1_array = mapcells_data['alpha2']
            self.alpha2_array = mapcells_data['alpha1']
            self.U0 = mapcells_data['U3']
            self.U1 = mapcells_data['U1']
            self.U2 = mapcells_data['U2']
            self.U3 = mapcells_data['U0']
            del mapcells_data.f 
            mapcells_data.close()
        # print('set_lensmap:',self.CurrentPlaneNum)
        
        return None

    def set_lensmap_CurrentPlane(self):
        
        self.set_lensmap()
        
        if self.CurrentPlaneNum == self.NumMainLensPlane:
            self.alpha1_CurrentPlane = self.alpha1_array
            self.alpha2_CurrentPlane = self.alpha2_array
            self.U0_CurrentPlane     = self.U0
            self.U1_CurrentPlane     = self.U1
            self.U2_CurrentPlane     = self.U2
            self.U3_CurrentPlane     = self.U3
        else:
            self.alpha1_CurrentPlane = self.alpha1_array
            self.alpha2_CurrentPlane = self.alpha2_array
            self.U0_CurrentPlane     = self.U0
            self.U1_CurrentPlane     = self.U1
            self.U2_CurrentPlane     = self.U2
            self.U3_CurrentPlane     = self.U3
        
        binL = self.maxComvDistance/self.NumLensPlanes
        if self.CurrentPlaneNum - 1 < 0:
            self.planeRadMinus1 = 0.0
        else:
            self.planeRadMinus1 = (self.CurrentPlaneNum - 1.0)*binL + binL/2.0

        self.planeRad = self.CurrentPlaneNum*binL + binL/2.0
    
        if self.CurrentPlaneNum + 1 == self.NumLensPlanes:
            self.planeRadPlus1 = self.maxComvDistance
        else:
            self.planeRadPlus1 = (self.CurrentPlaneNum + 1.0)*binL + binL/2.0            
        return None

    def poisson_slover(self):
        dsi = self.bsz/self.Ngrid #grid resolution in rad
        nc_ray = self.nc_ray
        al1_arr = self.alpha1_CurrentPlane
        al2_arr = self.alpha2_CurrentPlane
        calc_Uij= False
        if calc_Uij:
            af11, af12 = np.gradient(al1_arr, dsi) #[Uij]
            af21, af22 = np.gradient(al2_arr, dsi)
        else:
            af11 = self.U0_CurrentPlane
            af12 = self.U1_CurrentPlane
            af21 = self.U2_CurrentPlane
            af22 = self.U3_CurrentPlane
        
        if self.CurrentPlaneNum == 0:
            ###alpha_ij
            self.rays['alpha'][:, 0] = np.reshape(al1_arr, nc_ray*nc_ray)
            self.rays['alpha'][:, 1] = np.reshape(al2_arr, nc_ray*nc_ray)
            # self.rays['alpha'][:, 0] = np.zeros(nc_ray*nc_ray)
            # self.rays['alpha'][:, 1] = np.zeros(nc_ray*nc_ray)
            ###U_ij
            self.rays['U'][:, 0] = np.reshape(af11, nc_ray*nc_ray)
            self.rays['U'][:, 1] = np.reshape(af12, nc_ray*nc_ray)
            self.rays['U'][:, 2] = np.reshape(af21, nc_ray*nc_ray)
            self.rays['U'][:, 3] = np.reshape(af22, nc_ray*nc_ray)
        else: 
            ###alpha_ij
            self.rays['alpha'][:, 0] = np.reshape(cf.call_inverse_cic_omp(al1_arr,0.0+self.theta_0,0.0+self.phi_0,np.reshape(self.rays['x'][:, 0],[nc_ray,nc_ray]), np.reshape(self.rays['x'][:, 1],[nc_ray,nc_ray]), dsi), nc_ray*nc_ray)
            self.rays['alpha'][:, 1] = np.reshape(cf.call_inverse_cic_omp(al2_arr,0.0+self.theta_0,0.0+self.phi_0,np.reshape(self.rays['x'][:, 0],[nc_ray,nc_ray]), np.reshape(self.rays['x'][:, 1],[nc_ray,nc_ray]), dsi), nc_ray*nc_ray)
            # self.rays['alpha'][:, 0] = np.zeros(nc_ray*nc_ray)
            # self.rays['alpha'][:, 1] = np.zeros(nc_ray*nc_ray)

            ###U_ij
            self.rays['U'][:, 0] = np.reshape(cf.call_inverse_cic_omp(af11, 0.0+self.theta_0, 0.0+self.phi_0, np.reshape(self.rays['x'][:, 0],[nc_ray,nc_ray]), np.reshape(self.rays['x'][:, 1],[nc_ray,nc_ray]), dsi), nc_ray*nc_ray)
            self.rays['U'][:, 1] = np.reshape(cf.call_inverse_cic_omp(af12, 0.0+self.theta_0, 0.0+self.phi_0, np.reshape(self.rays['x'][:, 0],[nc_ray,nc_ray]), np.reshape(self.rays['x'][:, 1],[nc_ray,nc_ray]), dsi), nc_ray*nc_ray)
            self.rays['U'][:, 2] = np.reshape(cf.call_inverse_cic_omp(af21, 0.0+self.theta_0, 0.0+self.phi_0, np.reshape(self.rays['x'][:, 0],[nc_ray,nc_ray]), np.reshape(self.rays['x'][:, 1],[nc_ray,nc_ray]), dsi), nc_ray*nc_ray)
            self.rays['U'][:, 3] = np.reshape(cf.call_inverse_cic_omp(af22, 0.0+self.theta_0, 0.0+self.phi_0, np.reshape(self.rays['x'][:, 0],[nc_ray,nc_ray]), np.reshape(self.rays['x'][:, 1],[nc_ray,nc_ray]), dsi), nc_ray*nc_ray)
        return None



ttt = pointRTSim(bsz_mid_rad, size_mid, OutputRays=True, GridSearch=False)
ttt.raytracing()