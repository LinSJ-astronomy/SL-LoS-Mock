import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util import util
import lenstronomy.Util.param_util as param_util
import sys
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import astropy.units as u
sys.path.append('/huawei/osv1/linshijie_BNU/code/lib')
import cfuncs as cf
from astropy.cosmology import FlatLambdaCDM,z_at_value
cosmo = FlatLambdaCDM(H0=69.7, Om0=0.282, Ob0=0.044792699861138666)
from astropy.constants import c
speed_of_light_mpc_per_day = c.to('Mpc/day').value


def calc_lensingMaps(hdu_ini,hdu_fin):
    hdu = fitsio.open(hdu_ini)
    raysImg= hdu[1].data
    hdu.close()
    hdu = fitsio.open(hdu_fin)
    raysSrc= hdu[1].data
    nc_ray = hdu[1].header['nc_ray']
    hdu.close()
    
    daf1 = np.reshape(raysSrc['x1'] - raysImg['x1'], [nc_ray, nc_ray])
    daf2 = np.reshape(raysSrc['x2'] - raysImg['x2'], [nc_ray, nc_ray])
    daf1 = np.rad2deg(daf1)*3600 ###转成角秒
    daf2 = np.rad2deg(daf2)*3600

    A0 = raysSrc['A0']
    A1 = raysSrc['A1']
    A2 = raysSrc['A2']
    A3 = raysSrc['A3']
    # plt.figure()
    # plt.imshow(np.reshape(A0,[nnn,nnn]))
    # plt.colorbar()
    kap = 1.0-0.5*(A0 + A3)
    sh1 =    -0.5*(A0 - A3)
    sh2 =    -0.5*(A1 + A2)
    mua = 1.0/(A0*A3 - A1*A2)
    kap = np.reshape(kap, [nc_ray, nc_ray])
    mua = np.reshape(mua, [nc_ray, nc_ray])
    sh1 = np.reshape(sh1, [nc_ray, nc_ray])
    sh2 = np.reshape(sh2, [nc_ray, nc_ray])
    
    lensingMaps = {}
    lensingMaps['mua'] = mua
    lensingMaps['kappa'] = kap
    lensingMaps['gamma1']= sh1
    lensingMaps['gamma2']= sh2
    lensingMaps['alpha1']= daf1
    lensingMaps['alpha2']= daf2
    lensingMaps['A0'] = np.reshape(A0,[nc_ray, nc_ray])
    lensingMaps['A1'] = np.reshape(A1,[nc_ray, nc_ray])
    lensingMaps['A2'] = np.reshape(A2,[nc_ray, nc_ray])
    lensingMaps['A3'] = np.reshape(A3,[nc_ray, nc_ray])
    lensingMaps['xf1']    = np.reshape(np.rad2deg(raysImg['x1']),[nc_ray, nc_ray]) # degree
    lensingMaps['xf2']    = np.reshape(np.rad2deg(raysImg['x2']),[nc_ray, nc_ray])
    lensingMaps['yf1']    = np.reshape(np.rad2deg(raysSrc['x1']),[nc_ray, nc_ray]) # degree
    lensingMaps['yf2']    = np.reshape(np.rad2deg(raysSrc['x2']),[nc_ray, nc_ray])
    return lensingMaps

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

def Dc(z):                      # Mpc/h
    res = cosmo.comoving_distance(z).value*cosmo.h
    return res
def Dc2redshift(Dc):            # Mpc/h
    return z_at_value(cosmo.comoving_distance,Dc/cosmo.h*u.Mpc).value



case = 111
# read the case.txt file
case_file = '/huawei/osv1/linshijie_BNU/PAPER1/data/image_rays/case{:04}'.format(case)+'/Case.txt'
with open(case_file) as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('mlcase'):
            mlcase = int(line.split('=')[1])
        if line.startswith('highDens'):
            highdencase = int(line.split('=')[1])
f.close()

highden_file = '/huawei/osv1/linshijie_BNU/PAPER1/data/high_dens/case{:04}'.format(highdencase)+'/center_position.txt'
with open(highden_file) as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('num_lensplane'):
            num_lensplane = int(line.split('=')[1])
num_lensplane = 38
point_sources = np.load('/huawei/osv1/linshijie_BNU/PAPER1/data/image_rays/case{:04}'.format(case)+'/PointSources.npz')

# pathes
rays_path      = '/huawei/osv1/linshijie_BNU/PAPER1/data/image_rays/case{:04}/all'.format(case)
potential_path = '/huawei/osv1/linshijie_BNU/PAPER1/data/image_rays/case{:04}/potential_sl.npz'.format(case)
path_mapcells  = '/huawei/osv1/linshijie_BNU/PAPER1/data/high_dens/case{:04}'.format(highdencase)

maxComving = 3800               # in Mpc/h
lensnum    = 76
binL       = maxComving/lensnum
Dc_lens    = binL*(num_lensplane)+binL/2
z_lens     = Dc2redshift(Dc_lens)
z_source   = Dc2redshift(maxComving)
pixel_scale_arcsec = 0.05
piexl_scale_rad    = np.deg2rad(pixel_scale_arcsec/3600.0)


lens_params = read_parameters('/huawei/osv1/linshijie_BNU/PAPER1/data/image_rays/case{:04}'.format(case)+'/LensingParameters.txt')
theta_e = lens_params['theta_E']
phi_q,q = lens_params['phi'],lens_params['q']
print('theta_e:',theta_e)
print('phi:',phi_q)
print('q:',q)
e1, e2 = param_util.phi_q2_ellipticity(phi_q, q)
source_x, source_y = lens_params['source_x'], lens_params['source_y']

lens_model_list = ['SIE']
kwargs_spemd = {'theta_E': theta_e, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}  # parameters of the deflector lens model
# kwargs_shear = {'gamma1': 0.0, 'gamma2': -0.05}  # shear values to the source plane
kwargs_lens = [kwargs_spemd]
lens_model_class = LensModel(lens_model_list,cosmo=cosmo, z_lens=z_lens, z_source=z_source)
lensEquationSolver = LensEquationSolver(lens_model_class)
x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens,
                                                      min_distance=0.05, search_window=5)
print(x_image, y_image)

x_image = point_sources['x_oml']
y_image = point_sources['y_oml']
print(x_image, y_image)
t_days_lenstronomy = lens_model_class.arrival_time(x_image, y_image, kwargs_lens)
ddt_lenstronomy = t_days_lenstronomy-t_days_lenstronomy[0]
print('t_days:',t_days_lenstronomy)
print('ddt_lenstronomy:',ddt_lenstronomy)



potential_oml = lens_model_class.potential(x_image, y_image, kwargs_lens)*(np.pi/180/3600)**2
theta_ij_oml = np.sqrt((x_image-source_x)**2 + (y_image-source_y)**2)*(np.pi/180/3600)

geo_oml = Dc_lens*maxComving/(maxComving-Dc_lens)/cosmo.h/speed_of_light_mpc_per_day*0.5*theta_ij_oml**2
shp_oml = -Dc_lens*maxComving/(maxComving-Dc_lens)/cosmo.h/speed_of_light_mpc_per_day*potential_oml

t_days_oml = geo_oml + shp_oml
print('td_oml:',t_days_oml)

ddtgeo_oml = geo_oml-geo_oml[0]
ddtshp_oml = shp_oml-shp_oml[0]
ddt_oml = t_days_oml-t_days_oml[0]
print('ddtgeo_oml:',ddtgeo_oml)
print('ddtshp_oml:',ddtshp_oml)
print('ddt_oml:',ddt_oml)


print(t_days_lenstronomy/t_days_oml)
cali_factor = np.mean(t_days_lenstronomy/t_days_oml)
print('cali_factor:',cali_factor)


x_multi = point_sources['x_all']
y_multi = point_sources['y_all']
print('Multi-Lensing Images, x:',x_multi,'Multi-Lensing Images, y:',y_multi)
print('Singel-Lensing Images, x:',x_image,'Singel-Lensing Images, y:',y_image)


sorted_index = [0, 1, 2, 3]
x_multi = x_multi[sorted_index]
y_multi = y_multi[sorted_index]
print('Multi-Lensing Images, x:',x_multi,'Multi-Lensing Images, y:',y_multi)
print('Singel-Lensing Images, x:',x_image,'Singel-Lensing Images, y:',y_image)

srcx_lenpl, srcy_lenspl = [x_multi], [y_multi]
srcx_arr, srcy_arr = x_multi, y_multi 
potential_all = []

for i in range(lensnum+1):
    if i == 0:
        continue
    
    # hdu_ini = rays_path + '/testrays{:04}.fits'.format(i-1)
    hdu_fin = rays_path + '/testrays{:04}.fits'.format(i)
    hdu = fitsio.open(hdu_fin)
    raysSrc= hdu[1].data
    nc_ray = hdu[1].header['nc_ray']
    hdu.close()

    # alpha
    srcx = (np.reshape(np.rad2deg(raysSrc['x1']),[nc_ray, nc_ray]))*3600
    srcy = (np.reshape(np.rad2deg(raysSrc['x2']),[nc_ray, nc_ray]))*3600
    srcx_lenspl = cf.call_inverse_cic_omp(srcx,0.0,0.0,[x_multi], [y_multi] , pixel_scale_arcsec)
    srcy_lenspl = cf.call_inverse_cic_omp(srcy,0.0,0.0,[x_multi], [y_multi] , pixel_scale_arcsec)
    
    # srcx_lenspl, srcy_lenspl = xroot1_all_arcsec, xroot2_all_arcsec   
    srcx_arr = np.vstack([srcx_arr,srcx_lenspl])
    srcy_arr = np.vstack([srcy_arr,srcy_lenspl])

    # potential
    if i == num_lensplane+1:
        ###
        # potential = np.load(potential_path)['phi']
        # spx, spy = 0,0
        # potential_img = cf.call_inverse_cic_omp(potential, spx, spy, np.array([srcx_arr[i-1]]), np.array([srcy_arr[i-1]]), pixel_scale_arcsec)
        potential_img = lens_model_class.potential(np.array([srcx_arr[i-1]]), np.array([srcy_arr[i-1]]), kwargs_lens)*(np.pi/180/3600)**2
        print(np.round(np.array([srcx_arr[i-1]]),3), np.round(np.array([srcy_arr[i-1]]),3))
        print(potential_img)
    else:
        potential = np.load(path_mapcells+'/wlmap_middel/data/plane{:04}.npz'.format(i-1))['phi'].T
        spx, spy = 0,0
        potential_img = cf.call_inverse_cic_omp(potential, spx, spy, np.array([srcx_arr[i-1]]), np.array([srcy_arr[i-1]]) , pixel_scale_arcsec)
        DcCurrent = binL*(i-1)+binL/2
        potential_img *= (maxComving-DcCurrent)/maxComving
    if i == 1:
        potential_all = potential_img
    else:
        potential_all = np.vstack([potential_all,potential_img])
    if i == lensnum:
        print(np.round(srcx_lenspl,3), np.round(srcy_lenspl,3))
    # print(potential_img)
    # a=d


    Dij = binL
# TD = np.zeros_like(xroot1_all_arcsec)
TD_geo = np.zeros_like(x_multi)
TD_phi = np.zeros_like(y_multi)
# srcx_arr = np.deg2rad(np.array(srcx_arr)/3600)
# srcy_arr = np.deg2rad(np.array(srcy_arr)/3600)

for CurrentPlNum in range(1,lensnum+1,1):
    DcCurrent = binL*(CurrentPlNum-1)+binL/2
    z_Current = Dc2redshift(DcCurrent)
    Bij = binL*maxComving/((DcCurrent+binL)*(maxComving-DcCurrent))
    binL = 50
    
    if CurrentPlNum == lensnum:
        Bij = 1
        binL = 25
    
    thetaij = np.sqrt((srcx_arr[CurrentPlNum]-srcx_arr[CurrentPlNum-1])**2+(srcy_arr[CurrentPlNum]-srcy_arr[CurrentPlNum-1])**2)*(np.pi/180/3600)
    TD_geo = np.vstack([TD_geo,1/speed_of_light_mpc_per_day * (DcCurrent*(DcCurrent+binL)/binL) /cosmo.h*(0.5*thetaij**2)])
    # print( (DcCurrent*(DcCurrent+binL)/binL)/(2725*3800/(3800-2725)))
    # print(thetaij)
    TD_phi = np.vstack([TD_phi,-1/speed_of_light_mpc_per_day * (DcCurrent*(DcCurrent+binL)/binL)/cosmo.h*(Bij*potential_all[CurrentPlNum-1])])

TD_all_ml = TD_geo + TD_phi
TD_geo_all = np.sum(TD_geo,axis=0)
TD_phi_all = np.sum(TD_phi,axis=0)
TD_all = TD_geo_all + TD_phi_all

print('TD_geo_all:',TD_geo_all)
print('TD_phi_all:',TD_phi_all)
print('TD_all_sorted:',TD_all)

ddtgeo_multilens = TD_geo_all-TD_geo_all[0]
ddtshp_multilens = TD_phi_all-TD_phi_all[0]
ddt_multilens = TD_all-TD_all[0]
print('ddtgeo_multilens:',ddtgeo_multilens)
print('ddtshp_multilens:',ddtshp_multilens)
print('ddt_multilens:',ddt_multilens)

# print('TD_geo_all:',TD_geo_all-TD_geo_all[1])
# print('TD_phi_all:',TD_phi_all-TD_phi_all[1])
# print('TD_all:',(TD_geo_all+TD_phi_all)-(TD_geo_all[1]+TD_phi_all[1]))

print((ddt_multilens-ddt_oml)/ddt_oml)

print('x_oml:',point_sources['x_oml'])
print('y_oml:',point_sources['y_oml'])
print('mag_oml:',point_sources['mag_oml'])
print('ddt_oml:',ddt_oml)
print('x_all:',x_multi)
print('y_all:',y_multi)
mag_pert = point_sources['mag_pert'][sorted_index]
print('mag_pert:',mag_pert)
mag_all = point_sources['mag_all'][sorted_index]
print('mag_all:',mag_all)
print('ddt_multilens:',ddt_multilens)

# print(TD_all_ml)
# for i in range(len(TD_all_ml)):
#     print('TD_phi_{:02}:'.format(i),-(TD_all_ml[i]-TD_all_ml[i][0]))

np.savez('/huawei/osv1/linshijie_BNU/PAPER1/data/image_rays/case{:04}/tdays_ml.npz'.format(case),
         TD_all_ml=TD_all_ml,
         TD_geo=TD_geo,
         TD_phi=TD_phi)
