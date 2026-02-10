import os
import emcee
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from lmfit.lineshapes import voigt
from matplotlib.bezier import inside_circle
from scipy.interpolate import interp1d
#from pythreejs import parameters
from scipy.special import wofz
from lmfit import minimize, Parameters,report_fit
import pickle
from matplotlib.ticker import FixedLocator, MultipleLocator
if 1:
    # ==== PARAMETERS ==================

    c = 2.99792e10		# cm/s
    m_e = 9.1095e-28		# g
    e = 4.8032e-10		# cgs units


    # ==== VOIGT PROFILE ===============
    def H(a, x):
        P = x**2
        H0 = np.exp(-x**2)
        Q = 1.5/x**2
        return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


    def Voigt(l, l0, f, N, b, gam, z=0):
        """Calculate the Voigt profile of transition with
        rest frame transition wavelength: 'l0'
        oscillator strength: 'f'
        column density: N  cm^-2
        velocity width: b  Km/s
        """
        b = b*100000
        N = 10**N
        # ==================================
        # Calculate Profile

        C_a = np.sqrt(np.pi) * e**2 * f * l0 * 1.e-8 / m_e / c / b
        a = l0*1.e-8 * gam / (4.*np.pi * b)

        dl_D = b / c * l0
        l = l / (z + 1.)
        x = (l - l0) / dl_D + 0.0001

        tau = np.float64(C_a) * N * H(a, x)

        return tau

if 0:
    def h(a,u):
        return np.real(wofz(a+1j*u))

    def T_dla(lam,lam0,f,N_HI,b,gamma,zdla):
         # b is in km/s
        c = 2.99792e8		# m/s
        m_e = 9.1095e-28		# g
        e = 4.8032e-10		# cgs units

        #gamma = 6.625e8 #s^-1
        a = gamma*lam0/(4*np.pi*b*1e5)
        lam_obs = lam0*(1+zdla)
        u = c*(lam-lam_obs)/(lam_obs*b)

        k = np.sqrt(np.pi)*np.pow(e,2)/(m_e*c*b*1e5)

        tau = N_HI*k*f*h(a,u)

        return tau

def gauss(x, s):
    return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x / s) ** 2)

def errf_v2(x):
    a = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277]
    t = 1 / (1 + 0.5 * np.abs(x))
    tau = t * np.exp(-x ** 2 + a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * (a[4] + t * (a[5] + t * (a[6] + t * (a[7] + t * (a[8] + t * a[9])))))))))
    if x >= 0:
        return 1 - tau
    else:
        return tau - 1

def convolve_res2(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced.

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float
                        resolution of the instrument function. Assumed to be constant with wavelength.
                        i.e. the width of the instrument function is linearly dependent on wavelenth.

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    #sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    f = 1 - f

    il = 0
    for i, x in enumerate(l):
        sig = x / R / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        s = f[il] * (1 - errf_v2((x - l[il]) / np.sqrt(2) / sig)) / 2
        #ie = il + 30
        while k < n-1 and l[k+1] < x + delta * sig:
            #s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += (f[k+1] * gauss(l[k+1] - x, sig) + f[k] * gauss(l[k] - x, sig)) / 2 * (l[k+1] - l[k])
            #print(i, k , gauss(l[k] - x, sig))
            k += 1
        #input()
        s += f[k] * (1 - errf_v2(np.abs(l[k] - x) / np.sqrt(2) / sig)) / 2
        fc[i] = s

    return 1 - fc

def I(x):
    x = np.clip(x, 1e-6, 0.999999)
    t1 = (x**4.5)/(1-x)
    t2 = (9/7)*(x**3.5)
    t3 = (9/5)*(x**2.5)
    t4 = 3*x**1.5
    t5 = 9*x**0.5
    t6 = -4.5*(np.log((1+(x**0.5))/(1-(x**0.5))))
    val = t1+t2+t3+t4+t5+t6
    return val

def T_GP(z):
    tau = 3.88e5*((1+z)/7)**1.5
    return tau

def T_IGM_l(lam,x_HI,z_s,z_n):
    lam_alpha = 1215.6736
    t_0 = T_GP(z_s)
    delta = (lam-lam_alpha*(1+z_s))/(lam_alpha*(1+z_s))
    R = 2.02e-8
    x1 = (1+z_n)/((1+z_s)*(1+delta))
    x2 = 1/(1+delta)
    val = t_0*R*(I(x2)-I(x1))*(1+delta)**1.5
    return x_HI*val/np.pi

def f_intr(z_s,pl,sigma,A_ly,A_NV):
    lam_alpha = 1215.6736
    lam_NV1 = 1238.82
    lam_NV2 = 1242.8
    f = 0.67

    lam_g_mask = np.linspace(1180 * (1 + z_s), 1400 * (1 + z_s), 1000)
    mask = (pl_fe[:, 0] > lam_g_mask[0]) * (pl_fe[:, 0] < lam_g_mask[-1])
    lam = pl_fe[:, 0][mask]
    pl = pl_fe[:, 1][mask]

    lam_alpha_s = lam_alpha*(1+z_s)
    lam_NV1_s= lam_NV1*(1+z_s)
    lam_NV2_s = lam_NV2*(1+z_s)


    ly = np.exp(-(lam-lam_alpha_s)**2/(2*sigma**2))
    NV = f*np.exp(-(lam-lam_NV1_s)**2/(2*sigma**2))+(1-f)*np.exp(-(lam-lam_NV2_s)**2/(2*sigma**2))
    con = pl +  A_ly*ly + A_NV*NV

    return lam,con


def f_intr_mod(z_s,pl_fe,sigma_ly,sigma_NV,A_ly,A_NV,lam_NV,lam_start,lam_stop):
    lam_alpha = 1215.6736
    #lam_NV1 = 1238.82
    #lam_NV2 = 1242.8
    f = 0.67

    pl_fe_intr = interp1d(pl_fe[:, 0]*(1+z_s), pl_fe[:, 1])
    #lam = np.linspace(1180 * (1 + z_s), 1260 * (1 + z_s), 5000)
    lam = np.linspace(lam_start, lam_stop, 500)
    #mask = (pl_fe[:, 0] > lam_g_mask[0]) * (pl_fe[:, 0] < lam_g_mask[-1])
    #lam = pl_fe[:, 0][mask]
    #pl = pl_fe[:, 1][mask]
    pl = pl_fe_intr(lam)

    lam_alpha_s = lam_alpha*(1+z_s)
    #lam_NV1_s= lam_NV1*(1+z_s)
    lam_NV_s = lam_NV*(1+z_s)


    ly = np.exp(-(lam-lam_alpha_s)**2/(2*sigma_ly**2))
    f_1238 =1.560E-01
    f_1242 = 7.770E-02
    NV = np.exp(-(lam-lam_NV_s)**2/(2*sigma_NV**2))#+np.exp(-(lam-lam_NV1_s)**2/(2*sigma_NV**2))*((A_NV/f_1242)**(f_1238-f_1242))
    con = pl +  A_ly*ly + A_NV*NV
    cont_intr = interp1d(lam,con)
    return lam,cont_intr


def model(params,z_n,pl_fe,intr_cont,sp,lam_start,lam_stop):
    z_s = params["z_s"].value
#    z_s_v = params["z_s_v"].value
    IGM_v = params["IGM_v"].value
#    z_n = params["z_n"].value
    #z_DLA = params["z_DLA"].value
    sigma_ly = params["sigma_ly"].value
    #sigma_NV = params["sigma_NV"].value
    sigma_NV = params["sigma_NV"].value
    A_ly = params["A_ly"].value
    A_NV = params["A_NV"].value
    x_HI = params["x_HI"].value
    #NH = params["NH"].value
    #b = params["b"].value

    z_s_v = z_s - IGM_v
    #sigma_NV = del_NV
    #intrnsic spectra
    lam, intrn_flux_fn = f_intr_mod(z_s, pl_fe, sigma_ly, sigma_NV, A_ly, A_NV,lam_start,lam_stop)
    intrn_flux = intrn_flux_fn(lam)
    qs_spline = intr_cont(np.log10(lam / (1 + z_s)))
    #mask_intr = (lam > 1218 * (1 + z_s)) & (lam < 1275 * (1 + z_s))
    #res1 = qs_spline[mask_intr] - intrn_flux[mask_intr]

    #adding DLA and IGM absorption
    #tau_dla = Voigt(lam, 1215.6736, 1.388E-01, NH, b, 6.265E+08, z=z_DLA)

    tau_IGM = T_IGM_l(lam, x_HI, z_s_v, z_n)
    tau_tot = tau_IGM

    model_flux = intrn_flux*np.exp(-tau_tot)
    model_flux_fn = interp1d(lam,model_flux)

    mask_1 = (sp[:,0]>=lam_start)*(sp[:,0]<=lam_stop)
    mask_for_fit = mask_1*(sp[:,3]==False)
    #res = qs_spline-model_flux #
    res = (sp[:,1][mask_for_fit]-model_flux_fn(sp[:,0][mask_for_fit]))/sp[:,2][mask_for_fit]
    return res#np.concatenate([res1,res2])


def  get_vary_names(params):
    return [name for name in params.keys() if params[name].vary]


def log_likelihood_prev(theta,params,z_n,z_DLA,pl_fe,intr_cont,sp,lam_start,lam_stop,weights,conv=False):
    vary_names = get_vary_names(params)
    theta_dict = dict(zip(vary_names,theta))
    #z_s,IGM_v,sigma_ly,sigma_NV,A_ly,A_NV,x_HI,NH,b,lam_NV = theta

    z_s = theta_dict["z_s"] if "z_s" in theta_dict else params["z_s"].value
    IGM_v = theta_dict["IGM_v"] if "IGM_v" in theta_dict else params["IGM_v"].value
    sigma_ly = theta_dict["sigma_ly"] if "sigma_ly" in theta_dict else params["sigma_ly"].value
    sigma_NV = theta_dict["sigma_NV"] if "sigma_NV" in theta_dict else params["sigma_NV"].value
    A_ly = theta_dict["A_ly"] if "A_ly" in theta_dict else params["A_ly"].value
    A_NV = theta_dict["A_NV"] if "A_NV" in theta_dict else params["A_NV"].value
    x_HI = theta_dict["x_HI"] if "x_HI" in theta_dict else params["x_HI"].value
    NH = theta_dict["NH"] if "NH" in theta_dict else params["NH"].value
    b = theta_dict["b"] if "b" in theta_dict else params["b"].value
    lam_NV = theta_dict["lam_NV"] if "lam_NV" in theta_dict else params["lam_NV"].value

    #    z_s_v = params["z_s_v"].value
    #    z_n = params["z_n"].value
    # z_DLA = params["z_DLA"].value
    #sigma_NV = params["sigma_NV"].value

    z_s_v = z_s - IGM_v
    #sigma_NV = del_NV
    #intrnsic spectra
    lam, intrn_flux_fn = f_intr_mod(z_s, pl_fe, sigma_ly, sigma_NV, A_ly, A_NV,lam_NV,lam_start,lam_stop)
    intrn_flux = intrn_flux_fn(lam)
    #qs_spline = intr_cont(np.log10(lam / (1 + z_s)))
    #mask_intr = (lam > 1218 * (1 + z_s)) & (lam < 1275 * (1 + z_s))
    #res1 = qs_spline[mask_intr] - intrn_flux[mask_intr]

    #adding DLA and IGM absorption
    tau_dla = Voigt(lam, 1215.6736, 1.388E-01, NH, b, 6.265E+08, z=z_DLA)

    tau_IGM = T_IGM_l(lam, x_HI, z_s_v, z_n)
    tau_tot = tau_dla + tau_IGM

    model_flux = intrn_flux*np.exp(-tau_tot)

    if conv == True:
        conv_flux = convolve_res2(lam, model_flux, 2700)
        model_flux_fn = interp1d(lam, conv_flux)
    else:
        model_flux_fn = interp1d(lam,model_flux)

    mask_1 = (sp[:,0]>=lam_start)*(sp[:,0]<=lam_stop)
    mask_for_fit = mask_1*(sp[:,3]==False)
    #res = qs_spline-model_flux #
    res = (sp[:,1][mask_for_fit]-model_flux_fn(sp[:,0][mask_for_fit]))/sp[:,2][mask_for_fit]
    chi2 = np.sum((res**2)*weights)
    return -0.5*chi2

def log_likelihood(theta,params,z_n,z_DLA,sp,pred_conti_fn,lam_start,lam_stop,weights,conv=False):
    vary_names = get_vary_names(params)
    theta_dict = dict(zip(vary_names,theta))
    #z_s,IGM_v,x_HI,NH,b,lam_NV = theta

    z_s = theta_dict["z_s"] if "z_s" in theta_dict else params["z_s"].value
    IGM_v = theta_dict["IGM_v"] if "IGM_v" in theta_dict else params["IGM_v"].value
    x_HI = theta_dict["x_HI"] if "x_HI" in theta_dict else params["x_HI"].value
    NH = theta_dict["NH"] if "NH" in theta_dict else params["NH"].value
    b = theta_dict["b"] if "b" in theta_dict else params["b"].value

    """lam_start and lam_stop are the beginning and ending of the predicted continuum"""
    #    z_s_v = params["z_s_v"].value
    #    z_n = params["z_n"].value
    # z_DLA = params["z_DLA"].value
    #sigma_NV = params["sigma_NV"].value

    z_s_v = z_s - IGM_v

    #intrnsic spectra
    lam = np.linspace(lam_start, lam_stop, 500)
    intrn_flux = pred_conti_fn(lam)

    #adding DLA and IGM absorption
    tau_dla = Voigt(lam, 1215.6736, 1.388E-01, NH, b, 6.265E+08, z=z_DLA)

    tau_IGM = T_IGM_l(lam, x_HI, z_s_v, z_n)
    tau_tot = tau_dla + tau_IGM

    model_flux = intrn_flux*np.exp(-tau_tot)

    if conv == True:
        conv_flux = convolve_res2(lam, model_flux, 2700)
        model_flux_fn = interp1d(lam, conv_flux)
    else:
        model_flux_fn = interp1d(lam,model_flux)

    mask_1 = (sp[:,0]>=lam_start)*(sp[:,0]<=lam_stop)
    mask_for_fit = mask_1*(sp[:,3]==False)
    #res = qs_spline-model_flux #
    res = (sp[:,1][mask_for_fit]-model_flux_fn(sp[:,0][mask_for_fit]))/sp[:,2][mask_for_fit]
    chi2 = np.sum((res**2)*weights)
    return -0.5*chi2

def log_prior(theta,params):

    #theta = np.asarray(theta).ravel()
    #z_s, IGM_v, sigma_ly, sigma_NV, A_ly, A_NV, x_HI,NH,b,lam_NV = [float(x) for x in theta]
    #z_s,IGM_v,sigma_ly,sigma_NV,A_ly,A_NV,x_HI = theta
    vary_names = get_vary_names(params)
    theta_dict = dict(zip(vary_names,theta))
    #print(params.values())
    for name in vary_names:
        if not (params[name].min <= theta_dict[name] <= params[name].max):
            return -np.inf
    return 0.0

"""
def log_likelihood(theta):
#    chi2 = sum((model-flux)/err)
    return -0.5*chi2
"""

def log_posterior(theta,params,z_n,z_DLA,sp,pred_conti_fn,lam_start,lam_stop,weights,conv=False):
    if 0:
        theta_par = Parameters()
        param_names =list(params.keys())
        for i in range(len(param_names)):
            theta_par.add(param_names[i],value=theta[i])

    lp = log_prior(theta,params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp+log_likelihood(theta,params,z_n,z_DLA,sp,pred_conti_fn, lam_start, lam_stop, weights, conv)



#sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg  = ("/home/astro/projects/JWST/damped_lalp/data/UHS1634/slit/s2d/ext1d/tot/tot/spec/new_spec/",6.5188,6.016258253263437e-17,6.2891951,[(9070.6,9078),(9112.8,9115.3),(9134,9146),(9136.8,9140.8)],(8980,9240),[(9056.2,9136.7),(9136.8,9142.0),(9142.32,9147.04),(9219.92,9235.91)],(9000,9165)) #fit reg (9000,9388.9) (8900,9564.8) #inclu (9136.8,9140.8)
#sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg = ("/home/astro/projects/JWST/damped_lalp/data/DELSJ0411-0907/MAST_2025-06-04T2053/JWST/nrs1/stage2/ext1d/tot/tot/new_spec/",6.826,9.22138006861279e-18,6.7295207025,[(9387.094, 9405.493)],(9375,9800),[(9457.2,9495)],(9386,9528)) #(9250,9850)
#sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg = ("/home/astro/projects/JWST/damped_lalp/data/VDESJ0020-3653/MAST_2025-06-04T2057/JWST/nrs1/stage2/ext1d/tot/tot/new_spec/",6.834,4.776462211462217e-18,6.6683793,[(9476.7,9489.6)],(9450,9700),[(9510.4,9555)],(9476,9590)) #exclu (9490.5,9555) ext-fit

#sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg  = ("/home/astro/projects/JWST/damped_lalp/data/UHS1634/slit/s2d/ext1d/tot/tot/spec/new_spec/",6.5188,6.016258253263437e-17,6.2891951,[(9070.6,9078),(9112.8,9115.3),(9134,9146),(9136.8,9140.8)],(8980,9240),[(9056.2,9136.7),(9136.8,9142.0),(9142.32,9147.04),(9219.92,9235.91)],(9000,9165)) #fit reg (9000,9388.9) (8900,9564.8) #inclu (9136.8,9140.8)
#sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg = ("/home/astro/projects/JWST/damped_lalp/data/DELSJ0411-0907/MAST_2025-06-04T2053/JWST/nrs1/stage2/ext1d/tot/tot/new_spec/",6.826,9.22138006861279e-18,6.7295207025,[(9387.094, 9405.493)],(9375,9800),[(9457.2,9495)],(9386,9528)) #(9250,9850)
sp_path,z_s,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg = ("/home/astro/projects/JWST/QSO_ly_a/VDESJ0020-3653/",6.834,4.776462211462217e-18,6.6683793,[(9476.7,9489.6)],(9450,9700),[(9510.4,9555)],(9476,9590)) #exclu (9490.5,9555) ext-fit

filename = "combined_spec_file_modi.dat"

pred_folder_name  = "qsanndra_modi_6.834"
pred_fname = "combined_spec_file_modi_prediction_mean.txt"

with open(os.path.join(sp_path,pred_folder_name,pred_fname)) as f :
    firstline = f.readline().strip()

norm = float(firstline.split("Normalization constant:")[1])

pred_conti = np.loadtxt(os.path.join(sp_path,pred_folder_name,pred_fname),skiprows=1)
pred_conti[:,0] *= 1+z_s   #moving the predicted into observed frame
pred_conti[:,1] *= norm  #making the predicted flux to be of the same units as observed
pred_conti_fn = interp1d(pred_conti[:,0],pred_conti[:,1])

sp = np.loadtxt(sp_path+filename)    # spectra
sp[:,1] = sp[:,1]
sp[:,2] = sp[:,2]

#load QSmooth cont inter and ouliers #
with open(sp_path+"intr_cont_oulier_mask_new_spec.pkl","rb") as f:
    intr_cont,outlier_mask = pickle.load(f)

#print(outlier_mask)
sp = np.column_stack((sp,outlier_mask))
if 0:
    mask = (sp[:, 0] > incl_reg[0]) & (sp[:, 0] < incl_reg[1])
    sp[mask, 3] = False  # mark as not outlier

# Apply exclusion mask
for reg in exclu:
    mask = (sp[:, 0] > reg[0]) & (sp[:, 0] < reg[1])
    sp[mask, 3] = True  # mark as outlier


for reg in incl_reg:
    mask = (sp[:, 0] > reg[0]) & (sp[:, 0] < reg[1])
    sp[mask, 3] = False  # mark as not outlier



outlier_mask = sp[:, 3].astype(bool)  # Ensures boolean type

lam_start = np.min(pred_conti[:,0]) #fit_reg[0] #strating point of the region to fit in observed spectra
lam_stop = np.max(pred_conti[:,0])#fit_reg[1] #stopping point of the region to fit in observed spectra

z_n = 6 #end of reionization redshift

"""------------------------------------------Weight function-------------------------------------------------"""
if 1:
    mask_1 = (sp[:,0] >= lam_start) & (sp[:,0] <= lam_stop)
    mask_for_fit = mask_1 & (sp[:,3] == False)

    mask_fit = (sp[:,0]>fit_reg[0])*(sp[:,0]<fit_reg[1])
    mask_wing = (sp[:,0]>weight_reg[0])*(sp[:,0]<weight_reg[1])

    nblue = np.sum((sp[:,0]<weight_reg[0])*mask_for_fit)
    nred = np.sum((sp[:,0]>weight_reg[1])*mask_for_fit)
    nwing = np.sum(mask_wing)

    weight_value = (nblue+nred)/nwing

    x = sp[:,0][mask_for_fit]
    weights =np.where((x>weight_reg[0])*(x<weight_reg[1]),
                         weight_value,
                         1)
else:
    weights=1.0
"""--------------------------------------------------------------------------------------------------------"""

params = Parameters()
params.add("z_s", value=z_s,min=z_s-0.03,max=z_s+0.03,brute_step=0.01,vary=True)
params.add("IGM_v",value=0.1,min=0,max = 0.4,brute_step=0.2,vary=True) #max = 0.2
params.add("x_HI",value=0.6, min = 0.0, max =1,brute_step=0.5,vary=False)
params.add("NH",value=21, min=18, max=24,brute_step=2,vary=True)
params.add("b",value=25, min = 10, max = 100,brute_step=20,vary=False) # in km/s
params.add("lam_NV",value = 1227, min = 1220,max = 1260,brute_step=10,vary=False)
#params.add("z_s_v",min=z_s_l,max=z_s_u)
#params.add("z_n",z_n,vary=False)
#params.add("z_DLA",z_DLA,vary=False)
#params.add("sigma_NV", min=30)
#params.add("sigma_NV_1", value=38.74, min=10, max=100)
#params.add("sigma_NV_2", value=38.74, min=10, max=100)





"""--------------RUN MCMC---------------------------------------------------------------------------------------------------------"""
""" Steps and walkers to run MCMC"""
nwalkers = 500
nsteps = 500
conv = False

param_names = [name for name in params.keys() if params[name].vary]
print(param_names)
ndim = len(param_names)

#intialize walkers
int_values = np.array([params[i].value for i in param_names])
min_values = np.array([params[i].min for i in param_names])
max_values = np.array([params[i].max for i in param_names])
disp = np.asarray([params[i].brute_step for i in param_names])#(max_values-min_values)/6.0
#p0 = init_p0_bulk(params,nwalkers,max_attempts=100)

p0 = []

if 1:
    for i in range(nwalkers):
        prob = -np.inf
        #print(i)
        while np.isinf(prob):
            wal_pos= int_values+disp*np.random.randn(ndim)
            #print(wal_pos)

            prob = log_posterior(wal_pos,params, z_n, z_DLA, sp, pred_conti_fn, lam_start, lam_stop, weights, conv=False)  #keep the conv always false, no need to convolve in this step
        p0.append(wal_pos)

    #print(p0)
    from multiprocessing import Pool

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,log_posterior,pool=pool,args=(params,z_n,z_DLA,sp,pred_conti_fn,lam_start,lam_stop,weights,conv)) # here set the convolve according to your self
        start = time.time()
        sampler.run_mcmc(p0,nsteps,progress= True)
        end=time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    samples = sampler.chain  #nwalker,nsteps,ndim
    lnprob = sampler.get_log_prob()


    with open(sp_path+"IGM_DLA_qsa_XHI_"+str(params["x_HI"].value)+"_fixed.pkl","wb") as f:
        pickle.dump(sampler,f)

"""------------------------------------------------------LOAD and Read MCMC output---------------------------------------------------------------------------------------------------------------------"""
with open(sp_path+"IGM_DLA_qsa_XHI_"+str(params["x_HI"].value)+"_fixed.pkl","rb") as f:
    sampler=pickle.load(f)

samples = sampler.chain  #nwalker,nsteps,ndim
nwalkers,nsteps,ndim=samples.shape
lnprob = sampler.get_log_prob()

"""-------------------------------------Some checks----------------------------------------------------------------------------------------------"""
acc = sampler.acceptance_fraction
bad_idx = np.where(acc < 0.05)[0]
print("bad walkers:", bad_idx)
print("counts: good", np.sum(acc>=0.05), " bad", len(bad_idx))


"""----------------------------------Walker mean for each step plot-----------------------------------------------------------------------------------------------------------------"""
#print(sampler.acceptance_fraction)
means = np.mean(samples,axis=0) #this give the means over walkers as the shape is (walkers, steps, dim)
median = np.median(samples,axis=0)
stds = np.std(samples,axis=0)

walker_index =10
walker = samples[walker_index,:,:]
step_index = np.arange(nsteps)

if 0:
    #plots
    ncols = 4
    nrows =int(np.ceil(len(param_names)/ncols))
    fig,axes = plt.subplots(nrows,ncols,figsize=(4*ncols,3*nrows),sharex =True)
    axes = axes.flatten()

    for (i,pname) in enumerate(param_names):
        ax = axes[i]
        ax.plot(step_index, means[:, i], linewidth=1.5, label="mean across walkers")
        ax.plot(step_index, walker[:, i], linewidth=1.0, linestyle="--", label=f"walker {walker_index+1}")
        # optional: shaded ±1 sigma around mean
        ax.fill_between(step_index,
                        means[:, i] - stds[:, i],
                        means[:, i] + stds[:, i],
                        alpha=0.15)
        ax.set_xlabel("step")
        ax.set_ylabel(pname)
        ax.legend(fontsize="small", loc="best")
        ax.grid(alpha=0.3)

    for j in range(i+1,len(axes)):
        axes[j].axis("off")
    #nwalker,nsteps = samples[0],samples[1]

    plt.show()

"""-------------------------------------------Estimation of parameters using Chain consumer---------------------------------------------------------------------------------------------------------------"""
# cut the chain at burnin step
burnin = int(nsteps * 0.9)

# analyse the chain using chainconsumer



from chainconsumer import Chain,ChainConsumer,make_sample




chain = Chain.from_emcee(sampler, param_names, "an emcee chain", discard=burnin, thin=2, color="indigo")
consumer = ChainConsumer().add_chain(chain)

"""distribution plots"""
if 0:
    fig = consumer.plotter.plot_walks()
    fig = consumer.plotter.plot()
#plt.show()
summary =consumer.analysis.get_summary()
chain_name = list(summary.keys())[0]
summary_chain = summary[chain_name]

print(summary)

row  = []
for pname,bound in summary_chain.items():
    lo = bound.lower
    med = bound.center
    hi = bound.upper
    if lo == None:
        lo =0
    if med == None:
        med = 0
    if hi == None:
        hi =0
    print(f"{pname:8s}  median={med:.6g}  -{med - lo:.3g}/+{hi - med:.3g}  ( [{lo:.6g}, {hi:.6g}] )")

    params[pname].value = med
    row.append(med)
    row.append(med - lo)
    row.append(hi - med)


"""---------------------------------------------------- Best fit values --------------------------------------------------------------------------------"""

chain = samples[:,-1, :].reshape((-1, ndim))
chain_ln = -lnprob[-1, :].reshape(-1)
chi2 = 2*chain_ln

best_idx = np.argmin(chi2)
best_theta = chain[best_idx,:]
best_chi2 = chi2[best_idx]

best_low = np.argmin(chi2-best_chi2+1)
best_up = np.argmin(chi2-best_chi2-1)
print("best_chi2:", best_chi2)
print("best_lo:", best_low)
print("best_up:", best_up)

for i,name in enumerate(param_names):
    params[name].value = best_theta[i]
    print(f'{name} = {best_theta[i]}')

for i, name in enumerate(param_names):
    p = chain[:, i]

    mask = chi2 < best_chi2 + 1
    lo = p[mask].min()
    hi = p[mask].max()

    print(f"{name} = {best_theta[i]:.5f} -{best_theta[i]-lo:.5f} +{hi-best_theta[i]:.5f}")

z_s = params["z_s"].value
#z_s_v =result.params["z_s_v"].value
z_s_v =z_s - params["IGM_v"].value
sigma_ly = params["sigma_ly"].value
#sigma_NV = result.params["sigma_NV"].value
sigma_NV = params["sigma_NV"].value
A_ly = params["A_ly"].value
A_NV = params["A_NV"].value
x_HI = params["x_HI"].value
NH = params["NH"].value
b = params["b"].value
lam_NV = params["lam_NV"].value



"""-----------------------------------------------------------------model for plot ------------------------------------------------------------------------------------------"""
lam_start = fit_reg[0]
lam_stop = fit_reg[1]

lam_gauss,intrn_fn = f_intr_mod(z_s,pl_fe, sigma_ly,sigma_NV,A_ly,A_NV,lam_NV,lam_start, lam_stop)
intrn = intrn_fn(lam_gauss)
tau_dla = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH,b,6.265E+08,z=z_DLA)
tau_IGM = T_IGM_l(lam_gauss,x_HI,z_s_v,z_n)#
tau_tot = tau_IGM+tau_dla


lam_spg = sp[:,0][(sp[:,0]>lam_gauss[0])*(sp[:,0]<lam_gauss[-1])]
print("length of fit region:",len(lam_spg))
unconv_flux = intrn*np.exp(-tau_tot)
conv_flux = convolve_res2(lam_gauss,unconv_flux,2700)
conv_flux_int = interp1d(lam_gauss,conv_flux)
conv_flux_spg = conv_flux_int(lam_spg)

"""------------------------CHI-sq for best parameters---------------------"""
if 1:


    N_free_parameters = ndim
    dof = np.sum(mask_for_fit) - N_free_parameters

    model_flux = intrn*np.exp(-tau_tot)

    if conv == True:
        conv_flux = convolve_res2(lam_gauss, model_flux, 2700)
        model_flux_fn = interp1d(lam_gauss, conv_flux)
    else:
        model_flux_fn = interp1d(lam_gauss, model_flux)
    mask_1 = (sp[:, 0] >= lam_start) * (sp[:, 0] <= lam_stop)
    mask_for_fit = mask_1 * (sp[:, 3] == False)
    # res = qs_spline-model_flux #
    res = (sp[:, 1][mask_for_fit] - model_flux_fn(sp[:, 0][mask_for_fit])) / sp[:, 2][mask_for_fit]
    chi2 = np.sum((res ** 2) * weights)
    redchi2 = chi2/dof
    print("chi2 = ",chi2)
    print("redchi2 = ", redchi2)

    row.append(chi2)
    row.append(redchi2)
    df = pd.DataFrame([row])
    df.to_excel("parms.xlsx",index=False)
"""------------------------------------------------------------best fit plot----------------------------------------------------------------------------------------"""

lam_start = fit_reg[0]
lam_stop = fit_reg[1]

lam_gauss,intrn_fn = f_intr_mod(z_s,pl_fe, sigma_ly,sigma_NV,A_ly,A_NV,lam_NV,lam_start, lam_stop)
intrn = intrn_fn(lam_gauss)
tau_dla = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH,b,6.265E+08,z=z_DLA)
tau_IGM = T_IGM_l(lam_gauss,x_HI,z_s_v,z_n)#
tau_tot = tau_IGM+tau_dla



if 1:
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.plot(sp[:, 0], sp[:, 1], color="black", label="spectra")
    ax1.scatter(sp[:, 0][~outlier_mask], sp[:, 1][~outlier_mask], color="orange")
    # ax1.plot(lam_gauss,intr_cont(np.log10(lam_gauss/(1+z_s))),color="red", label="QSmooth")
    ax1.plot(lam_gauss, intrn * np.exp(-tau_tot), c="green", label="IGM+DLA")
    ax1.plot(lam_gauss, intrn * np.exp(-tau_IGM), c="brown", label="IGM")
    # ax1.plot(lam_gauss,intrn*np.exp(-tau_dla),c="blue",label="DLA")
    ax1.plot(lam_gauss, intrn, color="cyan", label="intrinsic")
    ax1.vlines((1215.6736 * (1 + z_s),lam_NV* (1 + z_s)), 0, 10)
    ax1.set_xlim(9000, 10000)
    ax1.set_title(f"z_the {z:.3f}, z_pred {z_s:.3f}, IGM_z {z_s_v:.3f}")
    ax1.legend()

    ax2.plot(sp[:, 0], sp[:, 1], color="black", label="spectra")
    # ax2.plot(lam_gauss,intr_cont(np.log10(lam_gauss/(1+z_s))),color="red", label="QSmooth")
    ax2.plot(lam_gauss, intrn, color="cyan", label="intrinsic")
    ax2.plot(lam_gauss, intrn * np.exp(-tau_dla), c="blue", label="DLA")
    ax2.plot(lam_gauss, intrn * np.exp(-tau_IGM), c="brown", label="IGM")
    ax2.set_xlim(8500, 10000)
    ax2.legend()

    ax1.set_ylabel("Normalised flux")
    ax2.set_ylabel("Normalised flux")
    ax2.set_xlabel("Wavelength")
"""-----------------------------------plot for proposal----------------------------------------------------------------"""
if 0 : # manual inputs to plotting model
    z_s =6.834
    #IGM_v = 0.03
    #z_s_v = z_s - IGM_v
    sigma_ly =69.1
    sigma_NV =69.7
    A_ly = 3.61
    A_NV = 1.62
    x_HI = 0.55
    NH =21.9
    b = 39.63
    lam_NV =1238.32

lam_start = 9200
lam_stop = 10300#fit_reg[1]

lam_gauss,intrn_fn = f_intr_mod(z_s,pl_fe, sigma_ly,sigma_NV,A_ly,A_NV,lam_NV,lam_start, lam_stop)
intrn = intrn_fn(lam_gauss)
tau_dla = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH,b,6.265E+08,z=z_DLA)
tau_IGM = T_IGM_l(lam_gauss,x_HI,z_s_v,z_n)#
tau_tot = tau_IGM+tau_dla

tau_dla_p = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH+0.1,b,6.265E+08,z=z_DLA)
tau_dla_n = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH-0.2,b,6.265E+08,z=z_DLA)
NH_err_plus = 0.1
NH_err_minus = 0.2
if 1: # to plot or not
    if 0: # additional gauss
        g =0.7*np.exp(-(lam_gauss - 9730) ** 2 / (2 * 35 ** 2))
        g_1 =0.5*np.exp(-(lam_gauss - 9860) ** 2 / (2 * 65 ** 2))
        g_2 = 0.2 * np.exp(-(lam_gauss - 9905) ** 2 / (2 * 50 ** 2))
        g_3 = 0.1 * np.exp(-(lam_gauss - 9898) ** 2 / (2 * 5 ** 2))
        g_4 = 0.1 * np.exp(-(lam_gauss - 9929) ** 2 / (2 * 5 ** 2))
        intrn = intrn+g+g_1+g_2+g_3 +g_4
        intrn_fn = interp1d(lam_gauss,intrn)

    mask = (sp[:,0]>lam_start) *(sp[:,0]<lam_stop)
    sp_p = sp[mask]
    sp_p[:,1]/=intrn_fn(sp_p[:,0])
    # === Figure setup (2:1 aspect ratio) ===
    fig, ax1 = plt.subplots(figsize=(12, 6))  # width:height = 2:1

    # === Main plots ===
    ax1.step(sp_p[:, 0], sp_p[:, 1], color="black", lw=3, alpha=1, label="Observed Spectrum")
    # ax1.scatter(sp[:, 0][~outlier_mask], sp[:, 1][~outlier_mask],
    #           color="orange", s=20, label="fitting points")
    ax1.plot(lam_gauss, intrn/intrn, color="red", lw=3, label="continuum", ls="--")
    ax1.plot(lam_gauss, intrn/intrn * np.exp(-tau_tot), color="red", lw=3, label = (
    r"$\mathbf{DLA\ (N_{HI} = "
    rf"{NH:0.1f}^{{+{NH_err_plus:0.1f}}}_{{-{NH_err_minus:0.1f}}}"
    r")\ +\ IGM}$"
)
)
    ax1.plot(lam_gauss, intrn/intrn * np.exp(-tau_IGM), color="blue", lw=3, label = rf"IGM only $\mathbf{{X_{{HI}} = {x_HI}}}$")
    #ax1.plot(lam_gauss, intrn/intrn * np.exp(-tau_dla), color="orange", lw=1.5, label=f"DLA NHI ={NH:0.1f}")

    ax1.fill_between(lam_gauss,np.exp(-tau_IGM-tau_dla_p),np.exp(-tau_IGM-tau_dla_n),color="red",alpha=0.2)
    #ax1.fill_between(lam_gauss, np.exp(- tau_dla_p), np.exp(- tau_dla_n), color="orange", alpha=0.2)
    # === DLA vertical marker ===
    lya_dla = 1215.67 * (1 + z_DLA)
    ax1.axvline(lya_dla, color="black", lw=3, ls="--")
    #ax1.text(lya_dla + 5, 0.5, r"Ly$\mathbf{\alpha}$ at z$_{\mathbf{DLA}}$",
    #         rotation=90, va="bottom", ha="left", color="black", fontsize=12, fontweight="bold")
    if 0:
        for line, name in zip(
                (1215.6736, 1238.821, 1242.8),
                (r"QSO Ly$\mathbf{\alpha}$", "NV λ1238", "NV λ1242"),
        ):
            wl = line * (1 + z_s)
            ax1.axvline(wl, color="gray", ls="--", lw=0.9)
            ax1.text(wl + 4, ax1.get_ylim()[1] * 0.7, name, rotation=90,
                     va="top", ha="left", fontsize=10, color="black", fontweight="bold")
    if 0:
        line=1215.6736
        name =r"QSO Ly$\mathbf{\alpha}$"
        wl = line * (1 + z_s)
        ax1.axvline(wl, color="gray", ls="--", lw=0.9)
        #ax1.text(wl + 4, ax1.get_ylim()[1] * 0.7, name, rotation=90,
        #             va="top", ha="left", fontsize=10, color="black", fontweight="bold")
    # === Axis labels, limits, ticks ===
    ax1.set_xlim(9300, 9680)
    ax1.set_ylim(-0.6, 1.5),
    #ax1.set_ylabel(r"$\mathbf{Normalized\ Flux\ }$", #(arbitary\ units)
    #               fontsize=15)

    #ax1.set_xlabel("Observed Wavelength [Å]", fontsize=15, fontweight="bold")

    # Inward ticks on all sides
    ax1.tick_params(direction="in", length=6, width=1.1, top=True, bottom=True, right=True)

    # === Legend ===
    handles, labels = ax1.get_legend_handles_labels()
    leg1 = ax1.legend(handles[:2],labels[:2],loc="upper right",fontsize=20)
    for line in leg1.get_lines():
        line.set_linewidth(4.5)

    for text in leg1.get_texts():
        text.set_fontweight("bold")
    leg1.get_frame().set_alpha(0)
    ax1.add_artist(leg1)

    leg2 =  ax1.legend(handles[2:],labels[2:],loc= "lower right", fontsize=22)
    for line in leg2.get_lines():
        line.set_linewidth(4.5)
    for text in leg2.get_texts():
        text.set_fontweight("bold")
    leg2.get_frame().set_alpha(0)
    ax1.add_artist(leg2)

    if 0:  #single legend
        leg = ax1.legend(ncol=2, frameon=False, loc="upper right", fontsize=18)

        for text in leg1.get_texts():
            text.set_fontweight("bold")

    # === Title ===
    # ax1.set_title(f"z_DLA = {z_DLA:.3f}", loc="center", pad=10)

    # === Secondary top axis for rest-frame wavelengths ===
    secax = ax1.secondary_xaxis(
        'top',
        functions=(lambda x: x / (1 + z_DLA), lambda x: x * (1 + z_DLA))
    )
    if 0:
        secax.set_xlabel(
            rf"$\bf{{Restframe\ Wavelength\ [\AA]\ at\ }}z_{{\mathrm{{DLA}}}} = {z_DLA:.1f}$",
            fontsize=15
        )
    bottom_ticks = ax1.get_xticks()

    # Convert to rest-frame
    rest_ticks = bottom_ticks / (1 + z_DLA)
    rest_ticks = [1215,1225, 1235, 1245,1255]
    secax.xaxis.set_major_locator(FixedLocator(rest_ticks))
    secax.set_xticklabels([f"{t:.0f}" for t in rest_ticks])

    #secax.xaxis.set_major_locator(MultipleLocator(5))
    secax.tick_params(direction="in", length=6, width=1.1, labelsize=26.5)
    for tick in secax.get_xticklabels():
        tick.set_fontweight("bold")

    ax1.tick_params(direction="in", length=6, width=1.1, top=False, bottom=True, right=True, labelsize=26.5)
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontweight("bold")

    # === Final layout ===
    plt.tight_layout()
    plt.savefig("/home/astro/vdes_damping wing_new_model_norm_tot.png",dpi=600)
    plt.show()

"""----------------------------------non normalised proposal plot----------------------------------------"""
if 0:

    if 0: # additional gauss
        g =0.7*np.exp(-(lam_gauss - 9730) ** 2 / (2 * 35 ** 2))
        g_1 =0.5*np.exp(-(lam_gauss - 9860) ** 2 / (2 * 65 ** 2))
        g_2 = 0.2 * np.exp(-(lam_gauss - 9905) ** 2 / (2 * 50 ** 2))
        g_3 = 0.1 * np.exp(-(lam_gauss - 9898) ** 2 / (2 * 5 ** 2))
        g_4 = 0.1 * np.exp(-(lam_gauss - 9929) ** 2 / (2 * 5 ** 2))
        g_manu = 0.08*np.exp(-(lam_gauss - 9644) ** 2 / (2 * 25 ** 2))
        intrn = intrn+g_manu#+g+g_1+g_2+g_3 +g_4

    mask = (sp[:,0]>lam_start) *(sp[:,0]<lam_stop)
    sp_p = sp[mask]
    #sp_p[:,1]/=intrn_fn(sp_p[:,0])
    # === Figure setup (2:1 aspect ratio) ===
    fig, ax1 = plt.subplots(figsize=(12, 6))  # width:height = 2:1

    # === Main plots ===
    ax1.step(sp_p[:, 0], sp_p[:, 1], color="black", lw=1.5, alpha=1, label="Observed Spectrum")
    # ax1.scatter(sp[:, 0][~outlier_mask], sp[:, 1][~outlier_mask],
    #           color="orange", s=20, label="fitting points")
    ax1.plot(lam_gauss, intrn, color="red", lw=1.5, label="intrn continuum", ls="--")
    ax1.plot(lam_gauss, intrn * np.exp(-tau_tot), color="red", lw=1.5, label = (
    r"$\mathbf{DLA\ (NHI = "
    rf"{NH:0.1f}^{{+{NH_err_plus:0.1f}}}_{{-{NH_err_minus:0.1f}}}"
    r")\ +\ IGM}$"
)
)
    ax1.plot(lam_gauss, intrn * np.exp(-tau_IGM), color="blue", lw=1.5, label=f"IGM only x_HI ={x_HI}")
    ax1.plot(lam_gauss, intrn * np.exp(-tau_dla), color="orange", lw=1.5, label=f"DLA NHI ={NH:0.1f}")

    ax1.fill_between(lam_gauss,intrn*np.exp(-tau_IGM-tau_dla_p),intrn*np.exp(-tau_IGM-tau_dla_n),color="red",alpha=0.2)
    #ax1.fill_between(lam_gauss, np.exp(- tau_dla_p), np.exp(- tau_dla_n), color="orange", alpha=0.2)
    # === DLA vertical marker ===
    lya_dla = 1215.67 * (1 + z_DLA)
    ax1.axvline(lya_dla, color="black", lw=3, ls="--")
    ax1.text(lya_dla + 5, 0.5, r"Ly$\mathbf{\alpha}$ at z$_{\mathbf{DLA}}$",
             rotation=90, va="bottom", ha="left", color="black", fontsize=12, fontweight="bold")
    if 0:
        for line, name in zip(
                (1215.6736, 1238.821, 1242.8),
                (r"QSO Ly$\mathbf{\alpha}$", "NV λ1238", "NV λ1242"),
        ):
            wl = line * (1 + z_s)
            ax1.axvline(wl, color="gray", ls="--", lw=0.9)
            ax1.text(wl + 4, ax1.get_ylim()[1] * 0.7, name, rotation=90,
                     va="top", ha="left", fontsize=10, color="black", fontweight="bold")
    if 1:
        line=1215.6736
        name =r"QSO Ly$\mathbf{\alpha}$"
        wl = line * (1 + z_s)
        ax1.axvline(wl, color="gray", ls="--", lw=0.9)
        #ax1.text(wl + 4, ax1.get_ylim()[1] * 0.7, name, rotation=90,
        #             va="top", ha="left", fontsize=10, color="black", fontweight="bold")
    # === Axis labels, limits, ticks ===
    ax1.set_xlim(9250, 9800)
    #ax1.set_ylim(-0.25, 1.5),
    ax1.set_ylabel(r"$\mathbf{Normalized\ Flux\ }$", #(arbitary\ units)
                   fontsize=15)

    ax1.set_xlabel("Observed Wavelength [Å]", fontsize=15, fontweight="bold")

    # Inward ticks on all sides
    ax1.tick_params(direction="in", length=6, width=1.1, top=True, bottom=True, right=True)

    # === Legend ===
    leg = ax1.legend(ncol=2, frameon=False, loc="upper right", fontsize=10)

    for text in leg.get_texts():
        text.set_fontweight("bold")

    # === Title ===
    # ax1.set_title(f"z_DLA = {z_DLA:.3f}", loc="center", pad=10)

    # === Secondary top axis for rest-frame wavelengths ===
    secax = ax1.secondary_xaxis(
        'top',
        functions=(lambda x: x / (1 + z_DLA), lambda x: x * (1 + z_DLA))
    )

    secax.set_xlabel(
        rf"$\bf{{Restframe\ Wavelength\ [\AA]\ at\ }}z_{{\mathrm{{DLA}}}} = {z_DLA:.1f}$",
        fontsize=15
    )

    secax.tick_params(direction="in", length=6, width=1.1, labelsize=15)
    for tick in secax.get_xticklabels():
        tick.set_fontweight("bold")

    ax1.tick_params(direction="in", length=6, width=1.1, top=True, bottom=True, right=True, labelsize=15)
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontweight("bold")

    # === Final layout ===
    plt.tight_layout()
    plt.savefig("/home/astro/vdes_damping wing_new_model_non_norm_tot_manu.png",dpi=600)
    plt.show()

"""---------------------pl_fe vs spec plot----------------------------------"""
if 0 :
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(sp[:, 0], sp[:, 1], color="black", lw=3, label="Spectrum")


"""---------------------------------------------------------QSANNDRA vs this code continuum comparison---------------------------------------"""
if 1:
    qsanndra_conti = np.loadtxt(sp_path+"qsanndra/combined_spec_file_prediction_mean.txt",skiprows=1)
    qsanndra_conti[:,0] = (1+z)*qsanndra_conti[:,0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(sp[:, 0], sp[:, 1], color="black", label="spectra")
    ax1.scatter(sp[:, 0][~outlier_mask], sp[:, 1][~outlier_mask], color="orange")
    ax1.plot(lam_gauss, intrn, color="cyan", label="model intrinsic")
    ax1.plot(qsanndra_conti[:,0],qsanndra_conti[:,1],color="red", label="qsanndra")

    ax1.set_xlabel("Wavelength")
    ax1.set_ylabel("Normalised flux")

    #ax1.vlines((1215.6736 * (1 + z_s),lam_NV* (1 + z_s)), 0, 10)
    ax1.set_xlim(9000, 10000)
    ax1.legend()
    #plt.show()

"""------------------------------------------------------------------------------------ Params vs chi2-----------------------------------------------------------------------"""
if 1:
    samples = sampler.chain
    burnin = int(nsteps * 0.7)
    chain = samples[:, -1, :].reshape((-1, ndim))
    chain_ln = -lnprob[-1, :].reshape(-1)
    chi2 = 2 * chain_ln

    # plots
    ncols = 4
    nrows = int(np.ceil(len(param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for (i, pname) in enumerate(param_names):
        ax = axes[i]
        ax.scatter(chain[:,i],chi2,s=2,alpha=0.3,c='r',label=r'$\chi^2$')
        ax.set_ylabel(r'$\chi^2$')
        ax.set_xlabel(pname)
        ax.legend(fontsize="small", loc="best")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
"""---------------------------------------------------------LAST step walkers model plot-------------------------------------------------------------------------"""
from matplotlib.collections import LineCollection

if 1 :
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(sp[:, 0], sp[:, 1], color="black", label="spectra")
    ax.scatter(sp[:, 0][~outlier_mask], sp[:, 1][~outlier_mask], color="orange")
    mask_1 = (sp[:,0]>=lam_start)*(sp[:,0]<=lam_stop)
    mask_for_fit = mask_1*(sp[:,3]==False)

    last_step = sampler.chain[:,-1,:]
    lines=[]
    intr_lines = []
    residuals = []
    vary_names = get_vary_names(params)
    for w in range(nwalkers):
        #print(w)
        theta = last_step[w]

        theta_dict = dict(zip(vary_names, theta))
        # z_s,IGM_v,sigma_ly,sigma_NV,A_ly,A_NV,x_HI,NH,b,lam_NV = theta

        z_s = theta_dict["z_s"] if "z_s" in theta_dict else params["z_s"].value
        IGM_v = theta_dict["IGM_v"] if "IGM_v" in theta_dict else params["IGM_v"].value
        sigma_ly = theta_dict["sigma_ly"] if "sigma_ly" in theta_dict else params["sigma_ly"].value
        sigma_NV = theta_dict["sigma_NV"] if "sigma_NV" in theta_dict else params["sigma_NV"].value
        A_ly = theta_dict["A_ly"] if "A_ly" in theta_dict else params["A_ly"].value
        A_NV = theta_dict["A_NV"] if "A_NV" in theta_dict else params["A_NV"].value
        x_HI = theta_dict["x_HI"] if "x_HI" in theta_dict else params["x_HI"].value
        NH = theta_dict["NH"] if "NH" in theta_dict else params["NH"].value
        b = theta_dict["b"] if "b" in theta_dict else params["b"].value
        lam_NV = theta_dict["lam_NV"] if "lam_NV" in theta_dict else params["lam_NV"].value


        z_s_v = z_s - IGM_v

        lam_gauss, intrn_fn = f_intr_mod(z_s, pl_fe, sigma_ly, sigma_NV, A_ly, A_NV,lam_NV, lam_start, lam_stop)
        intrn = intrn_fn(lam_gauss)
        tau_dla = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH,b,6.265E+08,z=z_DLA)
        tau_IGM = T_IGM_l(lam_gauss, x_HI, z_s_v, z_n)  #
        tau_tot = tau_IGM +tau_dla
        model_line = intrn * np.exp(-tau_tot)
        line = np.column_stack((lam_gauss,model_line)) #model fit line
        lines.append(line) #model fit lines array

        intr_line = np.column_stack((lam_gauss,intrn))
        intr_lines.append(intr_line)

        model_flux_fn = interp1d(lam_gauss, model_line)
        res = (sp[:, 1][mask_for_fit] - model_flux_fn(sp[:, 0][mask_for_fit])) / sp[:, 2][mask_for_fit]
        r = np.column_stack((sp[:,0][mask_for_fit],res))
        residuals.append(r)  #creating residual array

    lc_m = LineCollection(lines, linewidths=0.6, alpha=0.5, colors="r")
    lc_intr = LineCollection(intr_lines, linewidths=0.6, alpha=0.5, colors="b")

    ax.add_collection(lc_m)
    ax.add_collection(lc_intr)
    #ax.plot(lam_gauss, intrn * np.exp(-tau_IGM), c="brown", label="IGM")
    #ax.plot(lam_gauss, intrn, color="cyan", label="intrinsic")
    z_s = params["z_s"].value
    #ax1.vlines((1215.6736 * (1 + z_s),lam_NV* (1 + z_s)), 0, 10)
    ax.set_xlim(9000, 10000)
    ax.set_title(f"last step walkers models")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel(r"$Flux$")
    ax.legend()

    res_flatten = np.vstack(residuals)
    lc_r = LineCollection(residuals, linewidth=0.6, alpha=0.5, colors="b")
    ax2 = fig.add_subplot(2,1,2,sharex=ax)
    #ax2.add_collection(lc_r)
    ax2.scatter(res_flatten[:,0],res_flatten[:,1],color="blue",marker="x", label="residuals")
    ax2.set_xlabel("wavelength")
    ax2.set_ylabel("residual ((flux-model)/err)")

#plt.show()



"""------------------------------------------------------------Last step multiprocess plot (not correct)--------------------------------------------"""
if 0:
    import numpy as np
    from multiprocessing import Pool
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    # config
    nworkers = 15  # number of processes; tune to CPU cores
    plot_top_k = nwalkers  # how many walkers to plot
    plot_range = (9000, 10000)  # plotting x-limits
    alpha = 0.08

    # prepare last-step params and indices (top by final lnprob)
    chain = sampler.get_chain()  # (nwalkers, nsteps, ndim)
    last_step = chain[:, -1, :]
    lp = sampler.get_log_prob()[:, -1]
    idx_sorted = np.argsort(lp)[::-1]
    use_idx = idx_sorted[:plot_top_k]

    # shared args for model evaluation
    shared_args = (z_s, pl_fe, lam_start, lam_stop, z_n)


    def compute_model_for_theta(theta):
        # unpack theta
        z_s,IGM_v, sigma_ly, sigma_NV, A_ly, A_NV, x_HI,NH,b = theta
        z_s_v = z_s - IGM_v

        # model eval on a shared grid (maybe coarse)
        # choose lam_grid once globally if possible — but we'll eval f_intr_mod which returns lam
        lam_g, intrn_fn = f_intr_mod(z_s, pl_fe, sigma_ly, sigma_NV, A_ly, A_NV, lam_start, lam_stop)
        intrn = intrn_fn(lam_g)
        tau_IGM_p = T_IGM_l(lam_g, x_HI, z_s_v, z_n)
        tau_dla_p = Voigt(lam_g, 1215.6736, 1.388E-01, NH, b, 6.265E+08, z=z_DLA)
        tau_tot_p = tau_IGM_p+tau_dla_p
        model_p = intrn * np.exp(-tau_tot_p)

        # restrict to plotting region
        mask = (lam_g >= plot_range[0]) & (lam_g <= plot_range[1])
        return np.column_stack((lam_g[mask], model_p[mask]))


    # build theta list for selected walkers
    theta_list = [last_step[i] for i in use_idx]

    # compute in parallel
    with Pool(processes=nworkers) as pool:
        lines = pool.map(compute_model_for_theta, theta_list)

    # plot with LineCollection
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # add models
    lc = LineCollection(lines, linewidths=0.6, alpha=alpha, colors="C1")
    ax.add_collection(lc)
    # plot data points on top
    ax.plot(sp[:, 0], sp[:, 1], color="k", linewidth=0.8)
    ax.scatter(sp[:, 0][~outlier_mask], sp[:, 1][~outlier_mask], color="orange", s=6)
    ax.set_xlim(*plot_range)
    ax.autoscale_view()



if 0:
    rows = []
    for name in result.var_names:
        samples = result.flatchain[name]
        q16, q50, q84 = np.percentile(samples, [16, 50, 84])
        err_lo, err_hi = q50 - q16, q84 - q50
        rows.append(f"{name} & {q50:.3f}$^{{+{err_hi:.3f}}}_{{-{err_lo:.3f}}}$ \\\\")

    latex_table = (
            "\\begin{tabular}{lc}\n"
            "Parameter & Value \\\\\n"
            "\\hline\n"
            + "\n".join(rows) +
            "\n\\end{tabular}"
    )

    # --- Print LaTeX script ---
    print(latex_table)