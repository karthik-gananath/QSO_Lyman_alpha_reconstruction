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

from sympy.abc import alpha

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
    lam = np.linspace(lam_start, lam_stop, 5000)
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

def log_likelihood(theta,params,z_n,pl_fe,intr_cont,sp,lam_start,lam_stop,weights,conv=False):
    vary_names = get_vary_names(params)
    theta_dict = dict(zip(vary_names, theta))
    #z_s,IGM_v,sigma_ly,sigma_NV,A_ly,A_NV,x_HI,lam_NV = theta

    z_s = theta_dict["z_s"] if "z_s" in theta_dict else params["z_s"].value
    IGM_v = theta_dict["IGM_v"] if "IGM_v" in theta_dict else params["IGM_v"].value
    sigma_ly = theta_dict["sigma_ly"] if "sigma_ly" in theta_dict else params["sigma_ly"].value
    sigma_NV = theta_dict["sigma_NV"] if "sigma_NV" in theta_dict else params["sigma_NV"].value
    A_ly = theta_dict["A_ly"] if "A_ly" in theta_dict else params["A_ly"].value
    A_NV = theta_dict["A_NV"] if "A_NV" in theta_dict else params["A_NV"].value
    x_HI = theta_dict["x_HI"] if "x_HI" in theta_dict else params["x_HI"].value
    lam_NV = theta_dict["lam_NV"] if "lam_NV" in theta_dict else params["lam_NV"].value
    #NH = params["NH"].value
    #b = params["b"].value
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
    #tau_dla = Voigt(lam, 1215.6736, 1.388E-01, NH, b, 6.265E+08, z=z_DLA)

    tau_IGM = T_IGM_l(lam, x_HI, z_s_v, z_n)
    tau_tot = tau_IGM

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

def log_posterior(theta,params,z_n,pl_fe,intr_cont,sp,lam_start,lam_stop,weights,conv=False):
    if 0:
        theta_par = Parameters()
        param_names =list(params.keys())
        for i in range(len(param_names)):
            theta_par.add(param_names[i],value=theta[i])

    lp = log_prior(theta,params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp+log_likelihood(theta,params,z_n,pl_fe,intr_cont,sp,lam_start,lam_stop,weights,conv)



#sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg  = ("/home/astro/projects/JWST/damped_lalp/data/UHS1634/slit/s2d/ext1d/tot/tot/spec/new_spec/",6.5188,6.016258253263437e-17,6.2891951,(9136.8,9140.8),(8980,9400),[(9056.2,9136.7),(9136.8,9142.0),(9142.32,9147.04),(9219.92,9235.91)],(9015,9165)) #fit reg (9000,9388.9) (8900,9564.8) #inclu (9136.8,9140.8)
sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg = ("/home/astro/projects/JWST/damped_lalp/data/DELSJ0411-0907/MAST_2025-06-04T2053/JWST/nrs1/stage2/ext1d/tot/tot/new_spec/",6.826,9.22138006861279e-18,6.7295207025,(9387.094, 9405.493),(9375,9800),[(9457.2,9495)],(9386,9528)) #(9250,9850)
#sp_path,z,norm,z_DLA,incl_reg,fit_reg,exclu,weight_reg = ("/home/astro/projects/JWST/damped_lalp/data/VDESJ0020-3653/MAST_2025-06-04T2057/JWST/nrs1/stage2/ext1d/tot/tot/new_spec/",6.834,4.776462211462217e-18,6.6683793,(9476.7,9489.6),(9200,9790),[(9510.4,9555)],(9476,9590)) #exclu (9490.5,9555) ext-fit
filename = "combined_spec_file.dat"
sp = np.loadtxt(sp_path+filename)    # spectra
sp[:,1] = sp[:,1]/norm
sp[:,2] = sp[:,2]/norm
#load QSmooth cont inter and ouliers #
with open(sp_path+"intr_cont_oulier_mask_new_spec.pkl","rb") as f:
    intr_cont,outlier_mask = pickle.load(f)

#print(outlier_mask)
sp = np.column_stack((sp,outlier_mask))
if incl_reg:
    mask = (sp[:, 0] > incl_reg[0]) & (sp[:, 0] < incl_reg[1])
    sp[mask, 3] = False  # mark as not outlier

# Apply exclusion mask
for reg in exclu:
    mask = (sp[:, 0] > reg[0]) & (sp[:, 0] < reg[1])
    sp[mask, 3] = True  # mark as outlier

outlier_mask = sp[:, 3].astype(bool)  # Ensures boolean type

lam_start = fit_reg[0] #strating point of the region to fit in observed spectra
lam_stop = fit_reg[1] #stopping point of the region to fit in observed spectra

z_s = z
z_n = 6

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

pl_fe = np.loadtxt(sp_path+"pl_fe_mod_n.txt")   # powerlaw + fe model
pl_fe[:,0] = pl_fe[:,0]#*(1+z)  #wavelength should be redshifted as the PL_FE is in restframe
pl_fe_intr = interp1d(pl_fe[:,0],pl_fe[:,1])

params = Parameters()
params.add("z_s", value=z_s,min=z_s-0.05,max=z_s+0.05,brute_step=0.01,vary=True)
params.add("IGM_v",value=0.03,min=0,max = 0.1,brute_step=0.03,vary=True) #max = 0.2
params.add("sigma_ly",value=40, min=1,max=80,brute_step=20,vary=True)
params.add("sigma_NV", value=20, min =1,max =90,brute_step=10,vary=True )
params.add("A_ly",value=4, min = 0.0,max = 20,brute_step=4,vary=True)
params.add("A_NV",value=2, min= 0.0,max=20,brute_step=2,vary=True)
params.add("x_HI",value=0.5, min = 0.0, max =1,brute_step=0.5,vary=True)
params.add("lam_NV",value = 1240, min = 1220,max = 1260,brute_step=10,vary=True)
#params.add("z_s_v",min=z_s_l,max=z_s_u)
#params.add("z_n",z_n,vary=False)
#params.add("z_DLA",z_DLA,vary=False)
#params.add("sigma_NV", min=30)
#params.add("sigma_NV_1", value=38.74, min=10, max=100)
#params.add("sigma_NV_2", value=38.74, min=10, max=100)




"""--------------RUN MCMC---------------------------------------------------------------------------------------------------------"""
""" Steps and walkers to run MCMC"""
nwalkers = 500
nsteps = 1000
conv =False

param_names = [name for name in params.keys() if params[name].vary]
print(param_names)
ndim = len(param_names)


#initialize walkers
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

            prob = log_posterior(wal_pos,params,z_n,pl_fe,intr_cont,sp,lam_start,lam_stop,weights,conv=False)  #keep the conv always false, no need to convolve in this step
        p0.append(wal_pos)

    #print(p0)
    from multiprocessing import Pool

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,log_posterior,pool=pool,args=(params,z_n,pl_fe,intr_cont,sp,lam_start,lam_stop,weights,conv),a=5.0) # here set the convolve according to your self
        start = time.time()
        sampler.run_mcmc(p0,nsteps,progress= True)
        end=time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    samples = sampler.chain  #nwalker,nsteps,ndim
    lnprob = sampler.get_log_prob()


    with open(sp_path+"mcmc_emcee_new_NV_no_conv.pkl","wb") as f:
        pickle.dump(sampler,f)

"""------------------------------------------------------LOAD and Read MCMC output---------------------------------------------------------------------------------------------------------------------"""
with open(sp_path+"mcmc_emcee_new_NV_no_conv.pkl","rb") as f:
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
chain = samples[:, burnin:, :].reshape((-1, ndim))
chain_ln = -lnprob[burnin:, :].reshape(-1)
# analyse the chain using chainconsumer



from chainconsumer import Chain,ChainConsumer,make_sample




cc_chain = Chain.from_emcee(sampler, param_names, "an emcee chain", discard=burnin, thin=2, color="indigo")
consumer = ChainConsumer().add_chain(cc_chain)

"""distribution plots"""
if 1:
    fig = consumer.plotter.plot_walks()
    fig = consumer.plotter.plot()


#plt.show()
summary =consumer.analysis.get_summary()
chain_name = list(summary.keys())[0]
summary_chain = summary[chain_name]

print(summary)


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


z_s = params["z_s"].value
#z_s_v =result.params["z_s_v"].value
z_s_v =z_s - params["IGM_v"].value
sigma_ly = params["sigma_ly"].value
#sigma_NV = result.params["sigma_NV"].value
sigma_NV = params["sigma_NV"].value
A_ly = params["A_ly"].value
A_NV = params["A_NV"].value
x_HI = params["x_HI"].value
lam_NV = params["lam_NV"].value
#NH = result.params["NH"].value
#b = result.params["b"].value

lam_gauss,intrn_fn = f_intr_mod(z_s,pl_fe, sigma_ly,sigma_NV,A_ly,A_NV,lam_NV,lam_start, lam_stop)
intrn = intrn_fn(lam_gauss)
#tau_dla = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH,b,6.265E+08,z=z_DLA)
tau_IGM = T_IGM_l(lam_gauss,x_HI,z_s_v,z_n)#
tau_tot = tau_IGM#+tau_dla


lam_spg = sp[:,0][(sp[:,0]>lam_gauss[0])*(sp[:,0]<lam_gauss[-1])]

unconv_flux = intrn*np.exp(-tau_tot)
conv_flux = convolve_res2(lam_gauss,unconv_flux,2700)
conv_flux_int = interp1d(lam_gauss,conv_flux)
conv_flux_spg = conv_flux_int(lam_spg)

"""------------------------------------------------------------best fit plot----------------------------------------------------------------------------------------"""
if 1:


    fig  = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2,sharex=ax1)

    ax1.plot(sp[:,0],sp[:,1],color="black",label ="spectra")
    ax1.scatter(sp[:, 0][~outlier_mask], sp[:, 1][~outlier_mask], color="orange")
    #ax1.plot(lam_gauss,intr_cont(np.log10(lam_gauss/(1+z_s))),color="red", label="QSmooth")
    #ax1.plot(lam_gauss,intrn*np.exp(-tau_tot),c="green",label ="IGM+DLA")
    ax1.plot(lam_gauss,intrn*np.exp(-tau_IGM),c="brown",label ="IGM")
    #ax1.plot(lam_gauss,intrn*np.exp(-tau_dla),c="blue",label="DLA")
    ax1.plot(lam_gauss,intrn,color ="cyan",label="intrinsic")
    ax1.vlines((1215.6736 * (1 + z_s),lam_NV* (1 + z_s)), 0, 10)
    ax1.set_xlim(9000,10000)
    ax1.set_title(f"IGM des new mod tied z_the {z:.3f}, z_pred {z_s:.3f}, IGM_z {z_s_v:.3f}")
    ax1.legend()



    ax2.plot(sp[:,0],sp[:,1],color="black",label ="spectra")
    #ax2.plot(lam_gauss,intr_cont(np.log10(lam_gauss/(1+z_s))),color="red", label="QSmooth")
    ax2.plot(lam_gauss,intrn,color ="cyan",label="intrinsic")
    #ax2.plot(lam_gauss,np.exp(-tau_dla),c="blue",label="DLA")
    ax2.plot(lam_gauss,np.exp(-tau_IGM),c="brown",label ="IGM")
    ax2.set_xlim(9000,10000)
    ax2.legend()



#plt.plot(lam_gauss,intr_cont(np.log10(lam_gauss/(1+z_s))),color="red", label="QSmooth")

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
        lam_NV = theta_dict["lam_NV"] if "lam_NV" in theta_dict else params["lam_NV"].value

        z_s_v = z_s - IGM_v


        lam_gauss, intrn_fn = f_intr_mod(z_s, pl_fe, sigma_ly, sigma_NV, A_ly, A_NV,lam_NV, lam_start, lam_stop)
        intrn = intrn_fn(lam_gauss)
        # tau_dla = Voigt(lam_gauss,  1215.6736, 1.388E-01,NH,b,6.265E+08,z=z_DLA)
        tau_IGM = T_IGM_l(lam_gauss, x_HI, z_s_v, z_n)  #
        tau_tot = tau_IGM  # +tau_dla
        model_line = intrn * np.exp(-tau_tot)
        line = np.column_stack((lam_gauss,model_line))
        lines.append(line)

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
    ax1.vlines((1215.6736 * (1 + z_s),lam_NV* (1 + z_s)), 0, 10)
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

plt.show()

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
    print(chain.shape)
    last_step = chain[:, -1, :]
    lp = sampler.get_log_prob()[:, -1]
    idx_sorted = np.argsort(lp)[::-1]
    use_idx = idx_sorted[:plot_top_k]

    # shared args for model evaluation
    shared_args = (z_s, pl_fe, lam_start, lam_stop, z_n)


    def compute_model_for_theta(theta):
        # unpack theta
        z_s,IGM_v, sigma_ly, sigma_NV, A_ly, A_NV, x_HI = theta
        z_s_v = z_s - IGM_v

        # model eval on a shared grid (maybe coarse)
        # choose lam_grid once globally if possible — but we'll eval f_intr_mod which returns lam
        lam_g, intrn_fn = f_intr_mod(z_s, pl_fe, sigma_ly, sigma_NV, A_ly, A_NV, lam_start, lam_stop)
        intrn = intrn_fn(lam_g)
        tau = T_IGM_l(lam_g, x_HI, z_s_v, z_n)
        model = intrn * np.exp(-tau)

        # restrict to plotting region
        mask = (lam_g >= plot_range[0]) & (lam_g <= plot_range[1])
        return np.column_stack((lam_g[mask], model[mask]))


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

#plt.show()
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