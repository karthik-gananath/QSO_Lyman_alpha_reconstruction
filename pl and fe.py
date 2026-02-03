import lmfit
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from lmfit import Parameters,minimize
import matplotlib.pyplot as plt
import pickle

# Define the continuum fitting windows (in Å)
continuum_windows = [
    (1280,1290), (1350, 1360), (1445, 1465), (1700, 1705), (2155, 2400)]#,
#    (2480, 2675), (2925, 3500), (4200, 4230), (4435, 4700),
#    (5100, 5535), (6000, 6250), (6800, 7000)
#]

continuum_windows_1 = [
    (1279.7,1283.62), (1286.79,1289.93), (1354.31,1359.37), (2262.9,2292.67), (2363.6,2374.5), (2387.2,2399.7)
]

continuum_windows_2 = [
    (1283.82,1288.65), (1354.06,1359.91), (1455.4,1462.62), (2173.31,2181.54), (2196.53,2201.1), (2223.91,2240.91),
    (2245,2278.97), (2286.29,2290.7), (2294.95,2302.27), (2306.69,2312.5), (2314.51,2344.28), (2356.02,2369.36),
    (2376.31,2381.83), (2391.61,2399.19)
]

continuum_windows_3= [
    (1282.18,1285.27), (1288.98,1289.86), (1351.08,1353.65), (1355.54,1355.85), (1445.24,1453.2), (1457.6,1460.3)]#,
#    (2171.56,2201.67), (2205.97,2237.63), (2243.3,2248.47), (2255.02,2307.12), (2318.53,2349.32), (2352.66,2395.87)
#]

#masking function
def mask_continuum(wave,continuum_windows):
    mask = np.zeros_like(wave, dtype=bool)
    for (low, high) in continuum_windows:
        mask |= (wave >= low) & (wave <= high)
    return mask


# === Load observed spectrum and FeII template ===
#z = 6.826#6.5188#6.826#6.834
#sp_path,z,norm,continuum_windows = ("/home/astro/projects/JWST/damped_lalp/data/UHS1634/slit/s2d/ext1d/tot/tot/spec/new_spec/",6.5188,6.016258253263437e-17,continuum_windows_1)
#sp_path,z,norm,continuum_windows = ("/home/astro/projects/JWST/damped_lalp/data/DELSJ0411-0907/MAST_2025-06-04T2053/JWST/nrs1/stage2/ext1d/tot/tot/new_spec/",6.826,9.22138006861279e-18,continuum_windows_2)
sp_path,z,norm,continuum_windows = ("/home/astro/projects/JWST/damped_lalp/data/VDESJ0020-3653/MAST_2025-06-04T2057/JWST/nrs1/stage2/ext1d/tot/tot/new_spec/",6.834,4.776462211462217e-18,continuum_windows_3)
filename = "combined_spec_file.dat"
sp = np.loadtxt(sp_path+filename)
vw2001 = "/home/astro/projects/JWST/damped_lalp/papers/FeII template/VW2001/Fe_UVtemplt_A.asc"

obs_wave = sp[:,0]/(1+z)
obs_flux = sp[:,1]/norm #np.mean(sp[:,1])
err = sp[:,2]/norm
#obs_wave, obs_flux = np.loadtxt("obs_spectrum.txt", unpack=True)
feii_wave, feii_flux = np.loadtxt(vw2001, unpack=True)
feii_flux = feii_flux/np.mean(feii_flux)


# === Model Components ===

def powerlaw(wave, amp, alpha):
    return amp * (wave) ** alpha

mask = mask_continuum(obs_wave,continuum_windows)
lam_fit = obs_wave[mask]
flux_fit = obs_flux[mask]
err_fit = err[mask]
popt, _ = curve_fit(powerlaw, lam_fit, flux_fit, p0=[0.5, -0.87], maxfev=10000)

# --- Iterative sigma clipping loop ---
for sigma_clip in [3.0, 2.5, 2.0]:
    model = powerlaw(lam_fit, *popt)

    # Compute residuals and sigma
    residuals = (flux_fit - model)/err_fit
    sigma = np.std(residuals)

    # Mask out outliers
    keep = np.abs(residuals) < sigma_clip * sigma
    lam_fit = lam_fit[keep]
    flux_fit = flux_fit[keep]
    err_fit =err_fit[keep]
    # Fit power-law to current set
    popt, _ = curve_fit(powerlaw, lam_fit, flux_fit, p0=[0.5, -0.87], maxfev=10000)

A_fit, alpha_fit = popt
print(f"Final power-law fit: f(λ) = {A_fit:.2e} * λ^{alpha_fit:.2f}")

# After sigma clipping
fit_wave = lam_fit
fit_flux = flux_fit
cont_wind = np.column_stack((fit_wave,fit_flux))
#np.savetxt(sp_path+"cont_wind.txt",cont_wind)

def feii_component(wave, feii_wave, feii_flux, delta_v, sigma_v, norm):
    c = 299792.458
    shifted_wave = feii_wave * (1 + delta_v / c)
    dlam = np.mean(np.diff(shifted_wave))
    sigma_lam = np.mean(shifted_wave) * sigma_v / c
    sigma_pix = sigma_lam / dlam
    convolved_flux = gaussian_filter1d(feii_flux, sigma_pix)
    scaled_flux = norm * convolved_flux
    interp_func = interp1d(shifted_wave, scaled_flux, bounds_error=False, fill_value=0.0)
    return interp_func(wave)


def model(params, wave, feii_wave, feii_flux):
    amp = params['pl_amp']
    alpha = params['pl_alpha']
    delta_v = params['feii_dv']
    sigma_v = params['feii_sv']
    feii_norm = params['feii_norm']

    pl = powerlaw(wave, amp, alpha)
    feii = feii_component(wave, feii_wave, feii_flux, delta_v, sigma_v, feii_norm)
    return pl + feii


def residual(params, wave, flux,err, feii_wave, feii_flux):
    return (flux - model(params, wave, feii_wave, feii_flux))/err


# === Define Parameters ===
params = Parameters()
params.add('pl_alpha', value=-1.5, min=-3.0, max=1.0)
params.add('pl_amp', value=1e+4, min=0.0,max=1e+5)
params.add('feii_dv', value=-15, min=-1000, max=1000)
params.add('feii_sv', value=1500, min=300, max=5000)
params.add('feii_norm', value=1, min=0, max=5)  # or max=2

#cont_mask = mask_continuum(obs_wave, continuum_windows)



# === Fit ===
#result = minimize(residual, params, args=(obs_wave, obs_flux, feii_wave, feii_flux))
result = minimize(residual, params, args=(fit_wave, fit_flux,err_fit, feii_wave, feii_flux),method="emcee",run_mcmc_kwargs={"skip_initial_state_check":True}, steps=10000,
    burn=0,
    thin=1,
   is_weighted=True,
    progress=True,
    nwalkers=500)

with open(sp_path+"mcmc_pl_fe_new_spec.pkl", "wb") as f:
    pickle.dump(result, f)

print("saved the file",sp_path+"mcmc_pl_fe_new_spec.pkl" )
"""
with open(sp_path+"mcmc_pl_fe.pkl", "rb") as f:   #sp_path[:-9]+"mcmc_pl_fe.pkl"
    result = pickle.load(f)

"""


#fe_int,_ = curve_fit(feii_component,lam_fit,flux_fit,p0=[0.5,10,1e-3],maxfev=10000)
alpha_fit = result.params["pl_alpha"]
A_fit = result.params["pl_amp"]
pl_c = powerlaw(obs_wave, A_fit, alpha_fit)
fe_c = feii_component(obs_wave,feii_wave,feii_flux,result.params["feii_dv"],result.params["feii_sv"],result.params["feii_norm"])

# === Plot ===
best_fit = model(result.params, obs_wave, feii_wave, feii_flux)
#best_fit = pl_c+fe_c
pl_fe = np.column_stack((obs_wave,best_fit))
np.savetxt(sp_path+"pl_fe_mod_n.txt",pl_fe)

if 0 :
    sp = np.loadtxt(sp_path+"pl_fe_mod.txt")
    best_fit = sp[:,1]

plt.plot(obs_wave, obs_flux, label='Observed', color='black')
plt.plot(obs_wave,pl_c,label="power law", color="blue")
plt.plot(obs_wave,fe_c,label="FeII",color='brown')
plt.plot(obs_wave, best_fit, label='Power-law + FeII', color='red')
plt.plot(fit_wave, fit_flux, 'yellow',marker="o", label='Continuum Windows',alpha = 0.5)
plt.legend()
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.title("Fit: Power-law + FeII")


# === Print Results ===
result.params.pretty_print()
print("----------------------------------------------")
lmfit.report_fit(result)

plt.show()

"""
continuum_windows_1 = [
    (1280,1290), (1350, 1360), (1700, 1705), (2155, 2400)]#,
#    (2480, 2675), (2925, 3500), (4200, 4230), (4435, 4700),
#    (5100, 5535), (6000, 6250), (6800, 7000)
#]

continuum_windows_2 = [
    (1280,1290), (1350, 1360), (1445, 1465), (1700, 1705), (2155, 2400)]#,
#    (2480, 2675), (2925, 3500), (4200, 4230), (4435, 4700),
#    (5100, 5535), (6000, 6250), (6800, 7000)
#]

continuum_windows_3 = [
    (1280,1290), (1350, 1356), (1445, 1465), (1700, 1705), (2155, 2400)]#,
#    (2480, 2675), (2925, 3500), (4200, 4230), (4435, 4700),
#    (5100, 5535), (6000, 6250), (6800, 7000)
#]
"""