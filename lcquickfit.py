import sys

toi_num, path_name, model_mode = float(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

# 1272., '/Users/mason/lcquickfit/', 'duration'

import os, fnmatch
import lightkurve as lk
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
import scipy
import theano.tensor as tt
from scipy import stats
from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter, tess
import numpy as np
from numpy.random import normal, multivariate_normal
import uncertainties
from uncertainties import ufloat
import pandas as pd
import datetime
from celerite2.theano import terms, GaussianProcess
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from chainconsumer import ChainConsumer

from lcquickfit_functions import *


# General Constants
G = 6.6743 * 10**(-8)            # cm^3 / (g * s^2)
G_err = 1.5 * 10**(-12)

msun = 1.988409870698051*10**33           # g
msun_err = 4.468805426856864*10**28

rsun = 6.957 * 10**10           # cm
rsun_err = 1.4 * 10**7

rearth = 6.378137 * 10**8         # cm
rearth_err = 1.0 * 10**4

G_u = ufloat(G, G_err)
msun_u = ufloat(msun, msun_err)
rsun_u = ufloat(rsun, rsun_err)
rearth_u = ufloat(rearth, rearth_err)

rhosun_u = msun_u / (4/3*np.pi * rsun_u**3)
rhosun = rhosun_u.n
rhosun_err = rhosun_u.s           # g / cc

day = 86400                       # seconds




#####################################################


date = datetime.datetime.today().strftime("%d%b%y")

# Database for all TESS candidates import and relavent paths
lc_path = path_name #####'/u/scratch/m/macdouga/tess_ecc_test-ecc/'
xl_path = path_name #####'/u/home/m/macdouga/hmc_scripts/'
file_name = "lcquickfit_data.xlsx"
sheet_name = "Sheet1"

all_data = pd.read_excel(xl_path + file_name, sheet_name=sheet_name)

tess_pl = all_data




#####################################################


if type(toi_num) == float:
    pl = int(all_data[all_data['toi']==int(toi_num)].index[0])
elif type(toi_num) == int:
    pl = int(toi_num) - 1
else:
    print('ERROR: Incorrect type for \'toi_num\' input.')
    sys.exit()



# Establish system ID (TIC, TOI, planet ID within system)
tic_tev = all_data["tic-toi"][pl]
tic = int(tic_tev)
host_tic = str(tic)


# Determine if multi-system
tess_tic = list(all_data["tic-toi"])
if tess_tic.count(tic_tev) > 1:
    print('\nMultiplanet system! ')
    
system_data = all_data.loc[all_data["tic-toi"]==tic_tev]

full_toi_ids = list(system_data["full_toi_id"])
print(full_toi_ids)

toi = str(full_toi_ids[0]).split('.')[0]
host_toi = str(toi)
sys_name = 'TOI' + host_toi + '_TIC' + host_tic

pls, pl_ids, pl_names, candidate_ids, true_vals_all = compile_target_parameters(full_toi_ids, all_data, host_tic, host_toi, lc_path)

us, us_err = true_vals_all[0][-4], true_vals_all[0][-3]

    

#####################################################



lc = get_lc(lc_path, host_tic, host_toi)



dir_path = lc_path + sys_name + '-' + model_mode + '-' + date + '/'


if os.path.isdir(dir_path) == 0:
    os.mkdir(dir_path)
else:
    print(dir_path + ' already exists!')




#####################################################


t0s_true = []
pers_true = []
durs_true = []
rp_rss_true = []

for j in range(len(true_vals_all)):
    pers_true.append(float(true_vals_all[j][6]))
    durs_true.append(float(true_vals_all[j][8]))
    rp_rss_true.append(float(true_vals_all[j][-2]))
    t0s_true.append(float(true_vals_all[j][4]))
    
    
t0s_true = np.array(t0s_true)
pers_true = np.array(pers_true)
durs_true = np.array(durs_true)
rp_rss_true = np.array(rp_rss_true)

num_planets = len(pers_true)
    

print("t0_true = " + str(t0s_true))
print("per_true = " + str(pers_true))

###############################################



m = np.any(np.isfinite(lc['FLUX'])) & (lc['QUALITY'] == 0)



x = np.ascontiguousarray(lc["TIME"], dtype=np.float64)
y = np.ascontiguousarray(lc['FLUX'] - 1, dtype=np.float64)
yerr = np.ascontiguousarray(lc['FLUX_ERR'], dtype=np.float64)



texp = np.min(np.diff(x))

print('\nExposure time: ' + str(texp))



#####################################################



    
fig = plt.figure(figsize=(14, 7))
plt.plot(x, y, 'k.', label='data')
title = "Raw LC"
plt.title(title + " - TOI " + host_toi, fontsize=25, y=1.03)
plt.xlabel("Time [days]", fontsize=24)
plt.ylabel("Relative Flux", fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-raw_lc.png')
plt.close()


################################################


results = xo.estimators.lomb_scargle_estimator(
    x, y, max_peaks=1, min_period=1.0, max_period=np.nanmax(x)/2.0, 
    samples_per_peak=50
)

peak = results["peaks"][0]
print(peak)
freq, power = results["periodogram"]
fig = plt.figure(figsize=(14, 7))
plt.plot(1 / freq, power, "k")
plt.axvline(peak["period"], color="k", lw=4, alpha=0.3)
plt.xlim((1 / freq).min(), (1 / freq).max())
plt.xlabel("period [days]", fontsize=24)
plt.ylabel("power", fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-raw_lc-lomb_scargle.png')



period_match = [True for i in range(num_planets) 
 if pers_true[i]-durs_true[i] < 
 peak['period'] < pers_true[i]+durs_true[i]]





if peak['log_power'] > 0 and True not in period_match:
    print('Model stellar variability')
    with pm.Model() as model:

        # The mean flux of the time series
        mean = pm.Normal("mean", mu=0.0, sigma=10.0)

        # A jitter term describing excess white noise
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=2.0)

        # A term to describe the non-periodic variability
        sigma = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho = pm.InverseGamma(
            "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
        )

        # The parameters of the RotationTerm kernel
        sigma_rot = pm.InverseGamma(
            "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        log_period = pm.Normal("log_period", mu=np.log(peak["period"]), sigma=2.0)
        period = pm.Deterministic("period", tt.exp(log_period))
        log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
        log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=2.0)
        f = pm.Uniform("f", lower=0.1, upper=1.0)

        # Set up the Gaussian Process model
        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0)
        kernel += terms.RotationTerm(
            sigma=sigma_rot,
            period=period,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f,
        )
        gp = GaussianProcess(
            kernel,
            t=x,
            diag=yerr ** 2 + tt.exp(2 * log_jitter),
            mean=mean,
            quiet=True,
        )

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("gp", observed=y)

        # Compute the mean model prediction for plotting purposes
        pm.Deterministic("pred", gp.predict(y))

        # Optimize to find the maximum a posteriori parameters
        map_soln = pmx.optimize()

        
        
    fig = plt.figure(figsize=(14,7))
    plt.plot(x, y, "k.", label="data")
    plt.plot(x, map_soln["pred"], color="C1", label="model")
    plt.xlim(x.min(), x.max())
    plt.legend(fontsize=20)
    plt.xlabel("time [days]", fontsize=24)
    plt.ylabel("relative flux [ppt]", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + sys_name + '-raw_lc-variability_mod.png')
        
    y -= map_soln["pred"]





#####################################################




for j in range(len(full_toi_ids)):
    x_fold = (x - t0s_true[j] + 0.5*pers_true[j])%pers_true[j] - 0.5*pers_true[j]
    m = np.abs(x_fold) < durs_true[j]*3


    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Relative Flux", fontsize=24)
    plt.xlabel("Time Since Transit [days]", fontsize=24)
    
    plt.plot(x_fold[m], y[m], 'k.', label='data')
    
    title = "Detrended Phase Folded LC"
    plt.title(title + " - TOI " + str(full_toi_ids[j]), fontsize=25, y=1.03)
    plt.xlim(-2.0*durs_true[j],2.0*durs_true[j])
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-lc_flat-folded.png')
    plt.close()

    


#####################################################




if float(true_vals_all[0][-12]) > 0:
    rho_test = float(true_vals_all[0][-12])
    rho_test_err = float(true_vals_all[0][-11])
else:
    rho_test = rhosun
    rho_test_err = 0.1
    
shape = num_planets

bs = [0.5]*shape



start = None
mask = np.ones(len(x), dtype=bool)
# This is the current test - modified to look like my old starry model
if model_mode == 'full':
    with pm.Model() as model0:

        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0)

        # The stellar limb darkening parameters, using inputs from LDTK if stellar data is available
        #####BoundedNormal_u_star = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)
        #####u_star = BoundedNormal_u_star("u_star", mu=us, sd=us_err, shape=2)
        u = xo.QuadLimbDark("u", testval=us, shape=2)

        star_params = [mean, u]



        # Gaussian process noise model
        sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.median(yerr))
        log_sigma_gp = pm.Normal("log_sigma_gp", mu=0.0, sigma=10.0)
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(10.0), sigma=10.0)
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp), rho=tt.exp(log_rho_gp), Q=1.0 / 3
        )
        noise_params = [sigma, log_sigma_gp, log_rho_gp]


        # Planet parameters
        log_ror = pm.Normal(
            "log_ror", mu=np.log(rp_rss_true), sigma=10.0, shape=shape
        )
        ror = pm.Deterministic("ror", tt.exp(log_ror))

        # Orbital parameters
        log_period = pm.Normal("log_period", mu=np.log(pers_true), sigma=1.0, shape=shape)
        period = pm.Deterministic("period", tt.exp(log_period))

        BoundedNormal_t0 = pm.Bound(pm.Normal, lower=t0s_true-0.5*pers_true, upper=t0s_true+0.5*pers_true)
        t0 = BoundedNormal_t0("t0", mu=t0s_true, sigma=1.0, shape=shape)


        # The impact parameter as a free variable, not related to stellar radius ratio directly here
        b = xo.distributions.ImpactParameter("b", ror=ror, shape=shape, testval=bs)

        rho_star = pm.Normal("rho_star", mu=rho_test, sd=rho_test_err, shape=shape)

        ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
        ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))



        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, rho_star=rho_star, ecc=ecc, omega=omega)
        pm.Potential("jac_b", -tt.log(tt.abs_(orbit.jacobians["b"]["cos_incl"])))




        # Set up the mean transit model
        star = xo.LimbDarkLightCurve(u)

        def lc_model(t):
            return mean + tt.sum(
                star.get_light_curve(orbit=orbit, r=ror, t=t, texp=texp), axis=-1
            )

        pm.Deterministic("light_curves", star.get_light_curve(orbit=orbit, r=ror, t=x, texp=texp))


        # Finally the GP observation model
        gp = GaussianProcess(
            kernel, t=x, diag=yerr ** 2 + sigma ** 2, mean=lc_model, quiet=True
        )
        gp.marginal("obs", observed=y)


        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model0.check_test_point())

        # Optimize the model
        map_soln0 = model0.test_point

        map_soln0 = pmx.optimize(map_soln0, [sigma])
        map_soln0 = pmx.optimize(map_soln0, [ecs])
        map_soln0 = pmx.optimize(map_soln0, [ror, b, rho_star])
        map_soln0 = pmx.optimize(map_soln0, noise_params)
        map_soln0 = pmx.optimize(map_soln0, star_params)

elif model_mode == 'duration':
    with pm.Model() as model0:

        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0)

        # The stellar limb darkening parameters, using inputs from LDTK if stellar data is available
        #####BoundedNormal_u_star = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)
        #####u_star = BoundedNormal_u_star("u_star", mu=us, sd=us_err, shape=2)
        u = xo.QuadLimbDark("u", testval=us, shape=2)

        star_params = [mean, u]



        # Gaussian process noise model
        sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.median(yerr))
        log_sigma_gp = pm.Normal("log_sigma_gp", mu=0.0, sigma=10.0)
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(10.0), sigma=10.0)
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp), rho=tt.exp(log_rho_gp), Q=1.0 / 3
        )
        noise_params = [sigma, log_sigma_gp, log_rho_gp]


        # Planet parameters
        log_ror = pm.Normal(
            "log_ror", mu=np.log(rp_rss_true), sigma=10.0, shape=shape
        )
        ror = pm.Deterministic("ror", tt.exp(log_ror))

        # Orbital parameters
        log_period = pm.Normal("log_period", mu=np.log(pers_true), sigma=1.0, shape=shape)
        period = pm.Deterministic("period", tt.exp(log_period))

        BoundedNormal_t0 = pm.Bound(pm.Normal, lower=t0s_true-0.5*pers_true, upper=t0s_true+0.5*pers_true)
        t0 = BoundedNormal_t0("t0", mu=t0s_true, sigma=1.0, shape=shape)


        # The impact parameter as a free variable, not related to stellar radius ratio directly here
        b = xo.distributions.ImpactParameter("b", ror=ror, shape=shape, testval=bs)
        

        log_dur = pm.Normal("log_dur", mu=np.log(durs_true), sigma=10.0, shape=shape)
        dur = pm.Deterministic("dur", tt.exp(log_dur))



        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, duration=dur)
        pm.Potential("jac_dur", -tt.log(tt.abs_(orbit.jacobians["duration"]["rho_star"])))
        pm.Potential("jac_b", -tt.log(tt.abs_(orbit.jacobians["b"]["cos_incl"])))

        pm.Deterministic("rho_circ", orbit.rho_star)


        # Set up the mean transit model
        star = xo.LimbDarkLightCurve(u)

        def lc_model(t):
            return mean + tt.sum(
                star.get_light_curve(orbit=orbit, r=ror, t=t, texp=texp), axis=-1
            )

        pm.Deterministic("light_curves", star.get_light_curve(orbit=orbit, r=ror, t=x, texp=texp))

        # Finally the GP observation model
        gp = GaussianProcess(
            kernel, t=x, diag=yerr ** 2 + sigma ** 2, mean=lc_model, quiet=True
        )
        gp.marginal("obs", observed=y)


        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model0.check_test_point())

        # Optimize the model
        map_soln0 = model0.test_point

        map_soln0 = pmx.optimize(map_soln0, [sigma])
        map_soln0 = pmx.optimize(map_soln0, [ror, b, dur])
        map_soln0 = pmx.optimize(map_soln0, noise_params)
        map_soln0 = pmx.optimize(map_soln0, star_params)


for i in range(num_planets):
    with model0:
        gp_pred = pmx.eval_in_model(
            gp.predict(y, include_mean=False), map_soln0)

    plt.figure(figsize=(14, 7))
    x_fold = (x - map_soln0["t0"][i] + 0.5 * map_soln0["period"][i]) % map_soln0["period"][i] - 0.5 * map_soln0["period"][i]
    inds = np.argsort(x_fold)
    plt.scatter(x_fold, y - gp_pred - map_soln0["mean"], c=x, s=3)
    plt.plot(x_fold[inds], map_soln0["light_curves"][:, i][inds] - map_soln0["mean"], "k")
    plt.xlabel("Time since transit [days]", fontsize=24)
    plt.ylabel("Relative Flux", fontsize=24)
    title = "Detrended Phase Folded LC (relative coloring)"
    plt.title(title + " - TOI " + str(full_toi_ids[i]), fontsize=25, y=1.03)
    cb = plt.colorbar()
    cb.set_label(label="Time [days]", fontsize=20)
    cb.ax.tick_params(labelsize=16)
    _ = plt.xlim(-durs_true[i]*2, durs_true[i]*2)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + sys_name + '-raw_lc-variability_mod.png')

plot_light_curve(x, y, yerr, map_soln0, gp_pred, mask=None, g=-1, durations=durs_true, identifiers=[sys_name, pl_names], path=dir_path);

plot_light_curve(x, y, yerr, map_soln0, gp_pred, mask=None, g=0, idx='_transit0', durations=durs_true, identifiers=[sys_name, pl_names], path=dir_path);


mod = (
    gp_pred
    + map_soln0["mean"]
    + np.sum(map_soln0["light_curves"], axis=-1)
)
resid = y - mod
rms = np.sqrt(np.median(resid ** 2))
mask = np.abs(resid) < 6 * rms

fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=False)
plt.plot(x, resid, "k.")
plt.plot(x[~mask], resid[~mask], "xr", label="outliers")


colors = ['b', 'orange', 'g', 'r', 'm']
for j in range(num_planets):
    cc = colors[j]
    for i in np.arange(1+int((np.max(x)-np.min(x))/map_soln0['period'][j])):
        if i == 0:
            lab = 'planet ' + 'b'
        else:
            lab = None
        plt.axvline(map_soln0['t0'][j] + i * map_soln0['period'][j], alpha=0.5, ls='--', color=cc, label=lab)

plt.axhline(0, color="#aaaaaa", lw=1)
plt.ylabel("Residuals of Relative Flux", fontsize=24)
plt.xlabel("Time [days]", fontsize=24)
plt.legend(fontsize=20, loc=4)
plt.xlim(x.min(), x.max())
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
title = "Residuals of LC Fit"
plt.title(title + " - TOI " + str(toi), fontsize=25, y=1.03)
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(dir_path + sys_name + '-gp_residuals-mod.png')
plt.close()

x_orig, y_orig, yerr_orig = x.copy(), y.copy(), yerr.copy()

lc_data = pd.DataFrame(columns=['x','y','yerr'])
lc_data['x'] = x
lc_data['y'] = y
lc_data['yerr'] = yerr

lc_data.to_csv(dir_path + sys_name + '-detrended_lc-data.csv')

x, y, yerr = x_orig[mask], y_orig[mask], yerr_orig[mask]

transit_mask = np.zeros(len(x), dtype=bool)

for j in range(num_planets):
    transit_mask0 = (
        np.abs(
            (x - map_soln0['t0'][j] + 0.5 * map_soln0['period'][j]) % map_soln0['period'][j] - 0.5 * map_soln0['period'][j]
        ) < durs_true[j]*2)
    transit_mask = np.logical_or(transit_mask, transit_mask0)

x, y, yerr = x[transit_mask], y[transit_mask], yerr[transit_mask]



start = None
mask = np.ones(len(x), dtype=bool)
# This is the current test - modified to look like my old starry model
if model_mode == 'full':
    with pm.Model() as model:

        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0)

        # The stellar limb darkening parameters, using inputs from LDTK if stellar data is available
        #####BoundedNormal_u_star = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)
        #####u_star = BoundedNormal_u_star("u_star", mu=us, sd=us_err, shape=2)
        u = xo.QuadLimbDark("u", testval=us, shape=2)

        star_params = [mean, u]




        # Planet parameters
        log_ror = pm.Normal(
            "log_ror", mu=np.log(rp_rss_true), sigma=10.0, shape=shape
        )
        ror = pm.Deterministic("ror", tt.exp(log_ror))

        # Orbital parameters
        log_period = pm.Normal("log_period", mu=np.log(pers_true), sigma=1.0, shape=shape)
        period = pm.Deterministic("period", tt.exp(log_period))

        BoundedNormal_t0 = pm.Bound(pm.Normal, lower=t0s_true-0.5*pers_true, upper=t0s_true+0.5*pers_true)
        t0 = BoundedNormal_t0("t0", mu=t0s_true, sigma=1.0, shape=shape)



        # The impact parameter as a free variable, not related to stellar radius ratio directly here
        b = xo.distributions.ImpactParameter("b", ror=ror, shape=shape, testval=bs)

        rho_star = pm.Normal("rho_star", mu=rho_test, sd=rho_test_err, shape=shape)

        ecs = pmx.UnitDisk("ecs", testval=np.array([0.01, 0.0]))
        ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))



        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, rho_star=rho_star, ecc=ecc, omega=omega)
        pm.Potential("jac_b", -tt.log(tt.abs_(orbit.jacobians["b"]["cos_incl"])))



        # Set up the mean transit model
        star = xo.LimbDarkLightCurve(u)

        lc_mod = mean + tt.sum(
                star.get_light_curve(orbit=orbit, r=ror, t=x, texp=texp), axis=-1)

        pm.Normal("obs", mu=lc_mod, sd=yerr, observed=y)


        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model.check_test_point())

        # Optimize the model
        map_soln = model.test_point

        map_soln = pmx.optimize(map_soln)
        map_soln = pmx.optimize(map_soln, [ecs])
        map_soln = pmx.optimize(map_soln, [ror, b, rho_star])
        map_soln = pmx.optimize(map_soln, star_params)

elif model_mode == 'duration':
    with pm.Model() as model:

        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0)

        # The stellar limb darkening parameters, using inputs from LDTK if stellar data is available
        #####BoundedNormal_u_star = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)
        #####u_star = BoundedNormal_u_star("u_star", mu=us, sd=us_err, shape=2)
        u = xo.QuadLimbDark("u", testval=us, shape=2)

        star_params = [mean, u]


        # Planet parameters
        log_ror = pm.Normal(
            "log_ror", mu=np.log(rp_rss_true), sigma=10.0, shape=shape
        )
        ror = pm.Deterministic("ror", tt.exp(log_ror))

        # Orbital parameters
        log_period = pm.Normal("log_period", mu=np.log(pers_true), sigma=1.0, shape=shape)
        period = pm.Deterministic("period", tt.exp(log_period))

        BoundedNormal_t0 = pm.Bound(pm.Normal, lower=t0s_true-0.5*pers_true, upper=t0s_true+0.5*pers_true)
        t0 = BoundedNormal_t0("t0", mu=t0s_true, sigma=1.0, shape=shape)



        # The impact parameter as a free variable, not related to stellar radius ratio directly here
        b = xo.distributions.ImpactParameter("b", ror=ror, shape=shape, testval=bs)
        

        log_dur = pm.Normal("log_dur", mu=np.log(durs_true), sigma=10.0, shape=shape)
        dur = pm.Deterministic("dur", tt.exp(log_dur))



        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b, duration=dur)
        pm.Potential("jac_dur", -tt.log(tt.abs_(orbit.jacobians["duration"]["rho_star"])))
        pm.Potential("jac_b", -tt.log(tt.abs_(orbit.jacobians["b"]["cos_incl"])))

        pm.Deterministic("rho_circ", orbit.rho_star)


        # Set up the mean transit model
        star = xo.LimbDarkLightCurve(u)

        lc_mod = mean + tt.sum(
                star.get_light_curve(orbit=orbit, r=ror, t=x, texp=texp), axis=-1)

        pm.Normal("obs", mu=lc_mod, sd=yerr, observed=y)

        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model.check_test_point())

        # Optimize the model
        map_soln = model.test_point

        map_soln = pmx.optimize(map_soln)
        map_soln = pmx.optimize(map_soln, [ror, b, dur])
        map_soln = pmx.optimize(map_soln, star_params)


print(map_soln)
print('\n')

if len(pers_true) == 1 and model_mode == 'duration':
    chainz = 2
    corez = 1
    tunez = 10000
    drawz = 5000
elif len(pers_true) != 1 and model_mode == 'duration':
    chainz = 2
    corez = 1
    tunez = 6000
    drawz = 4000
elif model_mode == 'full':
    chainz = 2
    corez = 1
    tunez = 3000
    drawz = 2000

np.random.seed(286923464)
with model:
    trace = pmx.sample(
        tune=tunez,
        draws=drawz,
        start=map_soln,
        chains=chainz,
        cores=corez,
        return_inferencedata=False,
        target_accept=0.95
    )



tr = trace

if model_mode == 'full':
    varnames=["t0","period","ror","b","ecc","omega","rho_star","mean","u"]
    colnames=[]

    for vv in varnames[:7]:
        for j in range(len(pers_true)):
            colnames.append(vv + '__' + str(j))
            
    colnames += ["mean","u__0","u__1"]

    df = pm.trace_to_dataframe(tr, varnames=varnames)
    df.columns = colnames
    df.to_csv(dir_path + sys_name + "-trace.csv")

    df = pd.read_csv(dir_path + sys_name + "-trace.csv")

                
    #####################################################


    fig1 = pm.traceplot(tr, var_names=varnames, compact=False)
    plt.rc('text', usetex=False)
    figall = fig1[0][0].figure
    figall.savefig(dir_path + sys_name + '-trace_plots-all.png')
    plt.close()

    if len(pers_true) > 1:
        single_vars = ["ror", "b", "ecc", "omega", "rho_star"]
        for var in single_vars:
            fig0 = pm.traceplot(tr, var_names=[var], compact=False)
            plt.rc('text', usetex=False)
            figall = fig0[0][0].figure
            figall.savefig(dir_path + sys_name + '-trace_plots-' + var + '.png')
            plt.close()
            

    #####################################################


    c = ChainConsumer()
    c.configure(usetex=False)
    c.add_chain(np.array(df[colnames]),parameters=colnames)
    c.configure(usetex=False, label_font_size=10, tick_font_size=8)
    plt.rc('text', usetex=False)
    plt.gcf().subplots_adjust(bottom=0.01, left=0.01)

    fig2 = c.plotter.plot()
    fig2.tight_layout()
    fig2.savefig(dir_path + sys_name + '-corner_full.png')
    plt.close()


    #####################################################


    for j in range(len(pers_true)):
        c = ChainConsumer()
        c.configure(usetex=False)
        colsmain = ["ror__" + str(j),"b__" + str(j),"ecc__" + str(j),"omega__" + str(j),"rho_star__" + str(j)]
        colsmain_labels = ["ror", "b", "ecc", "omega", "rho_star"]
        c.add_chain(np.array(df[colsmain]),parameters=colsmain_labels)
        c.configure(usetex=False, label_font_size=10, tick_font_size=8)
        plt.rc('text', usetex=False)
        plt.gcf().subplots_adjust(bottom=0.01, left=0.01)

        fig3 = c.plotter.plot()
        fig3.tight_layout()
        fig3.savefig(dir_path + pl_names[j] + '-corner_main.png')
        plt.close()

        c = ChainConsumer()
        c.configure(usetex=False)
        colsmain = ["ecc__" + str(j),"omega__" + str(j)]
        colsmain_labels = ["ecc", "omega"]
        c.add_chain(np.array(df[colsmain]),parameters=colsmain_labels)
        c.configure(usetex=False, label_font_size=10, tick_font_size=8)
        plt.rc('text', usetex=False)
        plt.gcf().subplots_adjust(bottom=0.01, left=0.01)

        fig3 = c.plotter.plot()
        fig3.tight_layout()
        fig3.savefig(dir_path + pl_names[j] + '-corner_e-w.png')
        plt.close()


elif model_mode == 'duration':
    varnames=["t0","period","ror","b","dur","rho_circ","mean","u"]
    colnames=[]

    for vv in varnames[:6]:
        for j in range(len(pers_true)):
            colnames.append(vv + '__' + str(j))
            
    colnames += ["mean","u__0","u__1"]

    df = pm.trace_to_dataframe(tr, varnames=varnames)
    df.columns = colnames
    df.to_csv(dir_path + sys_name + "-trace.csv")

    df = pd.read_csv(dir_path + sys_name + "-trace.csv")

                
    #####################################################


    fig1 = pm.traceplot(tr, var_names=varnames, compact=False)
    plt.rc('text', usetex=False)
    figall = fig1[0][0].figure
    figall.savefig(dir_path + sys_name + '-trace_plots-all.png')
    plt.close()

    if len(pers_true) > 1:
        single_vars = ["ror", "b", "dur", "rho_circ"]
        for var in single_vars:
            fig0 = pm.traceplot(tr, var_names=[var], compact=False)
            plt.rc('text', usetex=False)
            figall = fig0[0][0].figure
            figall.savefig(dir_path + sys_name + '-trace_plots-' + var + '.png')
            plt.close()
            

    #####################################################


    c = ChainConsumer()
    c.configure(usetex=False)
    c.add_chain(np.array(df[colnames]),parameters=colnames)
    c.configure(usetex=False, label_font_size=10, tick_font_size=8)
    plt.rc('text', usetex=False)
    plt.gcf().subplots_adjust(bottom=0.01, left=0.01)

    fig2 = c.plotter.plot()
    fig2.tight_layout()
    fig2.savefig(dir_path + sys_name + '-corner_full.png')
    plt.close()


    #####################################################


    for j in range(len(pers_true)):
        c = ChainConsumer()
        c.configure(usetex=False)
        colsmain = ["ror__" + str(j),"b__" + str(j),"dur__" + str(j),"rho_circ__" + str(j)]
        colsmain_labels = ["ror", "b", "dur", "rho_circ"]
        c.add_chain(np.array(df[colsmain]),parameters=colsmain_labels)
        c.configure(usetex=False, label_font_size=10, tick_font_size=8)
        plt.rc('text', usetex=False)
        plt.gcf().subplots_adjust(bottom=0.01, left=0.01)

        fig3 = c.plotter.plot()
        fig3.tight_layout()
        fig3.savefig(dir_path + pl_names[j] + '-corner_main.png')
        plt.close()





cases = ['iso_teff',
         'iso_color',
         'tic']

p = []
t0 = []

for j in range(len(pers_true)):
    p.append(np.median(tr["period"][:, j]))
    t0.append(np.median(tr["t0"][:, j]))
    

##################################################
trace = tr
trace0 = tr

data_df = pd.DataFrame()
data_df['b'] = list(df['b__0'])
data_df['ror'] = list(df['ror__0'])

r_low = list(data_df.sort_values(by='b')['ror'])[0]


for j in range(len(pers_true)):
    if model_mode != 'duration':
        break
    # Plot the folded data
    
    mask = np.full(len(x), False)
    
    for jj in range(len(pers_true)):
        if j != jj:
            x_fold_jj = (x - t0[jj] + 0.5 * p[jj]) % p[jj] - 0.5 * p[jj]
            mask += np.abs(x_fold_jj) < durs_true[jj] * 0.75
            
    mask = ~mask

    masktemp = np.full(len(x), False)

    for jj in range(len(pers_true)):
        if j == jj:
            x_fold_j = (x - t0[jj] + 0.5 * p[jj]) % p[jj] - 0.5 * p[jj]
            masktemp += np.abs(x_fold_j) < durs_true[jj] * 0.75

    for mm in range(len(mask)):
        if mask[mm] == False and masktemp[mm] == True:
            mask[mm] = True
    
    x_fold = (x[mask] - t0[j] + 0.5 * p[j]) % p[j] - 0.5 * p[j]
    

    # Overplot the phase binned light curve
    if 4.0*durs_true[j] < 0.5*p[j]:
        bound = 4.0*durs_true[j]
    else:
        bound = 0.5*p[j]
    bins = np.linspace(-1*bound, bound, 85)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=y[mask] - np.median(tr["mean"]))
    denom[num == 0] = 1.0

    samples = np.empty((50, len(x[mask])))

    if model_mode == 'duration':
        with model:
            orbit = xo.orbits.KeplerianOrbit(period=p, t0=t0, b=b, duration=dur)
            # Compute the model light curve using starry
            light_curves = xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=ror, t=x[mask], texp=texp)
            light_curve = pm.math.sum(light_curves, axis=-1) + mean
            y_grid = light_curve
            for i, sample in enumerate(pmx.get_samples_from_trace(tr, size=50)):
                samples[i] = pmx.eval_in_model(y_grid, sample)

    isort = np.argsort(x_fold)


    rho_cases = [] 
    cases_tags = cases


    for c in ['iso_teff', 'iso_color']:
        rho_cases.append('rho-' + c)
        rho_cases.append('rho_err1-' + c)
        rho_cases.append('rho_err2-' + c)

    rho_cases += ["tic_ms", "tic_ms_err", "tic_rs", "tic_rs_err"]

    rho_data = np.array(all_data.loc[all_data["full_toi_id"] == all_data["full_toi_id"][pl]][rho_cases])[0]


    ecc_data_array = []
    
    ew_data = 0

    for i in range(len(cases)):

        case = cases[i]
        case_tag = cases_tags[i]

        if case != 'tic':
            rhostar_true = rho_data[3*i]
            rhostar_true_err1 = rho_data[3*i+1]
            rhostar_true_err2 = -1*rho_data[3*i+2]
            rhostar_true_err = np.mean([rhostar_true_err1, rhostar_true_err2])


            rhostar_case_u = ufloat(rhostar_true, rhostar_true_err)          # rho_sun (CHECK UNITS OF INPUT SOURCE)

            rhostar_case_unc = rhostar_case_u * rhosun_u
            rhostar_case = float(rhostar_case_unc.n)
            rhostar_err_case = float(rhostar_case_unc.s)             # g / cm^3

            rhostar_true = rhostar_case
            rhostar_err_true = rhostar_err_case
        elif case == 'tic':
            mass_u = ufloat(rho_data[-4], rho_data[-3])
            rad_u = ufloat(rho_data[-2], rho_data[-1])

            rhostar_case_unc = mass_u * msun_u / (4./3. * np.pi * (rad_u * rsun_u)**3)
            rhostar_case = float(rhostar_case_unc.n)
            rhostar_err_case = float(rhostar_case_unc.s)             # g / cm^3

            rhostar_true = rhostar_case
            rhostar_err_true = rhostar_err_case

        if rhostar_case > 0 and rhostar_err_case > 0:

            if rhostar_err_case < 0.05 * rhostar_case:
                print(case, ' error bar unreasonably small: ', rhostar_case, rhostar_err_case)
                rhostar_err_case = 0.05 * rhostar_case

            print(rhostar_case)
            print(rhostar_err_case)


            tr_temp = pd.read_csv(dir_path + sys_name + "-trace.csv")
            rho_tr = np.array(list(tr_temp['rho_circ__' + str(j)]))

            rho_obs = (rhostar_case, rhostar_err_case)
            
            rp_f0 = np.median(tr_temp["ror__" + str(j)])
            rp_f0_err1 = np.percentile(q=[15.865], a=tr_temp["ror__" + str(j)])[0]
            rp_f0_err2 = np.percentile(q=[84.135], a=tr_temp["ror__" + str(j)])[0]
            
            rp_rs_fin = ufloat(rp_f0, np.nanmean([rp_f0_err2-rp_f0, -1*(rp_f0_err1-rp_f0)]))
            if case != 'tic':
                rad_case = ufloat(all_data["rad-"+case][pl], np.nanmean([all_data["rad_err1-"+case][pl], -1*all_data["rad_err2-"+case][pl]]))
                rad_st_case = float(rad_case.n)
                rad_st_err_case = float(rad_case.s)
                
                mass_case = ufloat(all_data["mass-"+case][pl], np.nanmean([all_data["mass_err1-"+case][pl], -1*all_data["mass_err2-"+case][pl]]))    
                mass_st_case = float(mass_case.n)
                mass_st_err_case = float(mass_case.s)
                
            elif case == 'tic':
                rad_case = rad_u
                rad_st_case = float(rad_u.n)
                rad_st_err_case = float(rad_u.s)
                
                mass_st_case = float(mass_u.n)
                mass_st_err_case = float(mass_u.s)
                
                
            if rad_st_case > 0 and rad_st_err_case > 0:
                rad_pl_fin = rad_case * rp_rs_fin * rsun_u/rearth_u
                rad_pl_case = float(rad_pl_fin.n)
                rad_pl_err_case = float(rad_pl_fin.s)
            else:
                rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case, rad_pl_case, rad_pl_err_case = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
             
            
            lower_rho, upper_rho = 0.00001, 250.0
            mu_rho, sigma_rho = rho_obs
            XX = stats.truncnorm(
                (lower_rho - mu_rho) / sigma_rho, (upper_rho - mu_rho) / sigma_rho, loc=mu_rho, scale=sigma_rho)

            rho_obs_array = XX.rvs(len(rho_tr))

            rho_ratio = rho_tr/rho_obs_array
            rho_df = pd.DataFrame(np.array([list(rho_ratio), list(rho_tr), list(rho_obs_array)]).T, columns=["rho_ratio", "rho_circ", "rho_obs"])

            tot_len = len(host_toi)
            if tot_len == 4:
                prefix = 'T00'
            elif tot_len == 3:
                prefix = 'T000'
            
            rho_df.to_csv(dir_path + prefix + host_toi + pl_ids[j] + '-rho_ratio_posterior-' + case + '.csv')

            
            
            fupsample = 1000 # duplicate chains by this amount
            rho_circ = np.hstack([rho_tr.flatten()]*fupsample)
            ecc00 = np.random.uniform(0, 1, len(rho_circ))
            omega00 = np.random.uniform(-0.5*np.pi, 1.5*np.pi, len(rho_circ))
            g = (1 + ecc00 * np.sin(omega00)) / np.sqrt(1 - ecc00 ** 2)
            rho00 = rho_circ / g ** 3



            #####################################################


            # Build up interpolated KDE
            samples00 = rho_tr.flatten()
            smin, smax = np.min(samples00), np.max(samples00)
            width = smax - smin
            xx = np.linspace(smin, smax, 1000)
            yy = stats.gaussian_kde(samples00)(xx)
            xi = np.linspace(xx[0],xx[-1],1000)
            rhocircpost = lambda xi: np.interp(xi,xx,yy,left=0,right=0)


            # Generate a grid of e, omega, g
            emin, emax, esamp = 0,0.99,99
            omegamin, omegamax, omegasamp = -0.5*np.pi,1.5*np.pi,100
            ecc00 = np.linspace(emin,emax,esamp)
            omega00 = np.linspace(omegamin,omegamax,omegasamp)
            ecc2d,omega2d = np.meshgrid(ecc00,omega00,indexing='ij')
            g2d = (1+ecc2d*np.sin(omega2d))/np.sqrt(1-ecc2d**2) 

            # Compute the posterior probability as a function of g
            def func(rho00, g):
                rhocircobs = np.exp(-0.5 * ((rho00 - rho_obs[0]*g**3)/(rho_obs[1]*g**3))**2)
                return rhocircpost(rho00) * rhocircobs
            gp = np.logspace(np.log10(np.min(g)),np.log10(np.max(g)),1000)
            probgp = [scipy.integrate.quad(func,smin,smax,args=(_g),full_output=1)[0] for _g in gp]
            probg = lambda g: np.interp(g,gp,probgp)
            prob = probg(g2d)
            
            
            rho_post = rho_ratio

            rho_d = np.linspace(np.min(rho_post), np.max(rho_post), 1000)

            bw = 0.9 * np.min([np.std(rho_post), scipy.stats.iqr(rho_post)/1.34]) * (len(rho_d)**(-1/5))

            kde = KernelDensity(bandwidth=bw, kernel='gaussian')
            kde.fit(rho_post[:,None])

            logprob = kde.score_samples(rho_d[:, None])


            fig = plt.figure(figsize=[10,7])
            plt.hist(rho_post,50,alpha=0.5, density=True, label=r'$\rho_{circ}$/$\rho_{obs}$' + ' (modeled / isoclassify)', zorder=5)
            plt.plot(rho_d, np.exp(logprob), 'k', lw=3, label='KDE (w/ ' + r'$\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc)', zorder=10)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18) 
            plt.xlabel(r'$\rho_{circ}$/$\rho_{obs}$', fontsize=24)
            plt.ylabel('Count Density', fontsize=24)
            title = 'TOI ' + str(full_toi_ids[j]) + r': $\rho_{circ}$/$\rho_{obs}$ ' + ' - ' + case
            plt.title(title, fontsize=25, y=1.03)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-rho_ratio_hist-' + case + '.png')
            plt.close()

            interp_df = pd.DataFrame(columns=["rho_ratio", "weight", "logprob"])
            interp_df["rho_ratio"] = rho_d
            interp_df["weight"] = np.exp(logprob)
            interp_df["logprob"] = logprob

            interp_df.to_csv(dir_path + prefix + host_toi + pl_ids[j] + '-rho_ratio_interp-' + case + '.csv')

            posts = interp_df
            rho_ratios_in = np.array(list(posts["rho_ratio"]))
            weights_in = np.array(list(posts["weight"]))
            logprobs_in = np.array(list(posts["logprob"]))


            ecos_esin_interp(interp_df, rho_df, dir_path, pl_names[j], case)



            if np.sum(prob) > 0:
                try:
                    c = ChainConsumer()
                    #c.configure(usetex=False) #, label_font_size=22, tick_font_size=16)
                    plt.rc('text', usetex=False)

                    c.add_chain([ecc00,omega00*180/np.pi],grid=True,weights=prob,kde=True,smooth=True,parameters=['$e$','$\omega$'])
                    c.configure(usetex=False, label_font_size=26, tick_font_size=18)
                    fig = c.plotter.plot(figsize=[8,8])

                    # Annotate the plot with the planet's period
                    txt = "TOI " + str(full_toi_ids[j]) + " (" + case + ")"
                    plt.annotate(
                        txt,
                        (0, 0),
                        xycoords="axes fraction",
                        xytext=(110, 0),
                        rotation=270,
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=18,
                        annotation_clip=False
                    )

                    #fig.set_size_inches(3.0 + fig.get_size_inches())  # Resize fig for doco. You don't need this.
                    plt.gcf().subplots_adjust(bottom=0.15, right=0.8)
                    plt.tight_layout()
                    fig.savefig(dir_path + pl_names[j] + '-ecc-' + case + '.png')
                    plt.close()


                    if ew_data == 0:
                        post = pd.DataFrame(columns = ['ecc'])
                        post['ecc'] = ecc00
                        post.to_csv(dir_path + pl_names[j] + '-posteriors_e.csv')

                        post = pd.DataFrame(columns = ['omega'])
                        post['omega'] = omega00*180/np.pi
                        post.to_csv(dir_path + pl_names[j] + '-posteriors_w.csv')
                        ew_data += 1
                    

                    post = pd.DataFrame(prob)
                    post.to_csv(dir_path + pl_names[j] + '-posteriors_prob-' + case + '.csv')
                    

                except IndexError:
                    rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]
                    rhostar_case, rhostar_err_case, w_f, w_f_err1, w_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]

                    case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]

                    print("\nWeights sum to zero! Not deriving ecc-omega for " + host_tic + " " + case + "\n")
                    continue   
            else:
                rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]
                rhostar_case, rhostar_err_case, w_f, w_f_err1, w_f_err2 = [rhostar_case, rhostar_err_case, np.nan, np.nan, np.nan]

                case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]

                print("\nWeights sum to zero! Not deriving ecc-omega for " + host_tic + " " + case + "\n")
                continue




            ecc_out = c.analysis.get_summary()['$e$']
            w_out = c.analysis.get_summary()['$\omega$']


            ecc_f = ecc_out[1]
            if str(ecc_out[0]) != 'None' and str(ecc_out[1]) != 'None':
                ecc_f_err1 = ecc_out[0] - ecc_out[1]
            else:
                ecc_f_err1 = np.nan

            if str(ecc_out[2]) != 'None' and str(ecc_out[1]) != 'None':
                ecc_f_err2 = ecc_out[2] - ecc_out[1]
            else:
                ecc_f_err2 = np.nan



            w_f = w_out[1]
            if str(w_out[0]) != 'None' and str(w_out[1]) != 'None':
                w_f_err1 = w_out[0] - w_out[1]
            else:
                w_f_err1 = np.nan

            if str(w_out[2]) != 'None' and str(w_out[1]) != 'None':
                w_f_err2 = w_out[2] - w_out[1]
            else:
                w_f_err2 = np.nan

            print('\necc + errs:', ecc_f, ecc_f_err1, ecc_f_err2)
            print('omega + errs:', w_f, w_f_err1, w_f_err2)

            case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]



            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)

            plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".", color='silver', zorder=-1000)#, alpha=0.3)
            plt.plot(-0.11, 0, ".", ms=10, color='silver', zorder=-1000)#, alpha=0.3)

            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", ms=12, alpha=1.0, zorder=-100) ###### NEW!


            samples0 = np.empty((2, len(x[mask])))

            if model_mode == 'duration':
                with model:
                    orbit = xo.orbits.KeplerianOrbit(period=p, t0=t0, b=0, rho_star=rho_obs[0], ecc=0, omega=np.pi/2.0)
                    # Compute the model light curve using starry
                    light_curves = xo.LimbDarkLightCurve(us[0], us[1]).get_light_curve(orbit=orbit, r=np.median(tr['ror'][:, j]), t=x[mask], texp=texp)
                    light_curve = pm.math.sum(light_curves, axis=-1) + np.median(tr['mean'])
                    y_grid = light_curve
                    for i, sample in enumerate(pmx.get_samples_from_trace(tr, size=2)):
                        samples0[i] = pmx.eval_in_model(y_grid, sample)

            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.5, label="Sampled Model", zorder=-1)
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.2)

            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.5, lw=3, label="Circular Model", zorder=-10)
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.5, lw=3)


            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$" + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = '  + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Relative Flux", fontsize=24)
            plt.xlim(-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_circ-' + case_tag + '.png')
            plt.close()



            fig = plt.figure(figsize=(14, 7))
            plt.rc('text', usetex=False)
            #plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".k", label="data", zorder=-1000)
            plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", label="binned", zorder=-100, alpha=1.0, ms=12)
            plt.plot(x_fold[isort], samples0[0,isort] - np.median(tr["mean"]),color="blue", alpha=0.5, lw=3, label="Circular Posterior", zorder=-10)
            plt.plot(x_fold[isort], samples0[1:,isort].T - np.median(tr["mean"]),color="blue", alpha=0.5, lw=3)

            plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.5, label="Sampled Model", zorder=-1)
            plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.2)

            # Annotate the plot with the planet's period
            txt = "TOI " + str(full_toi_ids[j]) + "\nradius = {0:.3f} +/- {1:.3f} ".format(
                rad_pl_case, rad_pl_err_case) + "$R_{Earth}$"  + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
                np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
            )
            plt.annotate(
                txt,
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=20,
            )

            plt.legend(fontsize=20, loc=4)
            plt.title('Fit Model vs ' + r'Circ Model ($\rho_{obs}$ = ' + str(round(rhostar_true,2)) + ' [' + str(round(rhostar_true_err,2)) + '] g/cc - ' + case + ')', fontsize=25, y=1.03)
            plt.xlabel("Time since mid-transit [days]", fontsize=24)
            plt.ylabel("Relative Flux", fontsize=24)
            plt.xlim(-1*bound/2.0, bound/2.0);
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            fig.savefig(dir_path + pl_names[j] + '-folded_mod_binned-' + case_tag + '.png')
            plt.close()


        else:
            rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2 = [np.nan, np.nan, np.nan, np.nan, np.nan]
            rhostar_case, rhostar_err_case, w_f, w_f_err1, w_f_err2 = [np.nan, np.nan, np.nan, np.nan, np.nan]
            rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case, rad_pl_case, rad_pl_err_case = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

            case_data = [tic, full_toi_ids[j], case, rhostar_case, rhostar_err_case, ecc_f, ecc_f_err1, ecc_f_err2, w_f, w_f_err1, w_f_err2, rad_pl_case, rad_pl_err_case, rad_st_case, rad_st_err_case, mass_st_case, mass_st_err_case]

            print("\nNo valid stellar density available for case: " + case)

        
        ecc_data_array.append(case_data)
        
    ecc_output = pd.DataFrame(ecc_data_array, columns=['tic','full_toi_id','case','rho','rho_err','ecc','ecc_err1','ecc_err2','omega','omega_err1','omega_err2', 'rad_pl', 'rad_pl_err', 'rad_st', 'rad_st_err', 'mass_st', 'mass_st_err'])
    ecc_output.to_csv(dir_path +  pl_names[j] + '-ecc_data.csv')





    fig = plt.figure(figsize=(14, 7))
    plt.rc('text', usetex=False)

    plt.plot(x_fold, y[mask] - np.median(tr["mean"]), ".", color='silver', zorder=-1000)#, alpha=0.3)
    plt.plot(-0.11, 0, ".", ms=10, color='silver', zorder=-1000)#, alpha=0.3)

    plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", ms=12, alpha=1.0, zorder=-100) ###### NEW!


    plt.plot(x_fold[isort], samples[0,isort] - np.median(tr["mean"]),color="C1", alpha=0.5, label="Sampled Model", zorder=-1)
    plt.plot(x_fold[isort], samples[1:,isort].T - np.median(tr["mean"]),color="C1", alpha=0.2)


    # Annotate the plot with the planet's period
    txt = "TOI " + str(full_toi_ids[j]) + "\nperiod = {0:.5f} +/- {1:.5f} d".format(
        np.mean(tr["period"][:, j]), np.std(tr["period"][:, j])
    )
    plt.annotate(
        txt,
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=20,
    )

    plt.legend(fontsize=20, loc=4)
    plt.title('Model Fit for ' + candidate_ids[j], fontsize=25, y=1.03)
    plt.xlabel("Time since mid-transit [days]", fontsize=24)
    plt.ylabel("Relative Flux", fontsize=24)
    plt.xlim(-1*bound/2.0, bound/2.0);
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-folded_mod.png')
    plt.close()




    mod = samples[0,:]
    resid = y[mask] - mod
    rms = np.sqrt(np.median(resid ** 2))
    mask00 = np.abs(resid) < 5 * rms



    fig = plt.figure(figsize=[14,5])

    # Overplot the phase binned light curve
    if 4.0*durs_true[j] < 0.5*p[j]:
        bound = 4*durs_true[j]
    else:
        bound = 0.5*p[j]
    bins = np.linspace(-1*bound, bound, 85)
    denom, _ = np.histogram(x_fold[mask00], bins)
    num, _ = np.histogram(x_fold[mask00], bins, weights=resid[mask00] - np.median(tr["mean"]))
    denom[num == 0] = 1.0
    plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, "o", color="C2", ms=10, zorder=1000, alpha=1.0)


    # Plot the folded transit
    x_fold = (x[mask] - t0[j] + 0.5*p[j])%p[j] - 0.5*p[j]
    m = np.abs(x_fold) < durs_true[j] * 2
    #plt.plot(x_fold[m], resid[mask00][m], 'k.', label='data')

    plt.plot(x_fold[m], resid[m], '.', color='silver')
    plt.plot(-0.11, 0, '.', ms=10, color='silver')
    
    plt.title("Model Residuals for " + candidate_ids[j], fontsize=25, y=1.03)
    plt.xlabel("Time since mid-transit [days]", fontsize=24)
    plt.ylabel("Residuals of Relative Flux", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(-1*bound/2.0, bound/2.0);
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig(dir_path + pl_names[j] + '-residuals_mod.png')
    plt.close()
    


#################################################



summary = pm.summary(tr, var_names=varnames)
summary.to_csv(dir_path + sys_name + "-summary.csv")
print( summary ) 


tr_fin = pd.read_csv(dir_path + sys_name + '-trace.csv')


if model_mode == 'full':
    for j in range(len(pers_true)):
        mean_f, rp_f, b_f, rhostar_f, t0_f, per_f, u1_f, u2_f, ecc_f, omega_f = [np.median(tr_fin["mean"]), np.median(tr_fin["ror__" + str(j)]), np.median(tr_fin["b__" + str(j)]), np.median(tr_fin["rho_star__" + str(j)]), np.median(tr_fin["t0__" + str(j)]), np.median(tr_fin["period__" + str(j)]), np.median(tr_fin["u__0"]), np.median(tr_fin["u__1"]), np.median(tr_fin["ecc__" + str(j)]), np.median(tr_fin["omega__" + str(j)])]
        mean_f_err1, rp_f_err1, b_f_err1, rhostar_f_err1, t0_f_err1, per_f_err1, u1_f_err1, u2_f_err1, ecc_f_err1, omega_f_err1  = [np.percentile(q=[15.865], a=tr_fin["mean"])[0], np.percentile(q=[15.865], a=tr_fin["ror__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["b__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["rho_star__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["t0__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["period__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["u__0"])[0], np.percentile(q=[15.865], a=tr_fin["u__1"])[0], np.percentile(q=[15.865], a=tr_fin["ecc__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["omega__" + str(j)])[0]]
        mean_f_err2, rp_f_err2, b_f_err2, rhostar_f_err2, t0_f_err2, per_f_err2, u1_f_err2, u2_f_err2, ecc_f_err2, omega_f_err2  = [np.percentile(q=[84.135], a=tr_fin["mean"])[0], np.percentile(q=[84.135], a=tr_fin["ror__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["b__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["rho_star__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["t0__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["period__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["u__0"])[0], np.percentile(q=[84.135], a=tr_fin["u__1"])[0], np.percentile(q=[84.135], a=tr_fin["ecc__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["omega__" + str(j)])[0]]
        mean_f_u, rp_f_u, b_f_u, rhostar_f_u, t0_f_u, per_f_u, u1_f_u, u2_f_u, ecc_f_u, omega_f_u  = ['','rstar','','g/cc','days','days','', '', '', 'radians']

        outputs_all = [['period', per_f_u, per_f, per_f_err2-per_f, per_f_err1-per_f],
                       ['t0', t0_f_u, t0_f, t0_f_err2-t0_f, t0_f_err1-t0_f],
                       ['ror', rp_f_u, rp_f, rp_f_err2-rp_f, rp_f_err1-rp_f],
                       ['rhostar', rhostar_f_u, rhostar_f, rhostar_f_err2-rhostar_f, rhostar_f_err1-rhostar_f],
                       ['b', b_f_u, b_f, b_f_err2-b_f, b_f_err1-b_f],
                       ['mean', mean_f_u, mean_f, mean_f_err2-mean_f, mean_f_err1-mean_f],
                       ['u1', u1_f_u, u1_f, u1_f_err2-u1_f, u1_f_err1-u1_f],
                       ['u2', u2_f_u, u2_f, u2_f_err2-u2_f, u2_f_err1-u2_f],
                       ['ecc', ecc_f_u, ecc_f, ecc_f_err2-ecc_f, ecc_f_err1-ecc_f],
                       ['omega', omega_f_u, omega_f, omega_f_err2-omega_f, omega_f_err1-omega_f]]

        ecc_output = pd.DataFrame(outputs_all, columns=['parameter','units','value','error','error_lower'])
        ecc_output.to_csv(dir_path + pl_names[j] + '-outputs.csv')

elif model_mode == 'duration':
    for j in range(len(pers_true)):
        mean_f, rp_f, b_f, rhostar_f, t0_f, per_f, u1_f, u2_f, dur_f = [np.median(tr_fin["mean"]), np.median(tr_fin["ror__" + str(j)]), np.median(tr_fin["b__" + str(j)]), np.median(tr_fin["rho_circ__" + str(j)]), np.median(tr_fin["t0__" + str(j)]), np.median(tr_fin["period__" + str(j)]), np.median(tr_fin["u__0"]), np.median(tr_fin["u__1"]), np.median(tr_fin["dur__" + str(j)])]
        mean_f_err1, rp_f_err1, b_f_err1, rhostar_f_err1, t0_f_err1, per_f_err1, u1_f_err1, u2_f_err1, dur_f_err1  = [np.percentile(q=[15.865], a=tr_fin["mean"])[0], np.percentile(q=[15.865], a=tr_fin["ror__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["b__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["rho_circ__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["t0__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["period__" + str(j)])[0], np.percentile(q=[15.865], a=tr_fin["u__0"])[0], np.percentile(q=[15.865], a=tr_fin["u__1"])[0], np.percentile(q=[15.865], a=tr_fin["dur__" + str(j)])[0]]
        mean_f_err2, rp_f_err2, b_f_err2, rhostar_f_err2, t0_f_err2, per_f_err2, u1_f_err2, u2_f_err2, dur_f_err2  = [np.percentile(q=[84.135], a=tr_fin["mean"])[0], np.percentile(q=[84.135], a=tr_fin["ror__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["b__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["rho_circ__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["t0__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["period__" + str(j)])[0], np.percentile(q=[84.135], a=tr_fin["u__0"])[0], np.percentile(q=[84.135], a=tr_fin["u__1"])[0], np.percentile(q=[84.135], a=tr_fin["dur__" + str(j)])[0]]
        mean_f_u, rp_f_u, b_f_u, rhostar_f_u, t0_f_u, per_f_u, u1_f_u, u2_f_u, dur_f_u  = ['','rstar','','g/cc','days','days','', '', 'days']

        outputs_all = [['period', per_f_u, per_f, per_f_err2-per_f, per_f_err1-per_f],
                       ['t0', t0_f_u, t0_f, t0_f_err2-t0_f, t0_f_err1-t0_f],
                       ['ror', rp_f_u, rp_f, rp_f_err2-rp_f, rp_f_err1-rp_f],
                       ['rhocirc', rhostar_f_u, rhostar_f, rhostar_f_err2-rhostar_f, rhostar_f_err1-rhostar_f],
                       ['b', b_f_u, b_f, b_f_err2-b_f, b_f_err1-b_f],
                       ['mean', mean_f_u, mean_f, mean_f_err2-mean_f, mean_f_err1-mean_f],
                       ['u1', u1_f_u, u1_f, u1_f_err2-u1_f, u1_f_err1-u1_f],
                       ['u2', u2_f_u, u2_f, u2_f_err2-u2_f, u2_f_err1-u2_f],
                       ['dur', dur_f_u, dur_f, dur_f_err2-dur_f, dur_f_err1-dur_f]]

        ecc_output = pd.DataFrame(outputs_all, columns=['parameter','units','value','error','error_lower'])
        ecc_output.to_csv(dir_path + pl_names[j] + '-outputs.csv')



    for j in range(len(full_toi_ids)):
        
        ecc_data_all = pd.read_csv(dir_path + pl_names[j] + "-ecc_data.csv")
        case_names = list(ecc_data_all["case"])
        rho_vals = list(ecc_data_all["rho"])
        rho_errs = list(ecc_data_all["rho_err"])
        
        for n in range(len(rho_vals)):
            if rho_vals[n] > 0 and rho_errs[n] > 0:
                case_name = case_names[n]
                rhostar_case = rho_vals[n]
                rhostar_err_case = rho_errs[n]

                rho_obs = (rhostar_case, rhostar_err_case)

                rho_tr = np.array(list(tr_temp['rho_circ__' + str(j)]))
                b_tr = np.array(list(tr_temp['b__' + str(j)]))
                r_tr = np.array(list(tr_temp['ror__' + str(j)]))

                lower_rho, upper_rho = 0.00001, 250.0
                mu_rho, sigma_rho = rho_obs
                XX = stats.truncnorm(
                    (lower_rho - mu_rho) / sigma_rho, (upper_rho - mu_rho) / sigma_rho, loc=mu_rho, scale=sigma_rho)

                rho_obs_array = XX.rvs(len(rho_tr))

                rho_ratio = rho_tr/rho_obs_array

                b_need = (1+r_tr)*(1-1/rho_ratio**(2/3))**(1/2)

                fig = plt.figure(figsize=[14,7])
                plt.title("Degeneracy test between e and b given " + r"$\rho_{circ}/\rho_{obs}$" + " - " + case_name, fontsize=25, y=1.03)
                plt.hist(b_need, bins=100, label="Circular orbit - " + case_name)
                plt.hist(b_tr, bins=100, alpha=0.8, label='Modeled orbit')
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.legend(fontsize=20)
                plt.xlabel("impact parameter (b)", fontsize=24)
                plt.ylabel("Count density", fontsize=24)
                fig.savefig(dir_path + pl_names[j] + '-e_b-degeneracy-' + case_name + '.png')
                plt.close()

print(sys_name, ' - Complete!')

