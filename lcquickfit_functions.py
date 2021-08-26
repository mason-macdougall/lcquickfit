
import sys
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


def e_w_function(input_list):
    # Takes input list in the form of [secosw, sesinw]
    # Outputs f(e,w) = rho_circ/rho_obs
    # If result is complex or invalid, outputs -np.inf as result
    
    e = input_list[0]**2 + input_list[1]**2
    ratio = (1+(e**(1/2))*input_list[1])**3/(1-e**2)**(3/2)
    if type(ratio) == complex or str(ratio) == 'nan':
        ratio = -np.inf 
    return ratio



def compile_target_parameters(full_toi_ids, all_data, host_tic, host_toi, lc_path):
    tess_pl = all_data

    pls = []
    pl_ids = []
    pl_names = []
    candidate_ids = []
    true_vals_all = []


    for full_toi_id in full_toi_ids:
        
        pl = all_data.loc[all_data["full_toi_id"]==full_toi_id].index[0]


        full_toi_id = str(full_toi_id)


        pl_num = int(full_toi_id.split('.')[1])
        letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        pl_id = letters[pl_num - 1]

        # General ID string
        candidate_id = 'TIC ' + host_tic + pl_id + ' / TOI ' + host_toi + pl_id


        ########################################################################


        pl_name = 'TOI' + host_toi + pl_id + '_TIC' + host_tic + pl_id
        sys_name = 'TOI' + host_toi + '_TIC' + host_tic


        print('\n' + 'Beginning process for: ' + candidate_id)
        print("idx = " + str(pl) + '\n')


        dec_true = float(tess_pl["dec"][pl])
        ra_true = float(tess_pl["ra"][pl])
        tmag_true = float(tess_pl["t_mag"][pl])
        vmag_true = float(tess_pl["v_mag"][pl])

        t0_true = float(tess_pl["epoch"][pl])
        t0_err_true = float(tess_pl["epoch_err"][pl])
        per_true = float(tess_pl["period"][pl])
        per_err_true = float(tess_pl["period_err"][pl])

        dur_true = float(tess_pl["dur"][pl]) / 24.0
        dur_err_true = float(tess_pl["dur_err"][pl]) / 24.0
        dep_true = float(tess_pl["depth"][pl]) * 1e-6
        dep_err_true = float(tess_pl["depth_err"][pl]) * 1e-6

        print(t0_true, t0_err_true)

        rp_true = float(tess_pl["rp_exp"][pl])
        rp_err_true = float(tess_pl["rp_exp_err"][pl])

        evol_true = tess_pl["evol"][pl]
        disp_true = tess_pl["disp"][pl]
        snr_true = float(tess_pl["snr"][pl])

        sources_st = ['iso_teff', 'iso_color', 'tic']
        
        if float(tess_pl["rad-iso_teff"][pl]) > 0:
            source_st = 0
            rstar_true = float(tess_pl["rad-iso_teff"][pl])
            rstar_err_true = np.mean([float(tess_pl["rad_err1-iso_teff"][pl]), -float(tess_pl["rad_err2-iso_teff"][pl])])

            mstar_true = float(tess_pl["mass-iso_teff"][pl])
            mstar_err_true = np.mean([float(tess_pl["mass_err1-iso_teff"][pl]), -float(tess_pl["mass_err2-iso_teff"][pl])])

            teff_true = float(tess_pl["teff-iso_teff"][pl])
            teff_err_true = np.mean([float(tess_pl["teff_err1-iso_teff"][pl]), -float(tess_pl["teff_err2-iso_teff"][pl])])

            rhostar_true = float(tess_pl["rho-iso_teff"][pl])
            rhostar_err_true = np.mean([float(tess_pl["rho_err1-iso_teff"][pl]), -float(tess_pl["rho_err2-iso_teff"][pl])])

            feh_true = float(tess_pl["feh-iso_teff"][pl])
            feh_err_true = np.mean([float(tess_pl["feh_err1-iso_teff"][pl]), -float(tess_pl["feh_err2-iso_teff"][pl])])

            logg_true = float(tess_pl["logg-iso_teff"][pl])
            logg_err_true = np.mean([float(tess_pl["logg_err1-iso_teff"][pl]), -float(tess_pl["logg_err2-iso_teff"][pl])])

        elif float(tess_pl["rad-iso_color"][pl]) > 0:
            source_st = 1
            rstar_true = float(tess_pl["rad-iso_color"][pl])
            rstar_err_true = np.mean([float(tess_pl["rad_err1-iso_color"][pl]), -float(tess_pl["rad_err2-iso_color"][pl])])

            mstar_true = float(tess_pl["mass-iso_color"][pl])
            mstar_err_true = np.mean([float(tess_pl["mass_err1-iso_color"][pl]), -float(tess_pl["mass_err2-iso_color"][pl])])

            teff_true = float(tess_pl["teff-iso_color"][pl])
            teff_err_true = np.mean([float(tess_pl["teff_err1-iso_color"][pl]), -float(tess_pl["teff_err2-iso_color"][pl])])

            rhostar_true = float(tess_pl["rho-iso_color"][pl])
            rhostar_err_true = np.mean([float(tess_pl["rho_err1-iso_color"][pl]), -float(tess_pl["rho_err2-iso_color"][pl])])

            feh_true = float(tess_pl["feh-iso_color"][pl])
            feh_err_true = np.mean([float(tess_pl["feh_err1-iso_color"][pl]), -float(tess_pl["feh_err2-iso_color"][pl])])

            logg_true = float(tess_pl["logg-iso_color"][pl])
            logg_err_true = np.mean([float(tess_pl["logg_err1-iso_color"][pl]), -float(tess_pl["logg_err2-iso_color"][pl])])
        else:
            source_st = 2
            rstar_true = float(tess_pl["tic_rs"][pl])
            rstar_err_true = float(tess_pl["tic_rs_err"][pl])

            mstar_true = float(tess_pl["tic_ms"][pl])
            mstar_err_true = float(tess_pl["tic_ms_err"][pl])

            teff_true = float(tess_pl["tic_teff"][pl])
            teff_err_true = float(tess_pl["tic_teff_err"][pl])

            mstar_true_u = ufloat(mstar_true, mstar_err_true)
            rstar_true_u = ufloat(rstar_true, rstar_err_true)
            rhostar_true_u = mstar_true_u / (4./3. * np.pi * rstar_true_u**3)
            rhostar_true = float(rhostar_true_u.n)
            rhostar_err_true = float(rhostar_true_u.s)

            feh_true = float(tess_pl["tic_feh"][pl])
            feh_err_true = float(tess_pl["tic_feh_err"][pl])

            logg_true = float(tess_pl["tic_logg"][pl])
            logg_err_true = float(tess_pl["tic_logg_err"][pl])

        rhostar_case_u = ufloat(rhostar_true, rhostar_err_true)          # rho_sun (CHECK UNITS OF INPUT SOURCE)

        rhostar_case_unc = rhostar_case_u * rhosun_u
        rhostar_case = float(rhostar_case_unc.n)
        rhostar_err_case = float(rhostar_case_unc.s)             # g / cm^3

        rhostar_true = rhostar_case
        rhostar_err_true = rhostar_err_case
        rhostar_true_err = rhostar_err_true


        terminate = []

        if float(dec_true) < -20 or float(dec_true) == -99.0 or str(dec_true) == 'nan':
            print('\nDeclination issue: dec_true < -20 or no dec_true value\nDeclination value: ' + str(dec_true))
            terminate.append(1)

        if float(vmag_true) > 13:
            print('\nVmag issue: vmag_true > 13\nVmag value: ' + str(vmag_true))
            terminate.append(2)

        elif float(vmag_true) == -99 and (float(tmag_true) > 13 or float(tmag_true) == -99):
            print('\nTmag issue: vmag_true == -99 and (tmag_true > 13 or tmag_true == -99)\nTmag value: ' + str(tmag_true))
            terminate.append(2)

        if float(per_true) == -99 or float(per_err_true) == -99:
            print('\nPeriod issue: no per_true or no per_err_true\nPeriod value: ' + str(per_true) + ' [' + str(per_err_true) + ']')
            terminate.append(3)

        elif float(t0_true) == -99:
            print('\nt0 issue: t0_true == -99\nt0 value: ' + str(t0_true))
            terminate.append(3)

        elif float(dur_true) == -99:
            print('\nDuration issue: dur_true == -99\nDuration value: ' + str(dur_true))
            terminate.append(3)

        elif float(dep_true) == -99:
            print('\nDepth issue: dep_true == -99\nDepth value: ' + str(dep_true))
            terminate.append(3)

        if float(snr_true) < 10.0:
            print('\nSNR Issue: snr_true < 10.0\nSNR value: ' + str(snr_true))
            terminate.append(4)

        if 'B' in str(disp_true):
            print('\nSuspected binary of some sort\nDisposition: ' + str(disp_true))
            terminate.append(5)

        if 'K' in str(disp_true):
            print('\nKnown planet\nDisposition: ' + str(disp_true))
            terminate.append(6)

        if str(evol_true) != 'MS':
            print('\nStellar evolution issue\nEvolutionary phase: ' + str(evol_true))
            terminate.append(7)

        #if float(neighbors) != 0:
        #    print('\nNeighbors issue: neighbors != 0\nNumber of neighbors: ' + str(neighbors))
        #    terminate.append(8)

        

        if float(dep_true) != -99 and float(dep_err_true) != -99:
            dep_true_u = ufloat(float(dep_true), np.abs(float(dep_err_true)))
            rp_rs_true_unc = dep_true_u**(1/2)
            rp_rs_true = float(rp_rs_true_unc.n)
            rp_rs_err_true = float(rp_rs_true_unc.s)
        elif float(rp_true) != -99 and float(rp_err_true) != -99 and float(rstar_true) != -99 and float(rstar_err_true) != -99:
            rp_true_u = ufloat(rp_true, rp_err_true)
            rstar_true_u = ufloat(rstar_true, rstar_err_true)
            rp_rs_true_unc = rp_true_u / rstar_true_u
            rp_rs_true = float(rp_rs_true_unc.n)
            rp_rs_err_true = float(rp_rs_true_unc.s)
        else:
            print('\nRatio of planet-to-star radius cannot be computed!')
            rp_rs_true = -99
            rp_rs_err_true = -99
            terminate.append(8)

            
            
        if len(terminate) == 0:
            terminate.append(0)

        print(terminate)

        termination_code = ''
        for t in terminate:
            termination_code += str(t)

        #print(termination_code)


        ####### LDTK Analysis ########
        teff_err_u = teff_err_true

        if teff_err_u > 0:
            teff_u = teff_true
            logg_u = logg_true
            feh_u = feh_true
            
            #teff_err_u = 10.0
            logg_err_u = logg_err_true
            feh_err_u = feh_err_true
            
            print(teff_u, teff_err_u, logg_u, logg_err_u, feh_u, feh_err_u)
            
            filters_u = [tess]

            sc_u = LDPSetCreator(teff=(teff_u, teff_err_u), logg=(logg_u, logg_err_u), z=(feh_u, feh_err_u), filters=filters_u) #, cache='/u/scratch/m/macdouga/.ldtk/cache')

            ps_u = sc_u.create_profiles(nsamples=2000)
            ps_u.resample_linear_z(300)

            us, us_err = ps_u.coeffs_qd(do_mc=True, n_mc_samples=10000)
            us = us[0]
            us_err = us_err[0]
            print('Original u, v errors:', us_err)
            us_err = np.array([0.05, 0.05])
        else:
            us = np.array([0.4, 0.1])
            us_err = np.array([1.0, 1.0])
            terminate.append(9)

        print(us)
        print(us_err)

        true_vals = [dec_true, ra_true, tmag_true, vmag_true, t0_true, t0_err_true, per_true, per_err_true,
        dur_true, dur_err_true, dep_true, dep_err_true, rstar_true, rstar_err_true,
        rp_true, rp_err_true, teff_true, teff_err_true,
        rhostar_true, rhostar_err_true, 0, evol_true, 0, snr_true, disp_true, mstar_true, us, us_err, rp_rs_true, rp_rs_err_true]

        print(true_vals)


        print('\n' + 'Compiling data for: ' + candidate_id)


        param0 = ['period', 't0', 'duration', 'rp', 'rstar', 'rhostar', 'u1', 'u2', 'flags', 'st_source']
        units0 = ['days', 'days', 'days', 'rstar', 'rsun', 'g/cc', '', '', '', '']
        value0 = [per_true, t0_true, dur_true, rp_rs_true, rstar_true, rhostar_true, us[0], us[1], int(termination_code), sources_st]
        error0 = [per_err_true, t0_err_true, dur_err_true, rp_rs_err_true, rstar_err_true, rhostar_err_true, us_err[0], us_err[1], np.nan, np.nan]
        
        inputs = pd.DataFrame(columns = ['parameter', 'units', 'value', 'error'])
        inputs['parameter'] = param0
        inputs['units'] = units0
        inputs['value'] = value0
        inputs['error'] = error0
        inputs.to_csv(lc_path + pl_name + '-inputs.csv')
        
        
        # List of termination codes to only save input data
        tks_bad = [1, 2, 3, 5, 8]

        # List of termination codes to save input data, LC, and create folders
        tks_good = [0, 4, 6, 7, 9]

        for t in terminate:
            if t in tks_bad:
                tks_flag = 0
                break
            elif t in tks_good:
                tks_flag = 1

        #print(tks_flag)

        
        print(full_toi_id)
        print("\nPlanet Params + Errors")
        print("t0 (days), per (days), dur(days), rp (rstar)")
        print(t0_true, per_true, dur_true, rp_rs_true)
        print(t0_err_true, per_err_true, dur_err_true, rp_rs_err_true)
        
        print("\nStar Params + Errors")
        print("rstar (R_sun), rhostar (g/cc)")
        print(rstar_true, rhostar_true)
        print(rstar_err_true, rhostar_err_true)
        
        
        if tks_flag == 0: #3 in terminate or 8 in terminate:
            continue
        
        pls.append(pl)
        pl_ids.append(pl_id)
        pl_names.append(pl_name)
        candidate_ids.append(candidate_id)
        true_vals_all.append(true_vals)

        

    if len(pl_ids) == 0:
        sys.exit("\nNo planets in system TIC " + str(host_tic) + " meet requirements. Ending process...")
    else:
        return pls, pl_ids, pl_names, candidate_ids, true_vals_all


def get_lc(lc_path, host_tic, host_toi):
    sys_name = 'TOI' + host_toi + '_TIC' + host_tic

    lc_fname = sys_name + '_lc.fits'

    print(sys_name)
    print(lc_fname)
    print(lc_path)

    try:
        f = fits.open(lc_path + lc_fname)
        f.close()
        print('\nAlready saved!')

        lc_file = fits.open(lc_path + lc_fname)

        with lc_file as hdu:
            lc = hdu[1].data

        lc_file.close()   

    except FileNotFoundError:

        fail = 0

        print('\nSaving...')

        lc_download = lk.search_lightcurvefile(target="TIC " + host_tic, mission="TESS", author="SPOC").download_all(flux_column="pdcsap_flux")
        if str(type(lc_download)) == '<class \'NoneType\'>':
            print('not 1')
            lc_download = lk.search_lightcurvefile(target="TOI " + host_toi, mission="TESS", author="SPOC").download_all(flux_column="pdcsap_flux")
            if str(type(lc_download)) == '<class \'NoneType\'>':
                print('not 2')
                fail = 1

        if fail == 0:

            lc_full = lc_download.stitch().remove_nans().remove_outliers(sigma_lower=100,sigma_upper=5)
            lc_full.to_fits(lc_path + lc_fname)

            lc_file = fits.open(lc_path + lc_fname)

            with lc_file as hdu:
                lc = hdu[1].data

            lc_file.close()


        elif fail == 1:
            print('\nFile not found! Target: ' + sys_name)
            sys.exit()

    return lc



def plot_light_curve(x, y, yerr, soln, gp_pred, mask=None, g=-1, spread=0, ylim=[], idx=''):

    figs = []
    
    values = range(len(soln['t0']))
    
    for j in values:
        
        if spread == 0:
            spread = durs_true[j] * 2
        
        if mask is None:
            mask = np.ones(len(x), dtype=bool)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        plt.rc('text', usetex=False)
        ax = axes[0]
        ax.plot(x[mask], y[mask], "k.")
        gp_mod = gp_pred + soln["mean"]
        ax.plot(x[mask], gp_mod, color="C2", label="GP Model")
        ax.legend(fontsize=10, loc=4)
        ax.set_ylabel("Relative Flux")

        if ylim != [] and g > -1:
            ax.set_ylim(np.array(ylim))

        ax = axes[1]
        ax.plot(x[mask], y[mask] - gp_mod, "k.")
                   
        letters = ["b", "c", "d", "e", "f", "g"]
        planets = ''
        for jj in values:
            planets += letters[jj]
            
        for i, l in enumerate(planets):
            mod = soln["light_curves"][:, i]
            ax.plot(
                x[mask], mod, lw=1, label="planet {0}".format(l)
            )
            
        ax.legend(fontsize=10, loc=4)
        ax.set_ylabel("De-trended Relative Flux")

        if ylim != [] and g > -1:
            ax.set_ylim(np.array(ylim))

        ax = axes[2]
        mod = gp_mod + np.sum(soln["light_curves"], axis=-1)
        ax.plot(x[mask], y[mask] - mod, "k.")
        ax.axhline(0, color="#aaaaaa", lw=1)
        ax.set_ylabel("Residuals of Relative Flux")
        ax.set_xlim(x[mask].min(), x[mask].max())
        ax.set_xlabel("Time [days]")

        if g > -1:
            ax.set_xlim(np.array([soln["t0"][j]+(soln["period"][j]*g)-spread, soln["t0"][j]+(soln["period"][j]*g)+spread]))

        if ylim != [] and g > -1:
            ax.set_ylim(np.array(ylim))

        if g == -1:
            name_label = sys_name
        else:
            name_label = pl_names[j]

        fig.savefig(dir_path + name_label + '-gp_mod' + str(idx) + '-mod.png')
        
        if g == -1 or len(values) == 1:
            return fig

        figs.append(fig)

        plt.show()

        #plt.close()

    return figs;


def ecos_esin_interp(rho_weight_df, rho_data_df, path, target, label):
    emin, emax, esamp = 0,0.99,99
    omegamin, omegamax, omegasamp = -0.5*np.pi,1.5*np.pi,100
    ecc00 = np.linspace(emin,emax,esamp)
    omega00 = np.linspace(omegamin,omegamax,omegasamp)

    rho_ratio = np.array(list(rho_data_df["rho_ratio"]))
    rho_ratios_in = np.array(list(rho_weight_df["rho_ratio"]))
    weights_in = np.array(list(rho_weight_df["weight"]))
    logprobs_in = np.array(list(rho_weight_df["logprob"]))

    combos = []
    weights = []

    for ee in ecc00:
        for ww in omega00:
            combos.append([ee,ww])
            rho_ratio = e_w_function([np.sqrt(ee)*np.cos(ww), np.sqrt(ee)*np.sin(ww)])
            weight = np.interp(rho_ratio, rho_ratios_in, weights_in)
            weights.append(weight)

    esinws = []
    ecosws = []   


    finite_weights = []
    for n in weights:
        if n > -np.inf:
            finite_weights.append(n)

    finite_weights = np.array(finite_weights)*50.0/np.max(finite_weights)
    for combo, weight in zip(combos, finite_weights):
        weight = int(weight)
        esinw = combo[0]*np.sin(combo[1])
        ecosw = combo[0]*np.cos(combo[1])
        esinws += [esinw]*weight
        ecosws += [ecosw]*weight


    data_kde = np.array([ecosws, esinws])
    x_kde, y_kde = data_kde
    xnbins_kde = 100
    ynbins_kde = 100


    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    kd_xy = stats.gaussian_kde(data_kde)
    xi_kde_lin = np.linspace(x_kde.min(),x_kde.max(),xnbins_kde)
    yi_kde_lin = np.linspace(y_kde.min(),y_kde.max(),ynbins_kde)

    xi_kde, yi_kde = np.meshgrid(xi_kde_lin, yi_kde_lin,indexing='ij')

    zi_kde = kd_xy(np.vstack([xi_kde.flatten(), yi_kde.flatten()]))

    c = ChainConsumer()
    #c.configure(usetex=False) 
    plt.rc('text', usetex=False)

    weights_fin = zi_kde.reshape((len(xi_kde_lin),len(yi_kde_lin)))
    c.add_chain([xi_kde_lin,yi_kde_lin],grid=True,weights=weights_fin,kde=True,smooth=True,parameters=['$e cos(\omega)$','$e sin(\omega)$'])
    c.configure(usetex=False, label_font_size=20, tick_font_size=16)
    fig = c.plotter.plot(figsize=[8,8])

    # Annotate the plot with the planet's period
    txt = "TOI " + str(full_toi_ids[j]) + " (" + case + ")"
    plt.annotate(
        txt,
        (0, 0),
        xycoords="axes fraction",
        xytext=(105, -1),
        rotation=270,
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=18,
        annotation_clip=False
    )

    #fig.set_size_inches(6.2 + fig.get_size_inches())  
    plt.gcf().subplots_adjust(bottom=0.2, right=0.85)
    plt.tight_layout()
    fig.savefig(path + target + '-ecosw_esinw-' + label + '.png')
    #plt.close()

    post = pd.DataFrame(weights_fin)
    post.to_csv(path + target + '-ecosw_esinw_prob-' + label + '.csv')
