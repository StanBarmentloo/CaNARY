import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import scipy.signal as signal
import ast
import re
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
# Below we have routines related to reading in spectra

def read_observed_spectra():
    '''Outputs a list of all observed spectra of interest. The list is sorted by SN types, as well as SNe.
       Furthermore, a list of the spectra names is returned.'''
    
    base_path = '/home/stba7609/Documents/Work/Papers/2024_PaperII'
    observations_path = base_path + '/cleaned_observations/Optical_sample/'

    Ib_names = np.array(['SN1985F', 'SN1996aq', 'SN2004ao', 'SN2004dk', 'SN2004gq', 'SN2006ld', 'SN2007C',
                     'SN2007Y', 'SN2008D', 'SN2009jf', 'SN2012au', 'SN2015ah', 
                     'SNMASTEROTJ1204', 'SN2019odp'])

    Ic_names = np.array(['SN1987M', 'SN1997dq', 'SN1998bw', 'SN2003gf', 'SN2004aw', 'SN2007gr', 'SN2007I', 'SN2011bm', 
                    'SNPTF12gzk', 'SN2013ge', 'SN2014eh', 'SN2019yz'])

    IIb_names = np.array(['SN1993J', 'SN1996cb', 'SN2001ig', 'SN2003bg',
                     'SN2008ax', 'SN2008bo', 'SN2011dh', 'SN2011ei', 'SN2011hs',
                     'SNPTF12os', 'iPTF13bvn', 'SN2013df', 'SNASASSN14az', 'SN2016gkg', 'SN2020acat', 'SN2022crv'])

    name_list_by_type = [Ib_names, Ic_names, IIb_names]
    sn_types = ['Type_Ib', 'Type_Ic', 'Type_IIb']
    
    spectra_list, fname_list = [], []
    
    for z in range(len(sn_types)): # For each SN type...
        
        sn_names = name_list_by_type[z]
        sn_type = sn_types[z]

        observations = os.listdir(path = observations_path + sn_type + '/')

        for i in range(len(sn_names)): #For each individual supernova of this type

            #Read in the specific SN folder
            sn_name = sn_names[i]

            this_sn_fnames = []
            for j in range(len(observations)):
                if sn_name in observations[j]:
                    this_sn_fnames.append(observations[j])

            this_sn_fnames = sorted(this_sn_fnames) #These are all files of this particular SN

            for k in range(len(this_sn_fnames)): #For each spectrum of this particular SN

                this_sn_fname = this_sn_fnames[k]
                this_observation_data = np.loadtxt(observations_path + sn_type + '/' + this_sn_fname, delimiter = ',')
                
                spectra_list.append(this_observation_data)
                fname_list.append(this_sn_fname)
    
    
    return spectra_list, fname_list
    
    
    
    
def read_model_spectra():
    
    base_path = '/home/stba7609/Documents/Work/Papers/2024_PaperII'
    model_path = base_path + '/SUMO_output_folders/FINAL_SUMO/SUMO_Projects_chi30/'


    epoch_list = ['150d/', '200d/', '250d/', '300d/', '350d/', '400d/']

    name_list = ['NII_he3p3_AJ_40_60', 'NII_he4p0_AJ_18_82', 'NII_he6p0_AJ_10_90',
                        'NII_he8p0_AJ_10_90']
    
    spectra_list, fname_list = [], []
    
    for z in range(len(name_list)): # For each model
    
        for k in range(len(epoch_list)): # For each epoch

            this_spec_fname = model_path + epoch_list[k] + name_list[z] + '/out/modelspectra/spectrum.datrun001'

            this_observation_data = np.loadtxt(this_spec_fname, skiprows = 1)
            
            
            spectra_list.append(this_observation_data)
            fname_list.append(name_list[z] + '/' + epoch_list[k])
            
    return spectra_list, fname_list

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
# Below we have routines related to getting the actual CI 8727 fits


def constrain_parameter_space(wl, flux, epoch):
    
    print("Below, four typical cases of the Ca NIR triplet in observations are shown")
    
    #plot_typical_cases()
    
    print("Considering these cases, answer the following questions")
    
    MgI_answer = input("Is there any sign of Mg I 8806? [y/n]")
    
    OI_answer = input("Is there any sign of O I 8446? [y/n]")
    
    print("""Here you can constrain v_out. Below are fits of a few spectral lines,
          and their respective velocity estimates. Based on this, write a velocity range
          in the format [xxxx,xxxx] (NOTE: the maximum range is [3000,7000])""")
    
    plot_velocity_fits(wl, flux, epoch)
    
    velocity_range = ast.literal_eval(input())
    
    while velocity_range[0] > 7000:
        print('You entered a wrong velocity range, try again')
        
        velocity_range = ast.literal_eval(input())
        
    print("Finally, give the bounds of where the Ca NIR complex starts and ends (i.e. differ from continuum)")
    print("Be as specific as possible! Answer in the format [xxxx,xxxx]")

    line_complex_bounds = ast.literal_eval(input())

    print("Thank you. We now start the fitting process!")
    
    answer_list = [MgI_answer, OI_answer, velocity_range, line_complex_bounds] 
    
    return answer_list


def check_constraints(constraints_list, line_profile_name):
    
    valid = True
    
    if constraints_list[0] == 'n': # There is no sign of MgI 8806
        if ('pMgI_f00p00' in line_profile_name) == False:
            valid = False
        
    if constraints_list[1] == 'n': # There is no sign of OI
        if ('pOI_f00p00' in line_profile_name) == False:
            valid = False
        
    v_low, v_high = constraints_list[2][0], constraints_list[2][1]
    
    v_file = int(re.search(r'vout_(\d+)', line_profile_name).group(1))
    
    if v_file < v_low or v_file > v_high: # Is the profile velocity within the allowed range?
        valid = False
    
    return valid

def calculate_score(observed_wl, observed_flux, simulated_wl, simulated_flux, line_complex_bounds):
    
    # Assure that we evaluate the fluxes at the same wavelengths by interpolating the simulated data
    simulated_flux_interpolated = np.interp(observed_wl, simulated_wl, simulated_flux)
    
    # We want to not overestimate our chi2 with the continuum, as we do not model it.
    
    zoom_mask = (observed_wl > line_complex_bounds[0]) * (observed_wl < line_complex_bounds[1])
    residuals = observed_flux[zoom_mask] - simulated_flux_interpolated[zoom_mask]
    
    score = np.sum(residuals**2)
    
    return score

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
# Below we have routines related to data preprocessing

def rebin_spectrum(wl, flux):
    
    delta_wl = wl[1] - wl[0]
    bin_factor = int(np.ceil(10/delta_wl))# This gives bins of roughly 10 Å
    if bin_factor < 1:
        bin_factor = 1
        return wl, flux
    else:
        new_wl, new_flux = [], []
        for i in range(0, len(wl), bin_factor):
            new_wl.append(np.mean(wl[i:i+bin_factor]))
            new_flux.append(np.mean(flux[i:i+bin_factor]))

        return np.array(new_wl), np.array(new_flux)
    
def remove_continuum_flux(wl, flux, wl_left1 = 8200, wl_right1 = 8300, wl_left2 = 9000, wl_right2 = 9100):
    
    
    # Get the median between 8200-8300 Å and 9000-9100 Å
    flux_left, flux_right = flux[(wl > wl_left1) * (wl < wl_right1)], flux[(wl > wl_left2) * (wl < wl_right2)]
    left_median, right_median = np.median(flux_left), np.median(flux_right)
    
    average_continuum = (left_median + right_median)/2
    flux -= average_continuum # Remove a flat continuum
    
    adjusted_total_flux = integrate.cumulative_trapezoid(flux, wl)[-1] # We need this number to determine the absolute CI flux later
     
    flux = flux/np.max(flux) # Normalise the peak in this region to 1
    
    return flux, adjusted_total_flux
    
    
    
def extract_CI_fraction(line_profile_fnames):
    
    CI_fractions = np.zeros(len(line_profile_fnames))
    
    for i in range(len(line_profile_fnames)):
        
        fname = line_profile_fnames[i]
        
        el_probabilities_string = re.findall(r'\d+p\d+', fname)  # Find all occurrences of 'xxpxx' format
        el_probabilities_float = [float(m.replace('p', '.')) for m in el_probabilities_string]  # Convert to floats
        
        P_CI = el_probabilities_float[1] # It is always the second probability
        P_norm = sum(el_probabilities_float) + 10**-3 # Avoid division by 0 errors
        
        CI_fractions[i] = P_CI/P_norm
        
    return CI_fractions

def extract_v_out(line_profile_fnames):
    
    vout_array = np.zeros(len(line_profile_fnames))
    
    for i in range(len(line_profile_fnames)):
        
        fname = line_profile_fnames[i]
        
        vout_string = re.search(r'vout_(\d+)', fname)
        vout_float = float(vout_string.group(1))
        
        vout_array[i] = vout_float
        
    return vout_array

def extract_epoch(file_name):
    
    if 'discovery' in file_name:
        return float(file_name[-7:-4]) # The epoch is always at this position
    elif 'peaklight' in file_name:
        return float(file_name[-7:-4]) + 20

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================
# Below we have routines related to post-processing


def obtain_total_continuum(wl, flux):
    #This function obtains the continuum when determining the optical integrated flux of the spectrum.
    #As the continuum is quite varying between sources, this function is somewhat involved
    #We base it on four regions which according to our models typically have no emission lines and therefore little flux
    
    ranges = [[5740, 5790], [6020, 6070], [6850, 6900], [7950, 8000]]
    means, stds = [], []
    for i in range(len(ranges)):
        mask = (wl > ranges[i][0]) * (wl < ranges[i][1])
        means.append(np.mean(flux[mask]))
        
    sorted_means = sorted(means)
    
    #Now pick the three lowest values of these four typically 'continuum-like' regions
    continuum_flux = ( sorted_means[0] + sorted_means[1] + sorted_means[2])/3
    
    return continuum_flux

#============================================================================================================================
#============================================================================================================================
#============================================================================================================================


# Below we have routines related to plotting


def plot_individual_spectrum(wl, flux, wl_min, wl_max):
    
    fig, ax = plt.subplots(2, 1, figsize = (10, 8))
    
    #First plot the entire visible spectrum
    try:
        ax[0].plot(wl, flux/np.max(flux[(wl > 6000) * (wl < 7500)]))
    except:
        ax[0].plot(wl, flux/np.max(flux[(wl > 8300) * (wl < 8900)]))
        
    ax[0].set_xlim(4000, 9000)
    ax[0].set_ylim(0, 1)
    
    #Plot a few lines for redshift
    ax[0].axvline(x = 4861, linestyle = '--', color = 'k', alpha = 0.5)
    ax[0].axvline(x = 6300, linestyle = '--', color = 'k', alpha = 0.5)
    ax[0].axvline(x = 7774, linestyle = '--', color = 'k', alpha = 0.5)
    ax[0].axvline(x = 7291, linestyle = '--', color = 'k', alpha = 0.5)
    
    #=============================================================================================
    
    # Zoom in on region of interest
    zoom_mask = (wl > wl_min) * (wl < wl_max)
    
    # We zoom in. The if statement assures that the zoomed region is present. If it is not, skip this 
    wl, flux = wl[zoom_mask], flux[zoom_mask]
    
    ax[1].plot(wl, flux/np.max(flux), color = 'k') # Plot the observed spectrum
    
    #Plot some lines
    line_list_wavelengths = [8446, 8498, 8542, 8662, 8727, 8774, 8806]
    line_list_names = ['O I', 'Ca NIR 1', 'Ca NIR2', 'Ca NIR3', '[C I]', '?', 'Mg I']
    line_list_colours = ['purple', 'red', 'orange', 'yellow', 'green', 'blue', 'pink']
    
    for i in range(len(line_list_wavelengths)):
        line_wl, line_name, line_colour = line_list_wavelengths[i], line_list_names[i], line_list_colours[i]
        
        ax[1].axvline(x = line_wl, label = line_name, linestyle = '--', color = line_colour)
        
    ax[1].axhline(y = 0, linestyle = '--', alpha = 0.5, color = 'grey')
    
    minor_ticks = np.arange(wl_min, wl_max, 20)
    major_ticks = np.arange(wl_min, wl_max, 100)
    
    ax[1].set_xticks(major_ticks)
    ax[1].set_xticks(minor_ticks, minor=True)
    
    ax[1].tick_params(axis='x', which='major', length=10, labelsize=12)  # Larger for major ticks
    ax[1].tick_params(axis='x', which='minor', length=5, labelsize=8)
    
    ax[1].set_xticklabels(major_ticks)
    ax[1].grid()
        
    ax[1].set_ylim(-0.1, 1)
    ax[1].legend()
    ax[1].set_xlabel('Rest Wavelength [Å]')
    
    plt.show()
    
def plot_typical_cases():
    
    fig, ax = plt.subplots(4, 1, figsize = (6, 12))
    
    case1 = np.loadtxt(os.getcwd() + '/typical_cases/SN1993J_discovery208.csv', delimiter = ',')
    case2 = np.loadtxt(os.getcwd() + '/typical_cases/SN2007C_peaklight155.csv', delimiter = ',')
    case3 = np.loadtxt(os.getcwd() + '/typical_cases/SN2004dk_peaklight264.csv', delimiter = ',')
    case4 = np.loadtxt(os.getcwd() + '/typical_cases/SN2003gf_discovery181.csv', delimiter = ',')
    
    data_list = [case1, case2, case3, case4]
    obs_labels = ['SN 1993J, 208d', 'SN 2007C, 175d', 'SN 2004dk, 284d', 'SN 2003gf, 181d']
    
    line_wls = [8446, 8498, 8542, 8662, 8727, 8806]
    line_colours = ['purple', 'green', 'orange', 'red', 'blue', 'k']
    line_labels = ['O I 8446', 'Ca II 8498', 'Ca II 8542', 'Ca II 8662', '[C I] 8727', 'Mg I 8806']
    
    for i in range(4):
        
        wl, flux = data_list[i][:, 0], data_list[i][:, 1]
        wl, flux = wl[flux > -np.inf], flux[flux > -np.inf] # This step removes any NANs in the spectra
        
        wlmin, wlmax = 8200, 9100
        wl, flux = wl[(wlmin < wl) * (wlmax > wl)], flux[(wlmin < wl) * (wlmax > wl)] # Zoom in
        
        flux, dummy = remove_continuum_flux(wl, flux)
        ax[i].plot(wl, flux, c = 'k', label = obs_labels[i])
        ax[i].legend()
        
        if i == 0: # Separate case to get the line labels
            for j in range(len(line_wls)):
                ax[i].axvline(x = line_wls[j], linestyle = '--', c = line_colours[j], label = line_labels[j], alpha = 0.3, zorder = -1)

        else:
            for j in range(len(line_wls)):
                ax[i].axvline(x = line_wls[j], linestyle = '--', c = line_colours[j], alpha = 0.3, zorder = -1)

        ax[i].set_ylim(-0.1, 1.1)
                
            
            
    fig.text(0.3, 0.02, 'Observed Wavelength [Å]')
    fig.text(0.02, 0.3, 'Normalised Flux', rotation = 90)
    plt.figlegend(bbox_to_anchor=(0.02, 0.92, 1, 0), loc= 'center',
                  borderaxespad = 0, handletextpad = 0.3, ncol=4 ,columnspacing=1.5, frameon= False)
    
    
    plt.show()
    
    
def plot_velocity_fits(observed_wl, observed_flux, epoch):
    
    # Find the set with the closest model epoch
    model_epochs = np.array([150, 200, 250, 300, 350, 400])
    closest_epoch_index = np.argmin(abs(model_epochs-epoch))
    closest_epoch_string = str(model_epochs[closest_epoch_index])
    
    path_to_model_spectra = "/home/stba7609/Documents/Work/Papers/2024_PaperII/SUMO_output_folders/FINAL_SUMO/SUMO_Projects_chi30/"
    model_names = ['NII_he3p3_AJ_40_60', 'NII_he4p0_AJ_18_82', 
                   'NII_he6p0_AJ_10_90', 'NII_he8p0_AJ_10_90']
    
    # Lets plot linewidth comparisons for 3 different lines
    line_regions = [[5725, 6075], [6800, 7500], [6000, 6700]]
    
    fig, ax = plt.subplots(3, 1, figsize = (9, 12))
    
    for k in range(3):
        # First plot the observed spectrum
        
        # Zoom in on the line
        zoom_mask = (observed_wl > line_regions[k][0]) * (observed_wl < line_regions[k][1])
        observed_wl_zoom, observed_flux_zoom = observed_wl[zoom_mask], observed_flux[zoom_mask]
        
        # Remove the 'continuum' and Renormalise to the peak of the line
        wl_left1, wl_right1 = line_regions[k][0], line_regions[k][0] + 50
        wl_left2, wl_right2 = line_regions[k][1] - 50, line_regions[k][1]
        
        observed_flux_corr, dummy = remove_continuum_flux(observed_wl_zoom, observed_flux_zoom, 
                                                     wl_left1, wl_right1,
                                                     wl_left2, wl_right2)
        observed_wl_corr = observed_wl_zoom
        
        # Plot
        if k == 0:
            ax[k].plot(observed_wl_corr, observed_flux_corr, c = 'k', label = 'Observed', linestyle = '--')
        else:
            ax[k].plot(observed_wl_corr, observed_flux_corr, c = 'k', linestyle = '--')
        
    
        # Then the models
        model_colours = ['green', 'orange', 'red', 'blue', 'purple']
        model_labels = ['$V_{\text{out}}$ = 4300 kms', '$V_{\text{out}}$ = 4500 kms',
                       '$V_{\text{out}}$ = 5700 kms', '$V_{\text{out}}$ = 4000 kms']
        
        for i in range(len(model_names)):
            
            model_data = np.loadtxt(path_to_model_spectra + closest_epoch_string + 'd/' + model_names[i] + '/out/modelspectra/spectrum.datrun001', skiprows = 1)
            model_wl, model_flux = model_data[:, 0], model_data[:, 1]
            
            # The same as for the observed spectrum
            zoom_mask = (model_wl > line_regions[k][0]) * (model_wl < line_regions[k][1])
            model_wl_zoom, model_flux_zoom = model_wl[zoom_mask], model_flux[zoom_mask]
            model_wl_bin, model_flux_bin = rebin_spectrum(model_wl_zoom, model_flux_zoom)
            
            model_flux_corr, dummy = remove_continuum_flux(model_wl_bin, model_flux_bin, 
                                                       wl_left1, wl_right1,
                                                       wl_left2, wl_right2)
            model_wl_corr = model_wl_bin
            
            
            
            # Plot
            if k == 0:
                ax[k].plot(model_wl_corr, model_flux_corr, c = model_colours[i], label = model_labels[i], alpha = 0.5, zorder = -1)
            else:
                ax[k].plot(model_wl_corr, model_flux_corr, c = model_colours[i], alpha = 0.5, zorder = -1)
        
        ax[k].set_ylim(-0.1, 1.1)
        
    ax[-1].set_xlabel('Rest wavelength [Å]')
    ax[0].set_ylabel('Normalised flux')
    
    plt.figlegend(bbox_to_anchor=(0.02, 0.92, 1, 0), loc= 'center',
                  borderaxespad = 0, handletextpad = 0.3, ncol=3 ,columnspacing=1.5, frameon= False)

    plt.show()
    
    
    
def plot_best_fit(score_list, observed_wl, observed_flux, line_complex_bounds, pre_computed_path, line_profile_fnames):
    
    #==========================================================================================================
    # Get the best scoring profile
    score_list = np.array(score_list)    
    best_score_index = np.argmin(score_list)
    
    print('This profile had the following parameters: ', line_profile_fnames[best_score_index])

    best_data = np.loadtxt(pre_computed_path + line_profile_fnames[best_score_index], delimiter = ',')
    best_wl, best_flux = best_data[:, 0], best_data[:, 1]
    best_profile_name = line_profile_fnames[best_score_index]
    #==========================================================================================================
    
    #==========================================================================================================
    # Create the actual plot
    fig, ax = plt.subplots(1, figsize = (9, 6))
    
    ax.plot(observed_wl, observed_flux, label = 'Observed Spectrum')
    
    # Assure that we evaluate the fluxes at the same wavelengths by interpolating the simulated data
    best_flux_interpolated = np.interp(observed_wl, best_wl, best_flux)
    ax.plot(observed_wl, best_flux_interpolated, label = 'Best Fit')
    
    line_wls = [8446, 8498, 8542, 8662, 8727, 8806]
    line_colours = ['purple', 'green', 'orange', 'red', 'blue', 'k']
    line_labels = ['O I 8446', 'Ca II 8498', 'Ca II 8542', 'Ca II 8662', '[C I] 8727', 'Mg I 8806']
    for i in range(len(line_wls)):
        ax.axvline(x = line_wls[i], linestyle = '--', c = line_colours[i], label = line_labels[i])
        
    # Plot two lines indicating which part was considered when scoring the fit
    ax.axvline(x = line_complex_bounds[0], linestyle = '--', alpha = 0.5, c = 'olive', label = 'Complex edges')
    ax.axvline(x = line_complex_bounds[1], linestyle = '--', alpha = 0.5, c = 'olive')
        
    ax.set_xlabel('Observed Wavelength [Å]')
    ax.set_ylabel('Normalised Flux')
    ax.set_ylim(bottom = -0.1, top = 1.1)
    
    plt.figlegend(bbox_to_anchor=(0.02, 0.92, 1, 0), loc= 'center',
                  borderaxespad = 0, handletextpad = 0.3, ncol=4 ,columnspacing=1.5, frameon= False)
    
    plt.show()
    #==========================================================================================================
    
    
def plot_chi2_scores(score_list, line_profile_fnames, chi2_threshold, mode = 'CI'):
    
    fig, ax = plt.subplots(1, figsize = (9, 6))
    
    if mode == 'CI':
        
        x_array = extract_CI_fraction(line_profile_fnames)
        x_label = 'fraction of [C I] 8727 photons in spectrum'
    
    elif mode == 'v_out':
        
        x_array = extract_v_out(line_profile_fnames)
        x_label = '$V_{\text{out}}$'
    
    else:
        sys.exit()
        
    normalised_scores = np.array(score_list)/min(score_list)
    
    ax.scatter(x_array, normalised_scores, label = 'scores')
    
    x, y = x_array, normalised_scores
    df = pd.DataFrame({"x": x, "y": y})
    df = df.sort_values(by="x")

    # Define number of bins
    num_bins = 11 # Optimised by eye

    # Bin the x values and find the minimum y in each bin
    bins = np.linspace(df["x"].min(), df["x"].max(), num_bins)
    df["x_bin"] = np.digitize(df["x"], bins)

    # Group by bins and take the minimum y-value in each bin
    lower_envelope = df.groupby("x_bin").agg({"x": "mean", "y": "min"}).dropna()

    # Interpolate the lower envelope for smoothness
    interp_func = interp1d(lower_envelope["x"], lower_envelope["y"], kind="linear", fill_value="extrapolate")

    # Generate smooth x values and compute interpolated y values
    x_smooth = np.linspace(df["x"].min(), df["x"].max(), 300)
    y_smooth = interp_func(x_smooth)
    
    ax.plot(x_smooth, y_smooth)
    ax.legend()
        
        
    ax.set_xlabel(x_label)
    ax.set_ylabel('Normalised $\chi^{2}$')
    
    ax.set_ylim(np.min(normalised_scores), np.min(normalised_scores) + 3)    
    
    plt.show()
    
    best_CI = x_array[np.argmin(normalised_scores)]
    print('In the plot above, the best fit P_CI is: ', '{0:.2f}'.format(best_CI))
    
    left_CI_index = np.where(y_smooth < chi2_threshold * np.min(y_smooth))[0][0]
    right_CI_index = np.where(y_smooth < chi2_threshold * np.min(y_smooth))[0][-1] 
    
    left_CI_bound, right_CI_bound = x_smooth[left_CI_index], x_smooth[right_CI_index]
    
    if right_CI_index == len(x_smooth) - 1: # We can not exclude that chi2 < 2 * chi2_min even at only C I, so just put 1
        right_CI_bound = 1
        
    
    print('The uncertainty range is: ', '{0:.2f}'.format(left_CI_bound), ' to ', '{0:.2f}'.format(right_CI_bound))
    
    return best_CI, left_CI_bound, right_CI_bound