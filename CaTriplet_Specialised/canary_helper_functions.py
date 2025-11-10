import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.random import choice,shuffle
from scipy import integrate

def plot_result(hist_list, histscatt_list, binarr, n_destroyed = 0, n_scatterdeath = 0):
    
    if len(hist_list) == 1:
        print('Collected photons: ', np.sum(hist_list[0]))
        print('Destroyed photons: ', n_destroyed)
        print('Scattering death photons: ', n_scatterdeath)
    
    fig, ax = plt.subplots(1, figsize = (10, 5))
    
    if len(hist_list) == 1:
        norm = np.max(hist_list[0])
        
        ax.plot(binarr, hist_list[0]/norm, label = 'All escaped photons')
        ax.plot(binarr, histscatt_list[0]/norm, label = 'Scattered photons')

    else:
        plot_colours = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        for i in range(np.min( (len(hist_list), len(plot_colours)) )):
            norm = np.max(hist_list[i])
            ax.plot(binarr, hist_list[i]/norm, c = plot_colours[i], label = 'Model ' + str(i+1))
    
    line_wls = [8446, 8498, 8542, 8662, 8727, 8806]
    line_colours = ['purple', 'green', 'orange', 'red', 'blue', 'k']
    line_labels = ['O I 8446', 'Ca II 8498', 'Ca II 8542', 'Ca II 8662', '[C I] 8727', 'Mg I 8806']
    for i in range(len(line_wls)):
        ax.axvline(x = line_wls[i], linestyle = '--', c = line_colours[i], label = line_labels[i], alpha = 0.3, zorder = -1)
        
    ax.set_xlabel('Observed Wavelength [Å]')
    ax.set_ylabel('Observed photons')
    ax.set_ylim(0, 1.1)
    
    
    plt.figlegend(bbox_to_anchor=(0.02, 0.92, 1, 0), loc= 'center',
                  borderaxespad = 0, handletextpad = 0.3, ncol=4 ,columnspacing=1.5, frameon= False)
    
    plt.show()
    
def remove_continuum_flux(wl, flux, wl_left1 = 8200, wl_right1 = 8300, wl_left2 = 9000, wl_right2 = 9100):
    
    
    # Get the median between 8200-8300 Å and 9000-9100 Å
    flux_left, flux_right = flux[(wl > wl_left1) * (wl < wl_right1)], flux[(wl > wl_left2) * (wl < wl_right2)]
    left_median, right_median = np.median(flux_left), np.median(flux_right)
    
    average_continuum = (left_median + right_median)/2
    flux -= average_continuum # Remove a flat continuum
    
    adjusted_total_flux = integrate.cumulative_trapezoid(flux, wl)[-1] # We need this number to determine the absolute CI flux later
     
    flux = flux/np.max(flux) # Normalise the peak in this region to 1
    
    return flux, adjusted_total_flux
    
def extract_line_profile_fname(el_probabilities, scattering_lines, tau_MgI, v_out, line_shape):
    
    P_CaII, P_CI, P_OI, P_MgI = el_probabilities
    s_CaII, s_CI, s_OI, s_MgI = scattering_lines # Is the line optically thick?
            
    P_CaII_string = format_float_1(P_CaII, s_CaII)
    P_CI_string = format_float_1(P_CI, s_CI)
    P_OI_string = format_float_1(P_OI, s_OI)
    P_MgI_string = format_float_1(P_MgI, s_MgI)
    v_out_string = format_float_2(v_out)
    tau_MgI_string = format_float_3(tau_MgI)
    
    
    fname = 'pCaII_' + P_CaII_string + '_pCI_' + P_CI_string + '_pOI_' + P_OI_string + '_pMgI_' + P_MgI_string + '_vout_' + v_out_string + '_tauMgI_' + tau_MgI_string + '_' + line_shape + '.csv'
    
    return fname
    
def save_profile(wl, flux, el_probabilities, scattering_lines, v_out, tau_MgI, line_shape):
    
    # Lets store this profile in a csv, with a descriptive fname
    
    data_array = np.zeros((len(wl), 2))
    data_array[:, 0] = wl
    data_array[:, 1] = flux/np.max(flux) # Normalise it for easier comparison
    
    P_CaII, P_CI, P_OI, P_MgI = el_probabilities
    s_CaII, s_CI, s_OI, s_MgI = scattering_lines # Is the line optically thick?
            
    P_CaII_string = format_float_1(P_CaII, s_CaII)
    P_CI_string = format_float_1(P_CI, s_CI)
    P_OI_string = format_float_1(P_OI, s_OI)
    P_MgI_string = format_float_1(P_MgI, s_MgI)
    v_out_string = format_float_2(v_out)
    tau_MgI_string = format_float_3(tau_MgI)
    
    
    fname = 'pCaII_' + P_CaII_string + '_pCI_' + P_CI_string + '_pOI_' + P_OI_string + '_pMgI_' + P_MgI_string + '_vout_' + v_out_string + '_tauMgI_' + tau_MgI_string + '_' + line_shape + '.csv'
        
    
    np.savetxt(os.getcwd() + '/profile_library/' + fname, data_array, delimiter = ',')
    
        
def format_float_1(number, s):
    
    if s == True: # Is the line optically thick?
        return f"t{int(number):02d}p{int(round((number - int(number)) * 100)):02d}"
    elif s == False:
        return f"f{int(number):02d}p{int(round((number - int(number)) * 100)):02d}"

def format_float_2(number):
    return f"{int(number):04d}"

def format_float_3(number):
    return f"{int(number):01d}p{int(round((number - int(number)) * 100)):02d}"
    
