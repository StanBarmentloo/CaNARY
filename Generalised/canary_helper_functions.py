import numpy as np
import matplotlib.pyplot as plt
import os
import re
from numpy.random import choice,shuffle
from scipy import integrate


def plot_result(hist, histscatt, binarr, wllines, tau_vec, N_mc, n_destroyed = 0, n_scatterdeath = 0):
    
    print('Collected photons: ', '{:.1f}'.format(np.sum(hist)*100/N_mc), '%')
    print('Destroyed photons: ', '{:.1f}'.format(n_destroyed*100/N_mc), '%')
    print('Scattering death photons: ', '{:.1f}'.format(n_scatterdeath), '%')
    
    fig, ax = plt.subplots(1, figsize = (7, 5))
    
    norm = np.max(hist)

    ax.plot(binarr, hist/norm, label = 'All escaped photons')
    ax.plot(binarr, histscatt/norm, label = 'Scattered photons')
    
    line_colours = ['purple', 'green', 'orange', 'red', 'blue', 'k']
    
    for i in range(len(wllines)):
        if tau_vec[0] > 0:
            linestyle = '-'
        else:
            linestyle = '--'
        line_label = 'line ' + str(i+1) + ': ' + str(int(wllines[i])) + ' Å'
        
        ax.axvline(x = wllines[i], linestyle = linestyle, c = line_colours[i], label = line_label, alpha = 0.5, zorder = -1)
        
    ax.set_xlabel('Observed Wavelength [Å]')
    ax.set_ylabel('Normalised line profile')
    ax.set_ylim(0, 1.1)
    
    
    plt.figlegend(bbox_to_anchor=(0.02, 0.92, 1, 0), loc= 'center',
                  borderaxespad = 0, handletextpad = 0.3, ncol=4 ,columnspacing=1.5, frameon= False)
    
    plt.show()
    
def extract_line_profile_fname(v_out, lum_vec, tau_vec, wllines, line_shape):
    
    fname = line_shape + '_vout_'
    v_out_string = f"{int(v_out):04d}"
    
    fname += v_out_string
    
    for i in range(len(lum_vec)): # For each line
        wl_string = str(int(wllines[i]))
        lum_string = f"{lum_vec[i]:.2f}"
        tau_string = format_tau_number(tau_vec[i])
        
        line_string = '_wl_' + wl_string + '_L_' + lum_string + '_tau_' + tau_string
        fname += line_string
    
    fname += '.csv'
    
    return fname

def parse_line_profile_fname(fname): 
    """
    Inverse of extract_line_profile_fname().
    Takes a filename and returns:
      v_out (float), lum_vec (list), tau_vec (list), wllines (list)
    """
    # Remove .csv if present
    if fname.endswith('.csv'):
        fname = fname[:-4]

    # Example pattern:
    # gaussian_vout_0500_wl_6563_L_1.25_tau_0.10_wl_4861_L_0.85_tau_0.05
    # First, split off the line shape part:
    parts = fname.split('_vout_')
    if len(parts) != 2:
        raise ValueError("Filename format invalid: missing '_vout_' section.")
    
    line_shape = parts[0]
    rest = parts[1]

    # Extract v_out (first 4 digits after '_vout_')
    v_out_match = re.match(r'(\d+)', rest)
    if not v_out_match:
        raise ValueError("Could not parse v_out.")
    v_out = float(v_out_match.group(1))
    
    # The rest after the v_out digits
    rest = rest[len(v_out_match.group(1)):]
    
    # Now extract all groups of (_wl_<num>_L_<num>_tau_<num>)
    pattern = r'_wl_(\d+)_L_([\d.]+)_tau_([\d.+\-eE]+)'
    matches = re.findall(pattern, rest)
    
    wllines = []
    lum_vec = []
    tau_vec = []

    for wl, lum, tau in matches:
        wllines.append(float(wl))
        lum_vec.append(float(lum))
        tau_vec.append(float(tau))
    
    return v_out, lum_vec, tau_vec, wllines


def format_tau_number(x):
    return f"{x:.2e}"
    
def profile_info(v_out, lum_vec, tau_vec, wllines):
    
    info_msg = ""
    info_msg += "\nv_out: " + f"{int(v_out):04d}" + ' km/s\n'
    
    for i in range(len(lum_vec)):
        info_msg += 'Line ' + str(int(i+1)) + ': ' + str(wllines[i]) + ' Å, tau = ' + '{:.2e}'.format(tau_vec[i]) + ', normalised luminosity: ' + '{:.1f}'.format(lum_vec[i]*100) + '% \n'
        
    info_msg += '================================================================='
    
    print(info_msg)