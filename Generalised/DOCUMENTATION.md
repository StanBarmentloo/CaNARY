Hi there! Below we will give some short documentation on how to use the code.

The workflow of CaNARY consists of two main steps:
    - Creating synthetic spectra for a line blend of interest.
    - Comparing a library of synthetic spectra to an observation of the line blend in a SN, and obtaining a luminosity estimate
      for one of the lines in the blend.

As the names of the notebook suggest, each step has their own dedicated notebook. 
The remaining .py files contain functions that are used in the two notebooks. 
Let us now go through the steps required from the user to obtain a luminosity estimate!


Step 1: Download the 'Generalised' folder to your local machine.

Step 2: Enter Notebook 001_Creating_CaNARY_spectra.ipynb
  Step 2a: Enter your inputs under 'General Inputs'
  Step 2b: Enter your inputs under 'Line Selection'. Remember that you can pick your own lines in any way you want.
  Step 2c: Run the full notebook and wait for your line profiles to be created!

Step 3: Enter Notebook 002_Fitting_Line_Contribution.ipynb
  Step 3a: Enter your inputs under 'User Inputs'
  Step 3b: Place all SN spectra you want to obtain luminosities for in the 'spectra_to_be_fitted' folder.
  Step 3c: Run the full notebook. If you put 'autopilot' to False, provide the constraints for each SN spectrum.

You should now have obtained a results file, with the following columns:
sn_spectrum_fname,total_flux_in_line_complex,fraction_of_line_of_interest_of_col1,1sig_lower_limit_to_col2,1sig_upper_limit_to_col2,score_per_datapoint(measure of goodness of fit, aim for < 0.015)
