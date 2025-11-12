# CaNARY

Welcome to the CaNARY repository! The CaNARY code is a publicly available, simple, fast, Monte Carlo Radiative Transfer code. Its aim is to obtain luminosity estimates for emission lines that are part of a blend, in the context of nebular supernova spectra. The code tries to be as fast as possible, while still simulating the relevant physics.

The code was first presented in "Formation and Diagnostic Use of Carbon Lines in Stripped-Envelope Supernovae", Barmentloo &amp; Jerkstrand, 2025.
In this work, we use the code to obtain luminosity estimates for the [C I] 8727 line, which is part of the complex Ca NIR blend. 

At the time of writing (12-11-2025), this repository contains two versions of the code:
  - 'CaTriplet_Specialised': This is the version of the code used for the paper. It can only be used to obtain the [C I] 8727 luminosity from a Ca NIR triplet complex.
                             This version uses a few more physical assumptions than the generalised version of the code, so that we recommend this version if you are interested in
                             obtaining L_[C I]8727
  - 'Generalised': This version is usable for obtaining the luminosity of any optically thin line within a complex of multiple (both optically thin or thick) lines. It is
                   also better commented than the specialised version, and contains documentation. 

In case you want to use the code and questions show up, feel free to send an email to stan.barmentloo@astro.su.se
