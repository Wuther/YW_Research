#!/usr/bin/python
#

# Script implementing the multiplicative rules from the following
# article:
# 
# J.-L. Durrieu, G. Richard, B. David and C. Fevotte
# Source/Filter Model for Unsupervised Main Melody
# Extraction From Polyphonic Audio Signals
# IEEE Transactions on Audio, Speech and Language Processing
# Vol. 18, No. 3, March 2010
#
# with more details and new features explained in my PhD thesis:
#
# J.-L. Durrieu,
# Automatic Extraction of the Main Melody from Polyphonic Music Signals,
# EDITE
# Institut TELECOM, TELECOM ParisTech, CNRS LTCI

# copyright (C) 2010 Jean-Louis Durrieu
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import time

from numpy.random import randn
from string import join

def db(positiveValue):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(np.abs(positiveValue))

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return np.sum((-np.log(X / Y) + (X / Y) - 1))

def SIMM(# the data to be fitted to:
         SX,
         # the basis matrices for the spectral combs
         WF0,
         # and for the elementary filters:
         WGAMMA,
         # number of desired filters, accompaniment spectra:
         numberOfFilters=4, numberOfAccompanimentSpectralShapes=10,
         # if any, initial amplitude matrices for 
         HGAMMA0=None, HPHI0=None,
         HF00=None,
         WM0=None, HM0=None,
         # Some more optional arguments, to control the "convergence"
         # of the algo
         numberOfIterations=1000, updateRulePower=1.0,
         stepNotes=4, 
         lambdaHF0=0.00,alphaHF0=0.99,
         displayEvolution=False, verbose=True):
    """
    HGAMMA, HPHI, HF0, HM, WM, recoError =
        SIMM(SX, WF0, WGAMMA, numberOfFilters=4,
             numberOfAccompanimentSpectralShapes=10, HGAMMA0=None, HPHI0=None,
             HF00=None, WM0=None, HM0=None, numberOfIterations=1000,
             updateRulePower=1.0, stepNotes=4, 
             lambdaHF0=0.00, alphaHF0=0.99, displayEvolution=False,
             verbose=True)

    Implementation of the Smooth-filters Instantaneous Mixture Model
    (SIMM). This model can be used to estimate the main melody of a
    song, and separate the lead voice from the accompaniment, provided
    that the basis WF0 is constituted of elements associated to
    particular pitches.

    Inputs:
        SX
            the F x N power spectrogram to be approximated.
            F is the number of frequency bins, while N is the number of
            analysis frames
        WF0
            the F x NF0 basis matrix containing the NF0 source elements
        WGAMMA
            the F x P basis matrix of P smooth elementary filters
        numberOfFilters
            the number of filters K to be considered
        numberOfAccompanimentSpectralShapes
            the number of spectral shapes R for the accompaniment
        HGAMMA0
            the P x K decomposition matrix of WPHI on WGAMMA
        HPHI0
            the K x N amplitude matrix of the filter part of the lead
            instrument
        HF00
            the NF0 x N amplitude matrix for the source part of the lead
            instrument
        WM0
            the F x R the matrix for spectral shapes of the
            accompaniment
        HM0
            the R x N amplitude matrix associated with each of the R
            accompaniment spectral shapes
        numberOfIterations
            the number of iterations for the estimatino algorithm
        updateRulePower
            the power to which the multiplicative gradient is elevated to
        stepNotes
            the number of elements in WF0 per semitone. stepNotes=4 means
            that there are 48 elements per octave in WF0.
        lambdaHF0
            Lagrangian multiplier for the octave control
        alphaHF0
            parameter that controls how much influence a lower octave
            can have on the upper octave's amplitude.

    Outputs:
        HGAMMA
            the estimated P x K decomposition matrix of WPHI on WGAMMA
        HPHI
            the estimated K x N amplitude matrix of the filter part 
        HF0
            the estimated NF0 x N amplitude matrix for the source part
        HM
            the estimated R x N amplitude matrix for the accompaniment
        WM
            the estimate F x R spectral shapes for the accompaniment
        recoError
            the successive values of the Itakura Saito divergence
            between the power spectrogram and the spectrogram
            computed thanks to the updated estimations of the matrices.

    Please also refer to the following article for more details about
    the algorithm within this function, as well as the meaning of the
    different matrices that are involved:
        J.-L. Durrieu, G. Richard, B. David and C. Fevotte
        Source/Filter Model for Unsupervised Main Melody
        Extraction From Polyphonic Audio Signals
        IEEE Transactions on Audio, Speech and Language Processing
        Vol. 18, No. 3, March 2010
    """
    eps = 10 ** (-6)

    if displayEvolution:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()

    # renamed for convenience:
    K = numberOfFilters
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SX.shape
    Fwf0, NF0 = WF0.shape
    Fwgamma, P = WGAMMA.shape
    
    # Checking the sizes of the matrices
    if Fwf0 != F:
        return False # A REVOIR!!!
    if HGAMMA0 is None:
        HGAMMA0 = np.abs(randn(P, K))
    else:
        if not(isinstance(HGAMMA0,np.ndarray)): # default behaviour
            HGAMMA0 = np.array(HGAMMA0)
        Phgamma0, Khgamma0 = HGAMMA0.shape
        if Phgamma0 != P or Khgamma0 != K:
            print "Wrong dimensions for given HGAMMA0, \n"
            print "random initialization used instead"
            HGAMMA0 = np.abs(randn(P, K))

    HGAMMA = HGAMMA0
    
    if HPHI0 is None: # default behaviour
        HPHI = np.abs(randn(K, N))
    else:
        Khphi0, Nhphi0 = np.array(HPHI0).shape
        if Khphi0 != K or Nhphi0 != N:
            print "Wrong dimensions for given HPHI0, \n"
            print "random initialization used instead"
            HPHI = np.abs(randn(K, N))
        else:
            HPHI = np.array(HPHI0)

    if HF00 is None:
        HF00 = np.abs(randn(NF0, N))
    else:
        if np.array(HF00).shape[0] == NF0 and np.array(HF00).shape[1] == N:
            HF00 = np.array(HF00)
        else:
            print "Wrong dimensions for given HF00, \n"
            print "random initialization used instead"
            HF00 = np.abs(randn(NF0, N))
    HF0 = HF00

    if HM0 is None:
        HM0 = np.abs(randn(R, N))
    else:
        if np.array(HM0).shape[0] == R and np.array(HM0).shape[1] == N:
            HM0 = np.array(HM0)
        else:
            print "Wrong dimensions for given HM0, \n"
            print "random initialization used instead"
            HM0 = np.abs(randn(R, N))
    HM = HM0

    if WM0 is None:
        WM0 = np.abs(randn(F, R))
    else:
        if np.array(WM0).shape[0] == F and np.array(WM0).shape[1] == R:
            WM0 = np.array(WM0)
        else:
            print "Wrong dimensions for given WM0, \n"
            print "random initialization used instead"
            WM0 = np.abs(randn(F, R))
    WM = WM0
    
    # Iterations to estimate the SIMM parameters:
    WPHI = np.dot(WGAMMA, HGAMMA)
    SF0 = np.dot(WF0, HF0)
    SPHI = np.dot(WPHI, HPHI)
    SM = np.dot(WM, HM)
    hatSX = SF0 * SPHI + SM

    SX = SX + np.abs(randn(F, N)) ** 2
                                       # should not need this line
                                       # which ensures that data is not
                                       # 0 everywhere. 
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])

    # Array containing the reconstruction error after the update of each 
    # of the parameter matrices:
    recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose:
        print "Reconstruction error at beginning: ", recoError[0]
    counterError = 1
    if displayEvolution:
        h1 = plt.figure(1)

    # Main loop for multiplicative updating rules:
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        if verbose:
            print "iteration ", n, " over ", numberOfIterations
        if displayEvolution:
            h1.clf();imageM(db(HF0));
            plt.clim([np.amax(db(HF0))-100, np.amax(db(HF0))]);plt.draw();
            h1.clf();
            imageM(HF0 * np.outer(np.ones([NF0, 1]),
                                  1 / (HF0.max(axis=0))));

        # updating HF0:
        tempNumFbyN = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SPHI / np.maximum(hatSX, eps)

        # This to enable octave control
        HF0[np.arange(12 * stepNotes, NF0), :] \
           = HF0[np.arange(12 * stepNotes, NF0), :] \
             * (np.dot(WF0[:, np.arange(12 * stepNotes,
                                        NF0)].T, tempNumFbyN) \
                / np.maximum(
            np.dot(WF0[:, np.arange(12 * stepNotes, NF0)].T,
                   tempDenFbyN) \
            + lambdaHF0 * (- (alphaHF0 - 1.0) \
                           / np.maximum(HF0[
            np.arange(12 * stepNotes, NF0), :], eps) \
                           + HF0[
            np.arange(NF0 - 12 * stepNotes), :]),
            eps)) ** omega

        HF0[np.arange(12 * stepNotes), :] \
           = HF0[np.arange(12 * stepNotes), :] \
             * (np.dot(WF0[:, np.arange(12 * stepNotes)].T,
                      tempNumFbyN) /
               np.maximum(
                np.dot(WF0[:, np.arange(12 * stepNotes)].T,
                       tempDenFbyN), eps)) ** omega

##        # normal update rules:
##        HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) /
##                     np.maximum(np.dot(WF0.T, tempDenFbyN), eps)) ** omega
        
        
        SF0 = np.dot(WF0, HF0)
        hatSX = SF0 * SPHI + SM
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HF0   : ",
            print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
    
        # updating HPHI
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)
        HPHI = HPHI * (np.dot(WPHI.T, tempNumFbyN) / np.maximum(np.dot(WPHI.T, tempDenFbyN), eps)) ** omega
        sumHPHI = np.sum(HPHI, axis=0)
        HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
        HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)

        SF0 = np.dot(WF0, HF0)
        SPHI = np.dot(WPHI, HPHI)
        hatSX = SF0 * SPHI + SM
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HPHI  : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        
        # updating HM
        tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = 1 / np.maximum(hatSX, eps)
        HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) / np.maximum(np.dot(WM.T, tempDenFbyN), eps)) ** omega, eps)

        SM = np.dot(WM, HM)
        hatSX = SF0 * SPHI + SM
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HM    : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1

        # updating HGAMMA
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)
        HGAMMA = np.maximum(HGAMMA * (np.dot(WGAMMA.T, np.dot(tempNumFbyN, HPHI.T)) / np.maximum(np.dot(WGAMMA.T, np.dot(tempDenFbyN, HPHI.T)), eps)) ** omega, eps)

        sumHGAMMA = np.sum(HGAMMA, axis=0)
        HGAMMA[:, sumHGAMMA>0] = HGAMMA[:, sumHGAMMA>0] / np.outer(np.ones(P), sumHGAMMA[sumHGAMMA>0])
        HPHI = HPHI * np.outer(sumHGAMMA, np.ones(N))
        sumHPHI = np.sum(HPHI, axis=0)
        HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
        HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)
        
        WPHI = np.dot(WGAMMA, HGAMMA)
        SF0 = np.dot(WF0, HF0)
        SPHI = np.dot(WPHI, HPHI)
        hatSX = SF0 * SPHI + SM
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HGAMMA: ",
            print recoError[counterError] - recoError[counterError - 1]
            
        counterError += 1

        # updating WM, after a certain number of iterations (here, after 1 iteration)
        if n > 0: # this test can be used such that WM is updated only
                  # after a certain number of iterations
            tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
            tempDenFbyN = 1 / np.maximum(hatSX, eps)
            WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) /
                                  np.maximum(np.dot(tempDenFbyN, HM.T),
                                             eps)) ** omega, eps)
            
            sumWM = np.sum(WM, axis=0)
            WM[:, sumWM>0] = (WM[:, sumWM>0] /
                              np.outer(np.ones(F),sumWM[sumWM>0]))
            HM = HM * np.outer(sumWM, np.ones(N))
            
            SM = np.dot(WM, HM)
            hatSX = SF0 * SPHI + SM
            
            recoError[counterError] = ISDistortion(SX, hatSX)

            if verbose:
                print "Reconstruction error difference after WM    : ",
                print recoError[counterError] - recoError[counterError - 1]
            counterError += 1

    return HGAMMA, HPHI, HF0, HM, WM, recoError
