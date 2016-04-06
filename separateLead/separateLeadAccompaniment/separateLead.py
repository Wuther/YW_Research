#!/usr/bin/python

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

import SIMM

import scipy

import scipy.io.wavfile as wav

import os
#import scikits.audiolab

from tracking import viterbiTrackingArray

# SOME USEFUL, INSTRUMENTAL, FUNCTIONS

def db(val):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(val)

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return sum((-np.log(X / Y) + (X / Y) - 1))


# DEFINING SOME WINDOW FUNCTIONS

def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)

    Computes a "sinebell" window function of length L=lengthWindow

    The formula is:
        window(t) = sin(pi * t / L), t = 0..L-1
    """
    window = np.sin((np.pi * (np.arange(lengthWindow))) \
                    / (1.0 * lengthWindow))
    return window

def hann(args):
    """
    window = hann(args)

    Computes a Hann window, with NumPy's function hanning(args).
    """
    return np.hanning(args)

# FUNCTIONS FOR TIME-FREQUENCY REPRESENTATION

def stft(data, window=sinebell(2048), hopsize=256.0, nfft=2048.0, \
         fs=44100.0):
    """
    X, F, N = stft(data, window=sinebell(2048), hopsize=1024.0,
                   nfft=2048.0, fs=44100)

    Computes the short time Fourier transform (STFT) of data.

    Inputs:
        data                  : one-dimensional time-series to be
                                analyzed
        window=sinebell(2048) : analysis window
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation (the user has to provide an
                                even number)
        fs=44100.0            : sampling rate of the signal

    Outputs:
        X                     : STFT of data
        F                     : values of frequencies at each Fourier
                                bins
        N                     : central time at the middle of each
                                analysis window
    """

    # window defines the size of the analysis windows
    lengthWindow = window.size

    # !!! adding zeros to the beginning of data, such that the first
    # window is centered on the first sample of data
    data = np.concatenate((np.zeros(lengthWindow / 2.0),data))          
    lengthData = data.size
    
    # adding one window for the last frame (same reason as for the
    # first frame)
    numberFrames = np.ceil((lengthData - lengthWindow) / hopsize \
                           + 1) + 1  
    newLengthData = (numberFrames - 1) * hopsize + lengthWindow
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros([newLengthData - lengthData])))

    # the output STFT has nfft/2+1 rows. Note that nfft has to be an
    # even number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2.0 + 1

    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, nfft);
        
    F = np.arange(numberFrequencies) / nfft * fs
    N = np.arange(numberFrames) * hopsize / fs
    
    return STFT, F, N

def istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0):
    """
    data = istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0)

    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.

    Inputs:
        X                     : STFT of the signal, to be "inverted"
        window=sinebell(2048) : synthesis window
                                (should be the "complementary" window
                                for the analysis window)
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation
                                (the user has to provide an even number)

    Outputs:
        data                  : time series corresponding to the given
                                STFT the first half-window is removed,
                                complying with the STFT computation
                                given in the function 'stft'
    """
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = np.array(X.shape)
    lengthData = hopsize * (numberFrames - 1) + lengthWindow
    
    data = np.zeros(lengthData)
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], nfft)
        frameTMP = frameTMP[:lengthWindow]
        data[beginFrame:endFrame] = data[beginFrame:endFrame] \
                                    + window * frameTMP

    # remove the extra bit before data that was - supposedly - added
    # in the stft computation:
    data = data[(lengthWindow / 2.0):] 
    return data

# DEFINING THE FUNCTIONS TO CREATE THE 'BASIS' WF0

def generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048, stepNotes=4, \
                         lengthWindow=2048, Ot=0.5, perF0=2, \
                         depthChirpInSemiTone=0.5):
    """
    F0Table, WF0 = generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048,
                                        stepNotes=4, lengthWindow=2048,
                                        Ot=0.5, perF0=2,
                                        depthChirpInSemiTone=0.5)

    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:
    Inputs:
        minF0                the minimum value for the fundamental
                             frequency (F0)
        maxF0                the maximum value for F0
        Fs                   the desired sampling rate
        Nfft                 the number of bins to compute the Fourier
                             transform
        stepNotes            the number of F0 per semitone
        lengthWindow         the size of the window for the Fourier
                             transform
        Ot                   the glottal opening coefficient for
                             KLGLOTT88
        perF0                the number of chirps considered per F0
                             value
        depthChirpInSemiTone the maximum value, in semitone, of the
                             allowed chirp per F0

    Outputs:
        F0Table the vector containing the values of the fundamental
                frequencies in Hertz (Hz) corresponding to the
                harmonic combs in WF0, i.e. the columns of WF0
        WF0     the basis matrix, where each column is a harmonic comb
                generated by KLGLOTT88 (with a sinusoidal model, then
                transformed into the spectral domain)
    """
    
    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(Fs)
    stepNotes=np.double(stepNotes)
    
    # computing the F0 table:
    numberOfF0 = np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1
    F0Table=minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))

    numberElementsInWF0 = numberOfF0 * perF0
    
    # computing the desired WF0 matrix
    WF0 = np.zeros([Nfft, numberElementsInWF0],dtype=np.double)
    for fundamentalFrequency in np.arange(numberOfF0):
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0)
        WF0[:,fundamentalFrequency * perF0] = np.abs(odgdSpec) ** 2
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2 
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(odgdSpec) ** 2
    
    return F0Table, WF0

def generate_ODGD_spec(F0, Fs, lengthOdgd=2048, Nfft=2048, Ot=0.5, \
                       t0=0.0, analysisWindowType='sinebell'):
    """
    generateODGDspec:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """

    # converting input to double:
    F0 = np.double(F0)
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType=='sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType=='hanning' or \
               analysisWindowType=='hanning':
            analysisWindow = hann(lengthOdgd)
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / F0)
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot

    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 \
                 * (np.exp(-temp_array) \
                    + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                    - (6 * (1 - np.exp(-temp_array)) \
                       / (temp_array ** 2))) \
                       / temp_array

    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    odgd = np.exp(np.outer(2.0 * 1j * np.pi * F0 * frequency_numbers, \
                           timeStamps)) \
                           * np.outer(amplitudes, np.ones(lengthOdgd))
    odgd = np.sum(odgd, axis=0)
    
    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)

    return odgd, odgdSpectrum

def generate_ODGD_spec_chirped(F1, F2, Fs, lengthOdgd=2048, Nfft=2048, \
                               Ot=0.5, t0=0.0, \
                               analysisWindowType='sinebell'):
    """
    generateODGDspecChirped:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """
    
    # converting input to double:
    F1 = np.double(F1)
    F2 = np.double(F2)
    F0 = np.double(F1 + F2) / 2.0
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType == 'sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType == 'hanning' or \
               analysisWindowType == 'hann':
            analysisWindow = hann(lengthOdgd)
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / np.max(F1, F2))
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot

    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 * \
                 (np.exp(-temp_array) \
                  + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                  - (6 * (1 - np.exp(-temp_array)) \
                     / (temp_array ** 2))) \
                  / temp_array

    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    odgd = np.exp(2.0 * 1j * np.pi \
                  * (np.outer(F1 * frequency_numbers,timeStamps) \
                     + np.outer((F2 - F1) \
                                * frequency_numbers,timeStamps ** 2) \
                     / (2 * lengthOdgd / Fs))) \
                     * np.outer(amplitudes,np.ones(lengthOdgd))
    odgd = np.sum(odgd,axis=0)
    
    # spectrum:
    odgdSpectrum = np.fft.fft(real(odgd * analysisWindow), n=Nfft)

    return odgd, odgdSpectrum


def generateHannBasis(numberFrequencyBins, sizeOfFourier, Fs, \
                      frequencyScale='linear', numberOfBasis=20, \
                      overlap=.75):
    isScaleRecognized = False
    if frequencyScale == 'linear':
        # number of windows generated:
        numberOfWindowsForUnit = np.ceil(1.0 / (1.0 - overlap))
        # recomputing the overlap to exactly fit the entire
        # number of windows:
        overlap = 1.0 - 1.0 / np.double(numberOfWindowsForUnit)
        # length of the sine window - that is also to say: bandwidth
        # of the sine window:
        lengthSineWindow = np.ceil(numberFrequencyBins \
                                   / ((1.0 - overlap) \
                                      * (numberOfBasis - 1) + 1 \
                                      - 2.0 * overlap))
        # even window length, for convenience:
        lengthSineWindow = 2.0 * np.floor(lengthSineWindow / 2.0) 

        # for later compatibility with other frequency scales:
        mappingFrequency = np.arange(numberFrequencyBins) 

        # size of the "big" window
        sizeBigWindow = 2.0 * numberFrequencyBins

        # centers for each window
        ## the first window is centered at, in number of window:
        firstWindowCenter = -numberOfWindowsForUnit + 1
        ## and the last is at
        lastWindowCenter = numberOfBasis - numberOfWindowsForUnit + 1
        ## center positions in number of frequency bins
        sineCenters = np.round(\
            np.arange(firstWindowCenter, lastWindowCenter) \
            * (1 - overlap) * np.double(lengthSineWindow) \
            + lengthSineWindow / 2.0)
        
        # For future purpose: to use different frequency scales
        isScaleRecognized = True

    # For frequency scale in logarithm (such as ERB scales) 
    if frequencyScale == 'log':
        isScaleRecognized = False

    # checking whether the required scale is recognized
    if not(isScaleRecognized):
        print "The desired feature for frequencyScale is not recognized yet..."
        return 0

    # the shape of one window:
    prototypeSineWindow = hann(lengthSineWindow)
    # adding zeroes on both sides, such that we do not need to check
    # for boundaries
    bigWindow = np.zeros([sizeBigWindow * 2, 1])
    bigWindow[(sizeBigWindow - lengthSineWindow / 2.0):\
              (sizeBigWindow + lengthSineWindow / 2.0)] \
              = np.vstack(prototypeSineWindow)
    
    WGAMMA = np.zeros([numberFrequencyBins, numberOfBasis])
    
    for p in np.arange(numberOfBasis):
        WGAMMA[:, p] = np.hstack(bigWindow[np.int32(mappingFrequency \
                                                    - sineCenters[p] \
                                                    + sizeBigWindow)])

    return WGAMMA

# MAIN FUNCTION, FOR DEFAULT BEHAVIOUR IF THE SCRIPT IS "LAUNCHED"
def main():
    import optparse
    
    usage = "usage: %prog [options] inputAudioFile"
    parser = optparse.OptionParser(usage)
    # Name of the output files:
    parser.add_option("-v", "--vocal-output-file",
                      dest="voc_output_file", type="string",
                      help="name of the audio output file for the estimated\nsolo (vocal) part",
                      default="estimated_solo.wav")
    parser.add_option("-m", "--music-output-file",
                      dest="mus_output_file", type="string",
                      help="name of the audio output file for the estimated\nmusic part",
                      default="estimated_music.wav")
    parser.add_option("-p", "--pitch-output-file",
                      dest="pitch_output_file", type="string",
                      help="name of the output file for the estimated pitches",
                      default="pitches.txt")

    # Some more optional options:
    parser.add_option("-d", "--with-display", dest="displayEvolution",
                      action="store_true",help="display the figures",
                      default=False)
    parser.add_option("-q", "--quiet", dest="verbose",
                      action="store_false",
                      help="use to quiet all output verbose",
                      default=True)
    parser.add_option("--nb-iterations", dest="nbiter",
                      help="number of iterations", type="int",
                      default=50)
    parser.add_option("--window-size", dest="windowSize", type="float",
                      default=0.04644,help="size of analysis windows, in s.")
    parser.add_option("--Fourier-size", dest="fourierSize", type="int",
                      default=2048, help="size of Fourier transforms, in samples.")
    parser.add_option("--hopsize", dest="hopsize", type="float",
                      default=0.0058,
                      help="size of the hop between analysis windows, in s.")

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("incorrect number of arguments, use option -h for help.")

    displayEvolution = options.displayEvolution
    if displayEvolution:
        import matplotlib.pyplot as plt
        import imageMatlab

        plt.rc('text', usetex=True)
        plt.rc('image',cmap='gray_r')
        plt.ion()

    # Compulsory option: name of the input file:
    inputAudioFile = args[0]
    fs, data = wav.read(inputAudioFile)
    #data, fs, enc = scikits.audiolab.wavread(inputAudioFile)
    if data.shape[0] != data.size: # data is multi-channel
        data = np.mean(data,axis=1)

    # Processing the options:
    windowSizeInSamples = np.round(options.windowSize * fs)
    hopsize = np.round(options.hopsize * fs)
    NFT = options.fourierSize
    niter = options.nbiter

    if options.verbose:
        print "Size of analysis windows: ", windowSizeInSamples, "\n"
        print "Hopsize: ", hopsize, "\n"
        print "Size of Fourier transforms: ", NFT, "\n"
        print "Number of iterations to be done: ", niter, "\n"
    
    X, F, N = stft(data, fs=fs, hopsize=hopsize,
                   window=sinebell(windowSizeInSamples), nfft=NFT)
    # SX is the power spectrogram:
    SX = np.maximum(np.abs(X) ** 2, 10 ** -8)

    del data, F, N

    # TODO: also process these as options:
    minF0 = 100
    maxF0 = 800
    Fs = fs
    F, N = SX.shape
    stepNotes = 20 # this is the number of F0s within one semitone
    K = 50 # number of spectral shapes for the filter part
    R = 40 # number of spectral shapes for the accompaniment
    P = 30 # number of elements in dictionary of smooth filters
    chirpPerF0 = 1 # number of chirped spectral shapes between each F0
                   # this feature should be further studied before
                   # we find a good way of doing that.

    # Create the harmonic combs, for each F0 between minF0 and maxF0: 
    F0Table, WF0 = \
             generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048, \
                                  stepNotes=stepNotes, \
                                  lengthWindow=2048, Ot=0.25, \
                                  perF0=chirpPerF0, \
                                  depthChirpInSemiTone=.15)
    WF0 = WF0[0:F, :] # ensure same size as SX 
    NF0 = F0Table.size # number of harmonic combs
    # Normalization: 
    WF0 = WF0 / np.outer(np.ones(F), np.amax(WF0, axis=0))

    # Create the dictionary of smooth filters, for the filter part of
    # the lead isntrument:
    WGAMMA = generateHannBasis(F, 2048, Fs=fs, frequencyScale='linear', \
                               numberOfBasis=P, overlap=.75)

    if displayEvolution:
        plt.figure(1);plt.clf()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Leading source number $u$', fontsize=16)
        plt.ion()
        # plt.show()
        raw_input("Press Return to resume the program. \nBe sure that the figure has been already displayed, so that the evolution of HF0 will be visible. ")

    # First round of parameter estimation:
    HGAMMA, HPHI, HF0, HM, WM, recoError1 = SIMM.SIMM(
        # the data to be fitted to:
        SX,
        # the basis matrices for the spectral combs
        WF0,
        # and for the elementary filters:
        WGAMMA,
        # number of desired filters, accompaniment spectra:
        numberOfFilters=K, numberOfAccompanimentSpectralShapes=R,
        # putting only 2 elements in accompaniment for a start...
        # if any, initial amplitude matrices for 
        HGAMMA0=None, HPHI0=None,
        HF00=None,
        WM0=None, HM0=None,
        # Some more optional arguments, to control the "convergence"
        # of the algo
        numberOfIterations=niter, updateRulePower=1.,
        stepNotes=stepNotes, 
        lambdaHF0 = 0.0 / (1.0 * SX.max()), alphaHF0=0.9,
        verbose=options.verbose, displayEvolution=displayEvolution)

    if displayEvolution:
        plt.figure(3);plt.clf()
        plt.subplot(221)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:, 0])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")
        plt.subplot(222)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:,1])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")
        plt.subplot(223)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:, 2])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")
        plt.subplot(224)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:, 3])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")

        plt.figure(4);plt.clf()
        imageMatlab.imageM(db(np.dot(np.dot(WGAMMA, HGAMMA), HPHI)))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Frequency bin number $f$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure(5);plt.clf()
        imageMatlab.imageM(db(HF0), vmin=-100, cmap=plt.cm.gray_r)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Leading source number $u$', fontsize=16)
        # plt.xlim([3199.5, 3500.5]) # For detailed picture of HF0
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure(6);plt.clf()
        imageMatlab.imageM(db(np.dot(WM, HM)), vmin=-50)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Frequency bin number $f$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure(7);plt.clf()
        imageMatlab.imageM(db(WM), vmin=-50)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Element number $r$', fontsize=16)
        plt.ylabel(r'Frequency bin number $f$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
    if displayEvolution:
        h2 = plt.figure(2);plt.clf();
        imageMatlab.imageM(20 * np.log10(HF0))
        matMax = (20 * np.log10(HF0)).max()
        matMed = np.median(20 * np.log10(HF0))
        plt.clim([matMed - 100, matMax])

    # Viterbi decoding to estimate the predominant fundamental
    # frequency line
    scale = 1.0
    transitions = np.exp(-np.floor(np.arange(0,NF0) / stepNotes) * scale)
    cutoffnote = 2 * 5 * stepNotes
    transitions[cutoffnote:] = transitions[cutoffnote - 1]

    transitionMatrixF0 = np.zeros([NF0 + 1, NF0 + 1]) # toeplitz matrix
    b = np.arange(NF0)
    transitionMatrixF0[0:NF0, 0:NF0] = \
                              transitions[\
        np.array(np.abs(np.outer(np.ones(NF0), b) \
                        - np.outer(b, np.ones(NF0))), dtype=int)]
    pf_0 = transitions[cutoffnote - 1] * 10 ** (-90)
    p0_0 = transitions[cutoffnote - 1] * 10 ** (-100)
    p0_f = transitions[cutoffnote - 1] * 10 ** (-80)
    transitionMatrixF0[0:NF0, NF0] = pf_0
    transitionMatrixF0[NF0, 0:NF0] = p0_f
    transitionMatrixF0[NF0, NF0] = p0_0

    sumTransitionMatrixF0 = np.sum(transitionMatrixF0, axis=1)
    transitionMatrixF0 = transitionMatrixF0 \
                         / np.outer(sumTransitionMatrixF0, \
                                    np.ones(NF0 + 1))

    priorProbabilities = 1 / (NF0 + 1.0) * np.ones([NF0 + 1])
    logHF0 = np.zeros([NF0 + 1, N])
    normHF0 = np.amax(HF0, axis=0)
    barHF0 = np.array(HF0)

    logHF0[0:NF0, :] = np.log(barHF0)
    logHF0[0:NF0, normHF0==0] = np.amin(logHF0[logHF0>-np.Inf])
    logHF0[NF0, :] = np.maximum(np.amin(logHF0[logHF0>-np.Inf]),-100)

    indexBestPath = viterbiTrackingArray(\
        logHF0, np.log(priorProbabilities), np.log(transitionMatrixF0))

    np.savetxt(options.pitch_output_file,
               np.array([np.arange(N)*options.hopsize,
                         F0Table[np.array(indexBestPath,dtype=int)]]).T)

    if displayEvolution:
        h2.hold(True)
        plt.plot(indexBestPath, '-b')
        h2.hold(False)
        plt.axis('tight')
        raw_input("Press Return to resume the program...")

    del logHF0

    # Second round of parameter estimation, with specific
    # initial HF00:
    HF00 = np.zeros([NF0 * chirpPerF0, N])

    scopeAllowedHF0 = 1.0 / 1.0

    # indexes for HF00:
    # TODO: reprogram this with a 'where'?...
    dim1index = np.array(\
        np.maximum(\
        np.minimum(\
        np.outer(chirpPerF0 * indexBestPath,
                 np.ones(chirpPerF0 \
                         * (2 \
                            * np.floor(stepNotes / scopeAllowedHF0) \
                            + 1))) \
        + np.outer(np.ones(N),
                   np.arange(-chirpPerF0 \
                             * np.floor(stepNotes / scopeAllowedHF0),
                             chirpPerF0 \
                             * (np.floor(stepNotes / scopeAllowedHF0) \
                                + 1))),
        chirpPerF0 * NF0 - 1),
        0),
        dtype=int).reshape(1, N * chirpPerF0 \
                           * (2 * np.floor(stepNotes / scopeAllowedHF0) \
                              + 1))
    dim2index = np.outer(np.arange(N),
                         np.ones(chirpPerF0 \
                                 * (2 * np.floor(stepNotes \
                                                 / scopeAllowedHF0) + 1), \
                                 dtype=int)\
                         ).reshape(1, N * chirpPerF0 \
                                   * (2 * np.floor(stepNotes \
                                                   / scopeAllowedHF0) \
                                      + 1))
    HF00[dim1index, dim2index] = 1 # HF0.max()

    HF00[:, indexBestPath == (NF0 - 1)] = 0.0

    WF0effective = WF0
    HF00effective = HF00

    del HF0, HGAMMA, HPHI, HM, WM, HF00

    HGAMMA, HPHI, HF0, HM, WM, recoError2 = SIMM.SIMM(
        # the data to be fitted to:
        SX,
        # the basis matrices for the spectral combs
        WF0effective,
        # and for the elementary filters:
        WGAMMA,
        # number of desired filters, accompaniment spectra:
        numberOfFilters=K, numberOfAccompanimentSpectralShapes=R,
        # if any, initial amplitude matrices for
        HGAMMA0=None, HPHI0=None,
        HF00=HF00effective,
        WM0=None, HM0=None,
        # Some more optional arguments, to control the "convergence"
        # of the algo
        numberOfIterations=niter, updateRulePower=1.0,
        stepNotes=stepNotes, 
        lambdaHF0 = 0.0 / (1.0 * SX.max()), alphaHF0=0.9,
        verbose=options.verbose, displayEvolution=displayEvolution)

    WPHI = np.dot(WGAMMA, HGAMMA)
    SPHI = np.dot(WPHI, HPHI)
    SF0 = np.dot(WF0effective, HF0)
    SM = np.dot(WM, HM)

    hatSX = SPHI * SF0 + SM

    hatV = SPHI * SF0 / hatSX * X

    vest = istft(hatV, hopsize=hopsize, nfft=NFT,
                 window=sinebell(windowSizeInSamples)) / 4.0

   # scikits.audiolab.wavwrite(vest, options.voc_output_file, fs)
    wav.write(options.voc_output_file, fs, \
              vest)
    hatM = SM / hatSX * X

    mest = istft(hatM, hopsize=hopsize, nfft=NFT,
                 window=sinebell(windowSizeInSamples)) / 4.0

    #scikits.audiolab.wavwrite(mest, options.mus_output_file, fs)
    wav.write(options.mus_output_file, fs, \
              mest)
    if displayEvolution:
        plt.figure(13);plt.clf()
        plt.subplot(221)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:, 0])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")
        plt.subplot(222)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:, 1])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")
        plt.subplot(223)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:, 2])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")
        plt.subplot(224)
        plt.plot(db(np.dot(WGAMMA, HGAMMA[:, 3])))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim([-30, 0])
        plt.axis("tight")

        plt.figure(14);plt.clf()
        imageMatlab.imageM(db(np.dot(np.dot(WGAMMA, HGAMMA), HPHI)))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Frequency bin number $f$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure(141);plt.clf()
        SVhat = db(np.dot(np.dot(WGAMMA, HGAMMA), HPHI)) \
                + db(np.dot(WF0, HF0))
        imageMatlab.imageM(SVhat, vmax=SVhat.max(),
                           vmin=SVhat.max() - 50)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Frequency bin number $f$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure(15);plt.clf()
        imageMatlab.imageM(db(HF0), vmin=-100, cmap=plt.cm.gray_r)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Leading source number $u$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.xlim([3199.5, 3500.5]) # For detailed picture of HF0

        plt.figure(16)
        plt.clf()
        imageMatlab.imageM(db(np.dot(WM, HM)),
                           vmin=np.maximum(-50, db(SM.min())))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Frame number $n$', fontsize=16)
        plt.ylabel(r'Frequency bin number $f$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.figure(17)
        plt.clf()
        imageMatlab.imageM(db(WM), vmin=-50)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'Element number $r$', fontsize=16)
        plt.ylabel(r'Frequency bin number $f$', fontsize=16)
        cb = plt.colorbar(fraction=0.04)
        plt.axes(cb.ax)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        raw_input("Press Return to end the program...")
        print "Done!"

if __name__ == '__main__':
    main()
