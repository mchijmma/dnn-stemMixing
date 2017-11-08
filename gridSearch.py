

from __future__ import division
from __future__ import print_function


from datetime import datetime

import sys
import os
import pickle
import copy
from collections import OrderedDict



import h5py
import librosa
import librosa.display
import numpy as np 
import pandas as pd
from scipy.signal import lfilter

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler, scale
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


kLabels = ['b','g','v','k']
kSR = 44100
kN = 2048
kHopLength = 1024
kSeed = 7
kType = 'mono'
np.random.seed(kSeed)

kPathFiles = './DAE-WIMP/' 
kPathAudio = './Music/Data/MedleyDB/Audio/info.tracks'




def loadPickle(name, path = kPathFiles):  
    
    # load data from pkl file
    with open(path + name, "rb") as fp:
        loaded_data1 = pickle.load(fp)
    print('%s loaded, %s ' % (name, type(loaded_data1)))
    return loaded_data1

def dumpPickle(d, name, path = kPathFiles):
    
    with open(path + name, 'wb') as output:
    # Pickle dictionary using protocol 0.
        pickle.dump(d, output)
    print('%s Saved' % (name))
    
def loadH5(name, path = kPathFiles):  
    
    # load data from pkl file
    with h5py.File(path + name,'r') as h5f:
        b = h5f[name.split('.')[0]][:]
#     h5f.close()
    
    print('%s loaded, %s ' % (name, type(b)))
    
    return b

def saveH5(d, name, path = kPathFiles):
    with h5py.File(path + name, 'w') as h5f:
        h5f.create_dataset(name.split('.')[0], data = d)
#     h5f.close() 
    print('%s Saved' % (name))


def saveDataset(dataset, name):
    for i in range(len(dataset)-2):
        for j in kLabels:
            saveH5(dataset[i+2][j], name+'_'+str(i)+'_'+j+'.h5')
            
def loadDataset(name):
#     data = []
    rawMagnitude = OrderedDict()
    stemMagnitude = OrderedDict()
    rawPhase = OrderedDict()
    stemPhase = OrderedDict()
    for i in range(4):
        for j in kLabels:
            if i == 0:
                rawMagnitude[j] = (loadH5(name+'_'+str(i)+'_'+j+'.h5'))
            elif i == 1:
                stemMagnitude[j] = (loadH5(name+'_'+str(i)+'_'+j+'.h5'))
            elif i == 2:
                rawPhase[j] = (loadH5(name+'_'+str(i)+'_'+j+'.h5'))
            elif i == 3:
                stemPhase[j] = (loadH5(name+'_'+str(i)+'_'+j+'.h5'))
    
    return rawMagnitude, stemMagnitude, rawPhase, stemPhase


startTime = datetime.now()

gRawPath = loadPickle('gRawPath.pkl', path = kPathFiles)
gStemPath = loadPickle('gStemPath.pkl', path = kPathFiles)
gStemStereoPath = loadPickle('gStemStereoPath.pkl', path = kPathFiles)
gXtrain = loadPickle('gXtrain.pkl', path = kPathFiles)
gXtrainAug = loadPickle('gXtrainAug.pkl', path = kPathFiles)
gXtest = loadPickle('gXtest.pkl', path = kPathFiles)
gYtrainAug = loadPickle('gYtrainAug.pkl', path = kPathFiles)
gYtest = loadPickle('gYtest.pkl', path = kPathFiles)

print('\nExecuted in: {} \n'.format(str(datetime.now() - startTime))

def load_sound_files(file_paths, sr = kSR, loudnessNorm = True):
    raw_sounds = []
    for fp in file_paths:
        
#         print(fp)
        X,_sr = librosa.load(fp, sr=sr)

        if loudnessNorm:
            gain_lufs = (-23.0 - calculate_loudness(X, kSR))
            gain = np.power(10, gain_lufs/20)
            X = gain * X
            
        if X.shape[0] < 441000:
                
            X_ = np.zeros(441000)
            X_[:X.shape[0]] = X
            X = X_
            
        raw_sounds.append(X)
        
    return raw_sounds

def spectogram(audio_files, sound_clip, sr=kSR, n_fft=kN,
               hop_length=kHopLength):
    spectograms = []
    phase = []

    audio_files = get_name(audio_files)
             
    for f in range(len(audio_files)):

        D = librosa.stft(sound_clip[f], n_fft=n_fft, hop_length=hop_length)
        spectograms.append(np.abs(D))
        phase.append(np.angle(D))

    return np.asarray(spectograms), np.asarray(phase)
  


def get_name(audio_files):
    name = []
    for i in range(len(audio_files)):
        n = str(audio_files[i]).split('/')[len(str(audio_files[i]).split('/'))-1].split('.wav')[0]
        name.append(n)
    return name



# Plots stem and raw track given a name.
def plotStemRawTrack(t1, t2):
  
    plt.close()
    ax = plt.subplot(111)
    plt.rcParams['figure.figsize'] = (9,5)
    Time1=np.linspace(0, len(t1)/kSR, num=len(t1))
    
    
    lines1, = ax.plot(Time1, t2,'k', label='stem', alpha=1)
    lines2, = ax.plot(Time1, t1, 'c', label='raw',alpha=0.5)
    
    #Sets legend outside plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(handles=[lines1, lines2],loc='center', bbox_to_anchor=(0.8, -0.1),
          fancybox=True, shadow=True, ncol=2)

    plt.xlabel('time (s)')
    plt.ylabel('amplitude')     
    
# Plot spectrum
def plotStemRawSpectrum(rD, sD, type = 'spectrum', hop_size = kHopLength,
                        power = True, yaxis = 'log', cmap = 'summer',
                        colorbar = False):
    plt.close()
    
#     rD = gRawMagnitude[idx]
#     sD = gStemMagnitude[idx]
    
    if 'spectrum' in type:
        rD = librosa.logamplitude(np.abs(rD)**2, ref_power=np.nanmax)
        sD = librosa.logamplitude(np.abs(sD)**2, ref_power=np.nanmax)
  
    if power:
        rD = rD**2
        sD = sD**2
  
    plt.rcParams['figure.figsize'] = (9,10) 
#     plt.subplots_adjust(hspace=1.5)
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('Raw')
    librosa.display.specshow(rD,
                             cmap=cmap, sr = kSR, hop_length = hop_size,
                             y_axis=yaxis, x_axis='time')
                            
    if colorbar:
        plt.colorbar()
        lim = np.nanmax(np.abs(rD))
        plt.clim(0,lim)
    
    ax2 = plt.subplot(2,1,2)
    ax2.set_title('Stem')
    librosa.display.specshow(sD,
                             cmap=cmap, sr = kSR, hop_length = hop_size,
                             y_axis=yaxis, x_axis='time')
    if colorbar:
        lim = np.nanmax(np.abs(rD))
        plt.clim(0,lim)
        plt.colorbar()
        
        
#METHODS FOR LOUDNESS NORMALISATION

def calculate_loudness(signal,fs):
    # filter
    if len(signal.shape)==1: # if shape (N,), then make (N,1)
        signal_filtered = copy.copy(signal.reshape((signal.shape[0],1)))
    else:
        signal_filtered = copy.copy(signal)
        
    for i in range(signal_filtered.shape[1]):
        signal_filtered[:,i] = K_filter(signal_filtered[:,i], fs)

    # mean square
    G = [1.0, 1.0, 1.0, 1.41, 1.41]
    T_g = 0.400 # 400 ms gating block
    Gamma_a = -70.0 # absolute threshold: -70 LKFS
    overlap = .75 # relative overlap (0.0-1.0)
    step = 1 - overlap

    T = signal_filtered.shape[0]/fs # length of measurement interval in seconds
    j_range = np.arange(0,(T-T_g)/(T_g*step))
    z = np.ndarray(shape=(signal_filtered.shape[1],len(j_range))) # ?
    # write in explicit for-loops for readability and translatability
    for i in range(signal_filtered.shape[1]): # for each channel i
        for j in j_range.astype(int): # for each window j
            lbound = np.round(fs*T_g*j*step).astype(int)
            hbound = np.round(fs*T_g*(j*step+1)).astype(int)
            z[i,j] = (1/(T_g*fs))*np.sum(np.square(signal_filtered[lbound:hbound, i]))

    G_current = np.array(G[:signal_filtered.shape[1]]) # discard weighting coefficients G_i unused channels
    n_channels = G_current.shape[0]
    l = [-.691 + 10.0*np.log10(np.sum([G_current[i]*z[i,j.astype(int)] for i in range(n_channels)]))              for j in j_range]
    #print 'l: '+str(l)

    # throw out anything below absolute threshold:
    indices_gated = [idx for idx,el in enumerate(l) if el > Gamma_a] 
    z_avg = [np.mean([z[i,j] for j in indices_gated]) for i in range(n_channels)]
    Gamma_r = -.691 + 10.0*np.log10(np.sum([G_current[i]*z_avg[i] for i in range(n_channels)])) - 10.0
    # throw out anything below relative threshold:
    indices_gated = [idx for idx,el in enumerate(l) if el > Gamma_r] 
    z_avg = [np.mean([z[i,j] for j in indices_gated]) for i in range(n_channels)]
    L_KG = -.691 + 10.0*np.log10(np.sum([G_current[i]*z_avg[i] for i in range(n_channels)]))

    return L_KG

def K_filter(signal, fs, debug=False):
    # apply K filtering as specified in EBU R-128 / ITU BS.1770-4
       
    # pre-filter 1
    f0 = 1681.9744509555319
    G  = 3.99984385397
    Q  = 0.7071752369554193
    K  = np.tan(np.pi * f0 / fs) # TODO: precompute
    Vh = np.power(10.0, G / 20.0)
    Vb = np.power(Vh, 0.499666774155)
    a0_ = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_
    b1 = 2.0 * (K * K -  Vh) / a0_
    b2 = (Vh - Vb * K / Q + K * K) / a0_
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_
    signal_1 = lfilter([b0,b1,b2],[a0,a1,a2],signal)
    
    if debug:
        plt.figure(figsize=(9,9))
        ax1 = fig.add_subplot(111)
        w, h1 = freqz([b0,b1,b2], [a0,a1,a2], worN=8000)#np.logspace(-4, 3, 2000))
        plt.semilogx((fs * 0.5 / np.pi) * w, 20*np.log10(abs(h1)))
        plt.title('Pre-filter 1')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim([20, 20000])
        plt.ylim([-10,10])
        plt.grid(True, which='both')
        ax = plt.axes()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        plt.show()
    
    # pre-filter 2
    f0 = 38.13547087613982
    Q  =  0.5003270373253953
    K  = np.tan(np.pi * f0 / fs)
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0
    signal_2 = lfilter([b0,b1,b2],[a0,a1,a2],signal_1)
    
    if debug:
        plt.figure(figsize=(9,9))
        ax1 = fig.add_subplot(111)
        w, h2 = freqz([b0,b1,b2], [a0,a1,a2], worN=8000)#np.logspace(-4, 3, 2000))
        plt.semilogx((fs * 0.5 / np.pi) * w, 20*np.log10(abs(h2)))
        plt.title('Pre-filter 2')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim([10, 20000])
        plt.ylim([-30,5])
        plt.grid(True, which='both')
        ax = plt.axes()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        plt.show()
    
    return signal_2 # return signal passed through 2 pre-filters        


# In[7]:

def getAudioMagnitudePhase(rawPath, stemPath, loudnessNorm = True):
    
    rawAudio = OrderedDict()
    stemAudio = OrderedDict()
    rawMagnitude = OrderedDict()
    stemMagnitude = OrderedDict()
    rawPhase = OrderedDict()
    stemPhase = OrderedDict()
    
    for k in kLabels:
        
        

        rawPath['b'] = [ x for x in rawPath['b'] if 'EthanHein_BluesForNofi' not in x ]
        stemPath['b'] = [ x for x in stemPath['b'] if 'EthanHein_BluesForNofi' not in x ]
#         rawPath['b'] = [ x for x in rawPath['b'] if 'Lushlife_ToynbeeSuite2' not in x ]
#         stemPath['b'] = [ x for x in stemPath['b'] if 'Lushlife_ToynbeeSuite2' not in x ]
#         rawPath['b'] = [ x for x in rawPath['b'] if 'EthanHein_HarmonicaFigure' not in x ]
#         stemPath['b'] = [ x for x in stemPath['b'] if 'EthanHein_HarmonicaFigure' not in x ]
        rawPath['g'] = [ x for x in rawPath['g'] if 'EthanHein_GirlOnABridge' not in x ]
        stemPath['g'] = [ x for x in stemPath['g'] if 'EthanHein_GirlOnABridge' not in x ]
        rawPath['g'] = [ x for x in rawPath['g'] if 'BigTroubles_Phantom' not in x ]
        stemPath['g'] = [ x for x in stemPath['g'] if 'BigTroubles_Phantom' not in x ]
        rawPath['v'] = [ x for x in rawPath['v'] if 'Wolf_DieBekherte' not in x ]
        stemPath['v'] = [ x for x in stemPath['v'] if 'Wolf_DieBekherte' not in x ]
        rawPath['v'] = [ x for x in rawPath['v'] if 'MatthewEntwistle_Lontano' not in x ]
        stemPath['v'] = [ x for x in stemPath['v'] if 'MatthewEntwistle_Lontano' not in x ]
        rawPath['k'] = [ x for x in rawPath['k'] if 'JoelHelander_Definition' not in x ]
        stemPath['k'] = [ x for x in stemPath['k'] if 'JoelHelander_Definition' not in x ]

        print('Loading {} raw audio files \n'.format(k))
        rawAudio[k] = load_sound_files(rawPath[k], loudnessNorm=loudnessNorm)
        
        print('Loading {} stem audio files \n'.format(k))
        stemAudio[k] = load_sound_files(stemPath[k], loudnessNorm=loudnessNorm)

        print('Obtaining {} raw magnitude and phase \n'.format(k))
        rawMagnitude[k], rawPhase[k] = spectogram(rawPath[k], rawAudio[k])
        
        print('Obtaining {} stem magnitude and phase \n'.format(k))
        stemMagnitude[k], stemPhase[k] = spectogram(stemPath[k], stemAudio[k]) 
        
    return rawAudio, stemAudio, rawMagnitude, stemMagnitude, rawPhase, stemPhase


startTime = datetime.now()

Data = getAudioMagnitudePhase(gRawPath, gStemPath)

gXAudio = Data[0]
gYAudio = Data[1]
gXMagnitude = Data[2]
gYMagnitude = Data[3]
gXPhase = Data[4]
gYPhase = Data[5]

print('\nExecuted in: {} \n'.format(str(datetime.now() - startTime)),
     '{} Bass r/s tracks \n'.format(gYPhase['b'].shape[0]),
     '{} Guitar r/s tracks \n'.format(gYPhase['g'].shape[0]),
     '{} Keys r/s tracks \n'.format(gYPhase['k'].shape[0]),
     '{} Vocal r/s tracks \n'.format(gYPhase['v'].shape[0])) 


def DAE(neurons=128, init_mode='normal', act_function='relu', dropout_rate=0.3, weight_constraint=0):
    # create model
	model = Sequential()
    
	model.add(Dense(1025, input_dim=1025,
                    kernel_initializer=init_mode,
                    activation=act_function))
	model.add(Dropout(dropout_rate))
                    
        model.add(Dense(neurons*4, kernel_initializer=init_mode, activation=act_function))
        model.add(Dropout(dropout_rate))
                    
        model.add(Dense(neurons*2, kernel_initializer=init_mode, activation=act_function))
        model.add(Dropout(dropout_rate))
                    
        model.add(Dense(neurons*4, kernel_initializer=init_mode, activation=act_function))
        model.add(Dropout(dropout_rate))
                    
        model.add(Dense(1025, kernel_initializer=init_mode, activation=act_function))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
       	return model
# Gridsearch Nerual net.
gridResult2 = {}


for k in kLabels:
    
    startTime = datetime.now()
    label = k
    X = gXMagnitude[k].reshape(-1,gXMagnitude[k].shape[1])
    Y = gYMagnitude[k].reshape(-1,gYMagnitude[k].shape[1])


    batch_size = [10, 20, 40, 50, 60, 70]
    epochs = [50, 100, 200]
    init_mode = ['lecun_uniform', 'normal']
    activation = ['selu','relu','elu','tanh','sigmoid']
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4]
    neurons = [32, 64, 128, 256]


    est = KerasRegressor(build_fn=DAE, verbose=1)



    param_grid = dict(neurons=neurons, epochs=epochs)



    grid = GridSearchCV(estimator=est, param_grid=param_grid, n_jobs=1, verbose=1)
    grid_result = grid.fit(X, X)



    print("\n label - %s - Best: %f using %s" % (label, grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    gridResult2[label] = ("\n label - %s - Best: %f using %s" % (label, grid_result.best_score_, grid_result.best_params_))

    gridResult2[label+'mean']=means
    gridResult2[label+'stds']=means
    gridResult2[label+'params']=params
    dumpPickle(gridResult2, 'grid2-'+label+'.pkl')

    print('\nExecuted in: {} \n'.format(str(datetime.now() - startTime)))


