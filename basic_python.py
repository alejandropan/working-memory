#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:31:36 2019

@author: alex
"""
import csv
import pandas as pd
from brainbox.processing import bincount2D
import numpy as np
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import seaborn as sns
 

trial_on = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_on.npy')
trial_off = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_off.npy')
completed  = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_completed.npy')
spikes = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/spikes.times.npy')
spike_id = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/spikes.clusters.npy')
sample_on = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_sample_on.npy')
sample_off = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_sample_off.npy')
delay_on = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_delay_on.npy')
delay_off = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_delay_on.npy')
choice_on = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_choice_on.npy')
choice_off = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_choice_off.npy')
opto = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_opto_trial.npy') 
event = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_opto_event.npy') 
metrics  = pd.read_csv('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/clusters.metrics.csv')
spike_metrics = map_qual_2_ID(spike_id,metrics)
# Functions that classify per trial
def within_event(spikes, event_on, event_off, cushion = 0):
    in_event_spikes = []
    in_event_ids = []
    for i in range(len(event_on)):
        if not np.isnan(event_on[i])  and not np.isnan(event_off[i]):
            in_event_spikes.append(spikes[np.logical_and(spikes >= event_on[i] 
                                                         - cushion, 
                                                         spikes <= event_off[i]
                                                         + cushion)])
            in_event_ids.append(spike_id[np.logical_and(spikes >= event_on[i] 
                                                        - cushion, 
                                                         spikes <= event_off[i] 
                                                         + cushion)])
    return in_event_spikes, in_event_ids

def within_event_per_neuron (neuron_id,in_event_ids,in_event_spikes):
    neuron_event_activity =[]
    for j in range(len(in_event_spikes)):
        neuron_event_activity.append(in_event_spikes[j][in_event_ids[j] == neuron_id])
    
    return neuron_event_activity

# Functions that classify every spike

def event_2_spike(spikes, event_on, event_off):
    in_event_spikes = np.empty(len(spikes))
    in_event_spikes[:] = np.nan
    for i in range(len(event_on)):
        if not np.isnan(event_on[i])  and not np.isnan(event_off[i]):
           in_event_spikes[np.where(np.logical_and(spikes >= event_on[i],spikes <= event_off[i]))] = i
    return in_event_spikes


def spikes_2_relative_time (mega_pd, event, leeway =0):
    mega_pd['Relative_time'] = np.nan
    for i in range(mega_pd[event].nunique()):
        event0 = mega_pd.loc[mega_pd[event] == i, 'Time'].iloc[0] - leeway
        mega_pd.loc[mega_pd[event] == i, 'Relative_time'] = \
            mega_pd.loc[mega_pd[event] == i, 'Time'] - event0
    return mega_pd

        
def opto_2_opevent (list_opto,trial,event):    
    opto_event = pd.DataFrame({'opto_epoch' : np.empty(len(trial))}) # ["" for x in range(len(trial))] 
    opto_event[:] = np.nan
    for i in np.nditer(list_opto):
        for j in np.where(trial == i):
            opto_event.iloc[j] = event[i]
            
    return opto_event

def map_qual_2_ID(spike_id,metrics):
    """
    Maps metric label to all spikes ID
    """
    spike_metrics = []
    for i in spike_id:
        spike_metrics.append(metrics.iloc[i,-1])
        
    return spike_metrics

    
def mega_panda (spikes, spikes_id, event_time1, event_time2, opto, correct):
    trial = event_2_spike(spikes, trial_on, trial_off)
    sample = event_2_spike(spikes, sample_on, sample_off)
    delay = event_2_spike(spikes, delay_on, delay_off)
    choice = event_2_spike(spikes, choice_on, choice_off)
    list_completed  = np.where(completed == 1)
    complete = np.isin(trial, list_completed)
    list_opto = np.where(opto == 1)
    opto_trial = np.isin(trial, list_opto)
    opto_event = opto_2_opevent(list_opto,trial,event)
    mega_pd = pd.DataFrame(rows = [trial, sample, delay,choice, complete
                           , opto_trial])
    
    mega_pd = pd.DataFrame({'Time': spikes,'ID':spike_id, 'Trial': trial, 'Sample': sample, 'Delay': delay, 'Choice':choice 
                        , 'Completed': complete, 'Laser': opto_trial}, columns=['Time','ID', 'Trial', 'Sample',
                        'Delay', 'Choice', 'Completed', 'Laser'])
    mega_pd['opto_event'] = opto_event
    mega_pd['isolation_quality'] = spike_metrics

    sample_clock= spikes_2_relative_time (sample, leeway =0)
    delay_clock = spikes_2_relative_time (delay, leeway =0)
    trial_clock = spikes_2_relative_time (trial, leeway =0)
    choice_clock = spikes_2_relative_time (choice, leeway =0)
    
mega_pd.loc[(mega_pd['isolation_quality'] == 'good') &
            (mega_pd['Completed'] == True), :]

mega_pd  = spikes_2_relative_time (mega_pd, 'Delay', leeway =0)





##############
    
def Isomap_colored(binned_data, bounds):
    # obs_limit=1000 #else it's too slow
    low, high = bounds
    X = binned_data['summed_spike_amps'][:, low:high].T
    Y = Isomap(n_components=3).fit_transform(X)
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(binned_data['wheel_velocity'][low:high]), cmap='ocean')
    fig.colorbar(p)
    plt.title("Isomap Alex's motor cortex recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()

def PCA_colored(binned_data, bounds):
    # obs_limit=1000 #else it's too slow
    low, high = bounds
    X = binned_data['summed_spike_amps'][:, low:high].T
    Y = PCA(n_components=3, svd_solver = 'full').fit_transform(X)
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, c=abs(binned_data['wheel_velocity'][low:high]), cmap='ocean')
    fig.colorbar(p)
    plt.title("PCA Alex's motor cortex recording vs wheel speed")
    # plt.scatter(Y.T[0,:],Y.T[1,:],s=1,alpha=0.9,c=D_trial['wheel_velocity'][trial])
    plt.show()



   