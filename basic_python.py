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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

trial_on = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_on.npy')
trial_off = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_off.npy')
completed  = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_completed.npy')
spikes = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/spikes.times.npy')
spike_id = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/spikes.clusters.npy')
sample_on = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_sample_on.npy')
sample_off = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_sample_off.npy')
delay_on = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_delay_on.npy')
delay_off = np.load('/Volumes/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/20191205/raw_ephys_data000_1_g0/_trial_delay_off.npy')
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


def spikes_2_relative_time (mega_pd, event, event_on, leeway =0):
    mega_pd.loc['Relative_time'] = np.nan
    for i in mega_pd[event].unique()[1:]: #skip nan
        event0 = event_on[int(i)] - leeway
        mega_pd.loc[mega_pd[event] == int(i), 'Relative_time'] = \
        mega_pd.loc[mega_pd[event] == int(i), 'Time'] - event0
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

def bin_df(df,n_bin):
    """
    Bins relative times in a dataframe
    """
    df['Relative_time'].max()
    
    pd.cut(df['Relative_time'], bins=n_bin)
    

    
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





mega_pd  = spikes_2_relative_time (mega_pd, 'Delay', sample_on, leeway =0)
good = mega_pd.loc[(mega_pd['isolation_quality'] == 'good') &
            (mega_pd['Completed'] == True), :]

sns.distplot(good.loc[(good['Sample']>=0) & (good['opto_event'] == 'S') , 'Relative_time'], color = 'B')
sns.distplot(good.loc[(good['Sample']>=0) & (good['Laser'] == False) , 'Relative_time'], color = 'R')



seaborn.heatmap(, , )
(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, 
 annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white',
 cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', 
 yticklabels='auto', mask=None, ax=None, **kwargs)¶


##############
    
def Isomap_colored(binned_data):
    # obs_limit=1000 #else it's too slow
    X = binned_data
    Y = Isomap(n_components=3).fit_transform(X)
    x, y, z = np.split(Y, 3, axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(x, y, z, s=20, alpha=0.25, cmap='ocean')
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


##### Make script
    
    
## Prepare Data for Histogram

# df = dataframe with selection of cells of interest

#from med associates
    
right_trials = 


#Cut to period of interest and calculate relative time
epoch = 'Delay'
df = spikes_2_relative_time (df, epoch, delay_on, leeway = 0)
df = df.loc[df['Delay'] >0]

# Classify by epoch and choose ms delay
for i in df['Delay'].unique():
   df.loc[df['Delay'] == i, 'Dperiod'] = df.loc[df['Delay'] == i, 'Time'].max() - df.loc[df['Delay'] == i, 'Time'].min() 

df =  df.loc[(df['Dperiod']< 5.002) & (df['Dperiod']> 4.88)]


df['binned'] = pd.cut(df['Relative_time'], bins=100)

right = med_assoc.loc[med_assoc['trialtype'] == 'right', 'trial']
left = med_assoc.loc[med_assoc['trialtype'] == 'left', 'trial']

df.loc[(df['Trial']>51) &(df['Trial']<75), ['Delay', 'Trial']] = df.loc[(df['Trial']>51) &(df['Trial']<75), ['Delay', 'Trial']] + 1
df.loc[(df['Trial']>75), ['Delay', 'Trial']] = df.loc[(df['Trial']>75) , ['Delay', 'Trial']] + 1


df.loc[df['Trial'].isin(right), 'side'] = 'R'
df.loc[df['Trial'].isin(left), 'side'] = 'L'

df_light = df.dropna(subset=['binned'])


laser = 'D'

df_light = df_light.loc[df_light['opto_event']== laser]

df_light = df_light.loc[df_light['Laser']== False]


n_trials = len(df_light['Trial'].unique())

x_left = np.zeros([len(df_light['ID'].unique()), len(df_light['binned'].unique())])   
for i , bn in enumerate(df_light['binned'].unique()):
    for j , ids in enumerate(df_light['ID'].unique()):
        x_left[j, i] = df_light.loc[(df_light['ID'] == int(ids)) & (df_light['side']
                                                == 'L') & (df_light['binned'] == bn), 
                        'Time'].count()/n_trials

x_right = np.zeros([len(df_light['ID'].unique()), len(df_light['binned'].unique())])   
for i , bn in enumerate(df_light['binned'].unique()):
    for j , ids in enumerate(df_light['ID'].unique()):
        x_right[j, i] = df_light.loc[(df_light['ID'] == int(ids)) & (df_light['side']
                                                == 'R') & (df_light['binned'] == bn), 
                        'Time'].count()/n_trials
        

#Horizontal concatenations
X  = np.concatenate([x_left,x_right], axis = 1)   

#Discard low firing rate
X = np.delete(X, np.where(np.mean(X, axis =1)<0.01), 0)


#Left right info for coloring
cor_l = [-1] * 100
cor_r = [1] * 100
cor = cor_l +cor_r

#For suffle if interested
X  = np.transpose(X)
cor = np.array([cor])
X = np.hstack([np.transpose(cor), X])
np.random.shuffle(X)
X = X.T
cor = X[0,:]
X = X[1:,:]


# Trajectories

Y = PCA(n_components=3, svd_solver='full').fit_transform(np.transpose(X))
x, y, z = np.split(Y, 3, axis=1)
fig = plt.figure()
fig = plt.figure()
ax = Axes3D(fig)
p = ax.scatter(x[:100], y[:100], z[:100], s=20, alpha=0.25, color='b')
p = ax.scatter(x[100:], y[100:], z[100:], s=20, alpha=0.25, color='r')
fig.colorbar(p)
plt.show()



Y = PCA(n_components=3, svd_solver='full').fit_transform(np.transpose(X))
x, y, z = np.split(Y, 3, axis=1)
fig = plt.figure()
plt.scatter(x[:100], y[:100],  s=20, alpha=0.25, color = 'b' )
plt.scatter(x[100:], y[100:], s=20, alpha=0.25, color ='r')
plt.title("Delay: Left vs Right All trials")
plt.show()


Y = PCA(n_components=3, svd_solver='full').fit_transform(np.transpose(X))
x, y, z = np.split(Y, 3, axis=1)
fig = plt.figure()
plt.scatter(x[:100], y[:100],  s=20, alpha=0.25, color = 'b' )
plt.scatter(x[100:], y[100:], s=20, alpha=0.25, color ='r')
plt.plot(x[:100], y[:100],   alpha=0.25, color = 'b' )
plt.plot(x[100:], y[100:],  alpha=0.25, color ='r')
plt.title("Delay: Left vs Right All trials")
plt.show()


#with opto vs non-opto

#no opto
df_light = df.dropna(subset=['binned'])


df_light = df_light.loc[df_light['opto_event'] != laser]


n_trials = len(df_light['Trial'].unique())

x_left = np.zeros([len(df_light['ID'].unique()), len(df_light['binned'].unique())])   
for i , bn in enumerate(df_light['binned'].unique()):
    for j , ids in enumerate(df_light['ID'].unique()):
        x_left[j, i] = df_light.loc[(df_light['ID'] == int(ids)) & (df_light['side']
                                                == 'L') & (df_light['binned'] == bn), 
                        'Time'].count()/n_trials

x_right = np.zeros([len(df_light['ID'].unique()), len(df_light['binned'].unique())])   
for i , bn in enumerate(df_light['binned'].unique()):
    for j , ids in enumerate(df_light['ID'].unique()):
        x_right[j, i] = df_light.loc[(df_light['ID'] == int(ids)) & (df_light['side']
                                                == 'R') & (df_light['binned'] == bn), 
                        'Time'].count()/n_trials

#Horizontal concatenations
X  = np.concatenate([x_left,x_right], axis = 1)   

#Discard low firing rate
X = np.delete(X, np.where(np.mean(X, axis =1)<0.05), 0)

Y = PCA(n_components=3, svd_solver='full').fit_transform(np.transpose(X))
x, y, z = np.split(Y, 3, axis=1)
fig = plt.figure()
plt.scatter(x[:100], y[:100],  s=20, alpha=0.25, color = 'b' )
plt.scatter(x[100:], y[100:], s=20, alpha=0.25, color ='r')
plt.scatter(x[0], y[0], s=20, alpha=0.25, color ='r')
plt.plot(x[:100], y[:100],   alpha=0.25, color = 'b' )
plt.plot(x[100:], y[100:],  alpha=0.25, color ='r')
plt.title("Delay: Left vs Right Non Opto")
plt.show()

        
        
# opto        

df_light = df.dropna(subset=['binned'])


laser = 'D'

df_light = df_light.loc[df_light['opto_event']== laser]


n_trials = len(df_light['Trial'].unique())

x_left = np.zeros([len(df_light['ID'].unique()), len(df_light['binned'].unique())])   
for i , bn in enumerate(df_light['binned'].unique()):
    for j , ids in enumerate(df_light['ID'].unique()):
        x_left[j, i] = df_light.loc[(df_light['ID'] == int(ids)) & (df_light['side']
                                                == 'L') & (df_light['binned'] == bn), 
                        'Time'].count()/n_trials

x_right = np.zeros([len(df_light['ID'].unique()), len(df_light['binned'].unique())])   
for i , bn in enumerate(df_light['binned'].unique()):
    for j , ids in enumerate(df_light['ID'].unique()):
        x_right[j, i] = df_light.loc[(df_light['ID'] == int(ids)) & (df_light['side']
                                                == 'R') & (df_light['binned'] == bn), 
                        'Time'].count()/n_trials

#Horizontal concatenations
X  = np.concatenate([x_left,x_right], axis = 1)   

#Discard low firing rate
X = np.delete(X, np.where(np.mean(X, axis =1)<0.05), 0)

Y = PCA(n_components=3, svd_solver='full').fit_transform(np.transpose(X))
x, y, z = np.split(Y, 3, axis=1)
fig = plt.figure()
plt.scatter(x[:100], y[:100],  s=20, alpha=0.25, color = 'b' )
plt.scatter(x[100:], y[100:], s=20, alpha=0.25, color ='r')
plt.plot(x[:100], y[:100],   alpha=0.25, color = 'b' )
plt.plot(x[100:], y[100:],  alpha=0.25, color ='r')
plt.title("Delay: Left vs Right Opto")
plt.show()





#### PCA with nx trial (rows) timestamps  columns

x_total = np.zeros([len(df_light['ID'].unique())*len(df_light['Delay'].unique())
                    , len(df_light['binned'].unique())])   

for j , ids in enumerate(df_light['ID'].unique()):
    for t , dley in enumerate(df_light['Delay'].unique()):
        for i , bn in enumerate(df_light['binned'].unique()):
            x_total[j+t, i] = df_light.loc[(df_light['ID'] == int(ids)) &  \
                                           (df_light['binned'] == bn) \
                                           & (df_light['Delay'] == dley),  \
                                           'Time'].count()
                
trial_id =[]


for j , ids in enumerate(df_light['ID'].unique()):
    for t , dley in enumerate(df_light['Delay'].unique()):
            trial_id.append(df_light.loc[(df_light['ID'] == int(ids)) \
                                           & (df_light['Delay'] == dley),  \
                                           'side'].unique() == 'R')


Y = PCA(n_components=3, svd_solver='full').fit_transform(np.transpose(X))
x, y, z = np.split(Y, 3, axis=1)
zline = np.linspace(0, 15, 1000)
fig = plt.figure()
plt.scatter(x[trial_id==True], y[trial_id==True],  s=20, alpha=0.25, color = 'b' )
plt.scatter(x[trial_id==False], y[trial_id==False], s=20, alpha=0.25, color ='r')
plt.title("Delay: Left vs Right")
plt.show()

#### PCA with like the first one without averaging

x_total = np.zeros([len(df_light['ID'].unique()), len(df_light['binned'].unique()) \
                   * len(df_light['Delay'].unique())])
for t , dley in enumerate(df_light['Delay'].unique()):
    for i , bn in enumerate(df_light['binned'].unique()):
        for j , ids in enumerate(df_light['ID'].unique()):
            x_total[j, i] = df_light.loc[(df_light['ID'] == int(ids)) \
                                         & (df_light['binned'] == bn) \
                                         & (df_light['binned'] == bn)
                                         & (df_light['Delay'] == dley), \
                                          'Time'].count()

r_trials = []
l_trials = []
for t , dley in enumerate(df_light['Delay'].unique()):
    for i , bn in enumerate(df_light['binned'].unique()):
        for j , ids in enumerate(df_light['ID'].unique()):
            if df_light.loc[(df_light['ID'] == int(ids)) \
                                         & (df_light['binned'] == bn) \
                                         & (df_light['Delay'] == dley), \
                                          'side'].unique() == 'R':
                r_trials.append(int(dley))
            else:
                l_trials.append(int(dley))
                
Y = PCA(n_components=3, svd_solver='full').fit_transform(np.transpose(x_total))         
x, y, z = np.split(Y, 3, axis=1)          
plt.scatter(x[r_trials], y[r_trials],  s=20, alpha=0.25, color = 'b' )
plt.scatter(x[l_trials], y[l_trials],  s=20, alpha=0.25, color = 'r' )


ax = Axes3D(fig)
ax.scatter(x, y,z,  s=20, alpha=0.25, color = 'b' )
plt.title("Delay: Left vs Right")
plt.show()