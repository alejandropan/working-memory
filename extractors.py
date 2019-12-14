#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:53:22 2019

@author: ibladmin
"""

from ibllib.io.extractors.ephys_fpga import *
from pathlib import Path
from brainbox.core import Bunch
import matplotlib.pyplot as plt
import ibllib.plots as plots
import numpy as np

# Function definitions

def extract_camera_sync(sync, output_path=session, save=False, chmap=None):
    """
    Extract camera timestamps from the sync matrix
    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace
    :param output_path: where to save the data
    :param save: True/False
    :param chmap: dictionary containing channel indices. Default to constant.
    :return: dictionary containing camera timestamps
    """
    if chmap==None:
        chmap = {'trial_start': 0,
            'sample': 1,
            'delay': 2,
            'choice': 3,
            'outcome': 4,
            'opto': 5,
            'right_lever': 6,
            'imec': 7,
            'nosepoke': 22,
            'reward_pump' :21,
            'reward_port':23,
            'camera':16
            }
    
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()
    s = _get_sync_fronts(sync, chmap['camera'])
    np.save(output_path / '_Camera.times.npy', s.times[::2])
    
    
def cut_odd_events(end_time, event, save = True):
    """
    
    Cuts sync pulses so that they are even (e.g recording stop before the last
    trial ended)
    :param end_time:  Time in seconds of the end of the last trial
    :param event:  Syncing event to be corrected. It has to be a bunch object 
    (see brainbox.core bunch)
    :return: even syncing for the event (bunch object )
    """
    included_fronts  = tuple([event['times'] <= end_time])
    event['times'] =  event['times'][included_fronts]
    event['polarities'] =  event['polarities'][included_fronts]
    
    return event

def extract_behaviour_sync(sync, output_path=None, save=False, chmap=None):
    """
    Extract wheel positions and times from sync fronts dictionary
    :param sync: dictionary 'times', 'polarities' of fronts detected on sync 
    trace for all 16 chans
    :param output_path: where to save the data
    :param save: True/False
    :param chmap: dictionary containing channel index. Default to constant.
    :return: trials dictionary
    """
    if chmap==None:
        chmap = {'trial_start': 0,
            'sample': 1,
            'delay': 2,
            'choice': 3,
            'outcome': 4,
            'opto': 5,
            'right_lever': 6,
            'imec': 7,
            'nosepoke': 22,
            'reward_pump' :21,
            'reward_port':23,
            'camera':16
            }
        
    # Get fronts
    trial = _get_sync_fronts(sync, chmap['trial_start'])
    sample = _get_sync_fronts(sync, chmap['sample'])
    delay = _get_sync_fronts(sync, chmap['delay'])
    choice = _get_sync_fronts(sync, chmap['choice'])
    outcome = _get_sync_fronts(sync, chmap['outcome'])
    opto = _get_sync_fronts(sync, chmap['opto'])
    right_lever = _get_sync_fronts(sync, chmap['right_lever'])
    nosepoke = _get_sync_fronts(sync, chmap['nosepoke'])
    reward_pump = _get_sync_fronts(sync, chmap['reward_pump'])
    reward_port = _get_sync_fronts(sync, chmap['reward_port'])
    
     # Fix for unfinished trials
    if np.count_nonzero(trial['times']) % 2 != 0 :
        print('Warning: Unfinished trial, cutting last trial')
        new_end = trial['times'][-2]
        trial =  cut_odd_events(new_end,trial)
        sample =  cut_odd_events(new_end,sample)
        delay = cut_odd_events(new_end,delay)
        choice = cut_odd_events(new_end,choice)
        outcome = cut_odd_events(new_end,outcome)
        opto = cut_odd_events(new_end,opto)
        right_lever = cut_odd_events(new_end,right_lever)
        nosepoke = cut_odd_events(new_end,nosepoke)
        reward_pump = cut_odd_events(new_end,reward_pump)
        reward_port = cut_odd_events(new_end,reward_port)
    
    # Divide by on and off
    trial_on = trial['times'][::2] # {'times' : trial['times'][::2], 'polarities' : trial['polarities'][::2]}
    trial_off = trial['times'][1::2]
    sample_on = sample['times'][::2]
    sample_off = sample['times'][1::2]
    delay_on = delay['times'][::2]
    delay_off = delay['times'][1::2]
    choice_on = choice['times'][::2]
    choice_off = choice['times'][1::2]
    outcome_on = outcome['times'][::2]
    outcome_off = outcome['times'][1::2]
    opto_on = opto['times'][::2]
    opto_off = opto['times'][1::2]
    right_lever_on = right_lever['times'][::2]
    right_lever_off = right_lever['times'][1::2]
    nosepoke_on = nosepoke['times'][::2]
    nosepoke_off = nosepoke['times'][1::2]
    reward_pump_on = reward_pump['times'][::2]
    reward_pump_off = reward_pump['times'][1::2]
    
    # Calculate some trial variables
  
    
    for t in len(trial_on):
        # Giving 1 ms leeway for syncing pulse error
        trial_on_t = trial_on[t]
        trial_off_ = trial_off[t] 
        
        if #To do fill nan for other time periods is not intrial and time if in trial
        #re calculate everything below with that
        sample_on_t = sample_on[t] 
        sample_off_t = sample_off[t] 
        delay_on_t = delay_on[t] 
        delay_off_t = delay_off[t] 
        choice_on_t = delay_on[t] 
        choice_off_t = delay_off[t] 
        
        trial_side[t] = 'R' if any(np.logical_and(right_lever_on>= sample_on_t - 0.001, 
                            right_lever_on<=sample_off_t + 0.001)) else 'L'
        opto_trial[t] = True if any(np.logical_and(opto_on>= trial_on_t, 
                            opto_on<=trial_off_t)) else False
        if opto_trial == True:
            if any(np.logical_and(opto_on>= sample_on_t, 
                            opto_on<=sample_off_t)):
                opto_event[t] = 'S' 
            if any(np.logical_and(opto_on>= delay_on_t, 
                            opto_on<=delay_off_t)):
                opto_event[t] =  'D'
            if any(np.logical_and(opto_on>= choice_on_t, 
                            opto_on<=choice_off_t)):
                opto_event[t] =  'C'
        
        
    
    events = {'trial': trial, 'delay': delay, 'choice': choice, 'outcome':
        outcome, 'opto': opto, 'right_lever': right_lever,'nosepoke': nosepoke,
        'reward_pump': reward_pump, 'reward_port': reward_port}
    
    #Assertion QC
    assert np.count_nonzero(trial['times']) % 2 ==0,'ERROR: Uneven trial fronts'
    assert np.count_nonzero(trial['times']) % 2 ==0,'ERROR: Uneven trial fronts'

    if DEBUG_PLOTS:
        plt.figure()
        ax = plt.gca()
        plots.squares(trial['times'], trial['polarities'] * 0.4 + 1,
                      ax=ax, label='trial=1', color='k')
        plots.squares(frame2ttl['times'], frame2ttl['polarities'] * 0.4 + 2,
                      ax=ax, label='frame2ttl=2', color='k')
        plots.squares(audio['times'], audio['polarities'] * 0.4 + 3,
                      ax=ax, label='audio=3', color='k')
        plots.vertical_lines(t_ready_tone_in, ymin=0, ymax=4,
                             ax=ax, label='ready tone in', color='b', linewidth=0.5)
        plots.vertical_lines(t_trial_start, ymin=0, ymax=4,
                             ax=ax, label='start_trial', color='m', linewidth=0.5)
        plots.vertical_lines(t_error_tone_in, ymin=0, ymax=4,
                             ax=ax, label='error tone', color='r', linewidth=0.5)
        plots.vertical_lines(t_valve_open, ymin=0, ymax=4,
                             ax=ax, label='valve open', color='g', linewidth=0.5)
        plots.vertical_lines(t_stim_freeze, ymin=0, ymax=4,
                             ax=ax, label='stim freeze', color='y', linewidth=0.5)
        plots.vertical_lines(t_stim_off, ymin=0, ymax=4,
                             ax=ax, label='stim off', color='c', linewidth=0.5)
        ax.legend()


def _get_sync_fronts(sync, channel_nb):
    return Bunch({'times': sync['times'][sync['channels'] == channel_nb],
                  'polarities': sync['polarities'][sync['channels'] == channel_nb]})


_, sync  = extract_sync(session_path, save=False, force=False, ephys_files=None)

        
extract_camera_sync(sync, output_path=session, save=True, chmap=sync_chmap)
extract_behaviour_sync(sync, output_path=None, save=True, chmap=None)






