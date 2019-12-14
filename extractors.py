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
    
    #Assertion QC
    assert np.count_nonzero(trial['times']) % 2 ==0,'ERROR: Uneven trial fronts'
    assert np.count_nonzero(trial['times']) % 2 ==0,'ERROR: Uneven trial fronts'
    
    
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
    trial_trial_on = trial_trial_off = trial_sample_on = trial_sample_off = \
    trial_delay_on = trial_delay_off = trial_choice_on = trial_choice_off = \
    trial_right_lever_on = trial_right_lever_off = trial_reward_pump_on = \
    trial_reward_pump_off = np.empty(len(trial_on))
    
    trial_trial_on[:] = trial_trial_off[:] = trial_sample_on[:] = trial_sample_off[:] = \
    trial_delay_on[:] = trial_delay_off[:] = trial_choice_on[:] = trial_choice_off[:] = \
    trial_right_lever_on[:] = trial_right_lever_off[:] =  trial_reward_pump_on[:] = \
    trial_reward_pump_off[:] = np.nan
    
    #Empty matrix for variables with variable length per trial
    trial_outcome_on = trial_outcome_off = trial_opto_on = trial_opto_off = \
    trial_nosepoke_on = trial_nosepoke_off = []
    
    # Fill trial vectors
    trial_trial_on = trial_on
    trial_trial_off = trial_off
    trial_sample_on = sample_on
    trial_sample_off = sample_off
    
    assert len(trial_on) == len(trial_off) == len(sample_on) == len(sample_off) \
        , 'ERROR: Samples and trials dont match!'
    
    
    # Fill in trial vectors that require computation, giving 0.001 leeway for
    # some variables
    for t in range(len(trial_on)):
        # Giving 1 ms leeway for syncing pulse error
        # Variables that require logic computation
        trial_side[t] = 'R' if any(np.logical_and(right_lever_on>= sample_on[t] 
                      - 0.001, right_lever_on<=sample_off[t] + 0.001)) else 'L'
        opto_trial[t] = True if any(np.logical_and(opto_on>= trial_on[t], 
                            opto_on<=trial_off[t])) else False
        if opto_trial == True:
            if any(np.logical_and(opto_on>= sample_on[t], 
                            opto_on<=sample_off[t])):
                opto_event[t] = 'S' 
            if any(np.logical_and(opto_on>= delay_on_t, 
                            opto_on<=delay_off[t])):
                opto_event[t] =  'D'
            if any(np.logical_and(opto_on>= choice_on_t, 
                            opto_on<=choice_off[t])):
                opto_event[t] =  'C'
        
        #Variables that can only have one value per trial
        
        trial_delay_on[t] = delay_on[np.logical_and(delay_on>= trial_on[t] 
                      , delay_on<=trial_off[t])] if any(np.logical_and(delay_on
                        >= trial_on[t], delay_on <= trial_off[t])) else np.nan
        
        trial_delay_off[t] = delay_off[np.logical_and(delay_off>= trial_on[t] 
                      , delay_off<=trial_off[t])] if any(np.logical_and(delay_off
                        >= trial_on[t], delay_off <= trial_off[t])) else np.nan

        trial_choice_on[t] = choice_on[np.logical_and(choice_on>= trial_on[t] 
                      , choice_on<=trial_off[t])] if any(np.logical_and(choice_on
                        >= trial_on[t], choice_on <= trial_off[t])) else np.nan
        
        trial_choice_off[t] = choice_off[np.logical_and(choice_off>= trial_on[t] 
                      , choice_off<=trial_off[t])] if any(np.logical_and(choice_off
                        >= trial_on[t], choice_off <= trial_off[t])) else np.nan
        
        trial_reward_pump_on[t] = reward_pump_on[np.logical_and(reward_pump_on>= trial_on[t] 
                      , reward_pump_on<=trial_off[t])] if any(np.logical_and(reward_pump_on
                        >= trial_on[t], reward_pump_on <= trial_off[t])) else np.nan
        
        trial_reward_pump_off[t] = reward_pump_off[np.logical_and(reward_pump_off>= trial_on[t] 
                      , reward_pump_off<=trial_off[t])] if any(np.logical_and(reward_pump_off
                        >= trial_on[t], reward_pump_off <= trial_off[t])) else np.nan
        
        # Variables that can have more than on value per trial
        trial_outcome_on.append(outcome_on[np.logical_and(outcome_on>= trial_on[t] 
                      , outcome_on<=trial_off[t])] if any(np.logical_and(outcome_on
                        >= trial_on[t], outcome_on <= trial_off[t])) else np.nan)
        
        trial_outcome_off.append(outcome_off[np.logical_and(outcome_off>= trial_on[t] 
                      , outcome_off<=trial_off[t])] if any(np.logical_and(outcome_off
                        >= trial_on[t], outcome_off <= trial_off[t])) else np.nan)
        
        trial_opto_on.append(opto_on[np.logical_and(opto_on>= trial_on[t] 
                      , opto_on<=trial_off[t])] if any(np.logical_and(opto_on
                        >= trial_on[t], opto_on <= trial_off[t])) else np.nan)
        
        trial_opto_off.append(opto_off[np.logical_and(opto_off>= trial_on[t] 
                      , opto_off<=trial_off[t])] if any(np.logical_and(opto_off
                        >= trial_on[t], opto_off <= trial_off[t])) else np.nan)
        
        trial_nosepoke_on.append(nosepoke_on[np.logical_and(nosepoke_on>= trial_on[t] 
                      , nosepoke_on<=trial_off[t])] if any(np.logical_and(nosepoke_on
                        >= trial_on[t], nosepoke_on <= trial_off[t])) else np.nan)
        
        trial_nosepoke_off.append(nosepoke_off[np.logical_and(nosepoke_off>= trial_on[t] 
                      , nosepoke_off<=trial_off[t])] if any(np.logical_and(nosepoke_off
                        >= trial_on[t], nosepoke_off <= trial_off[t])) else np.nan)
        
        # Calculate vector of completed trials
        trial_completed = np.logical_and(trial_delay_on[t] != np.nan,trial_choice_on[t] != np.nan
                      , nosepoke_on<=trial_off[t])
        
        
        # Calculates vector of correct trials
        trial_correct = np.logical_and(trial_delay_on[t] != np.nan,trial_choice_on[t] != np.nan,
                       trial_reward_pump_on[t] != np.nan)
        
        # Calculate vector with extremes of outcome,opto and nosepoke
        trial_outcome_first = min(i) for i in trial_outcome_on[t]
        trial_outcome_last = min(i) for i in trial_outcome_off[t]
        trial_nosepoke_first = min(i) for i in trial_nosepoke_on[t]
        trial_nosepoke_last = min(i) for i in trial_nosepoke_off[t]
        trial_opto_first = min(i) for i in trial_opto_on[t]
        trial_opto_last = min(i) for i in trial_opto_off[t]
        
        if save == True:
            
    


def _get_sync_fronts(sync, channel_nb):
    return Bunch({'times': sync['times'][sync['channels'] == channel_nb],
                  'polarities': sync['polarities'][sync['channels'] == channel_nb]})


_, sync  = extract_sync(session_path, save=False, force=False, ephys_files=None)

        
extract_camera_sync(sync, output_path=session, save=True, chmap=sync_chmap)
extract_behaviour_sync(sync, output_path=None, save=True, chmap=None)






