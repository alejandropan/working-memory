#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:53:22 2019

@author: ibladmin
"""
from ibllib.io.extractors.ephys_fpga import  _get_sync_fronts
from ibllib.io.extractors.ephys_fpga import *
from pathlib import Path
from brainbox.core import Bunch
import ibllib.plots as plots
import numpy as np
import math
import sys
from pathlib import Path, PureWindowsPath
from ibllib.ephys import spikes


# Function definitions

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

    # Check and merge abnormal pulses
    for i  in [trial, sample, delay, choice, outcome, opto, 
               right_lever, nosepoke, reward_pump, reward_port]:
       if np.any(np.diff(i['times']) < 0.025):
           prom = np.where(np.diff(i['times']) < 0.025)
           prom =  np.array(prom)
           prom1  = prom + 1
           todel = np.concatenate((prom, prom1))
           todel = np.unique(todel)
           i['times'] = np.delete(i['times'],todel)
           i['polarities'] = np.delete(i['polarities'],todel)  
    
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
    reward_port_on = reward_port['times'][::2]
    reward_port_off = reward_port['times'][1::2]

    # Calculate some trial variables
    trial_trial_on = np.empty(len(trial_on))
    trial_trial_off = np.empty(len(trial_on))
    trial_opto_trial = np.empty(len(trial_on))
    trial_sample_on = np.empty(len(trial_on))
    trial_sample_off = np.empty(len(trial_on))
    trial_delay_on = np.empty(len(trial_on))
    trial_delay_off = np.empty(len(trial_on))
    trial_choice_on = np.empty(len(trial_on))
    trial_choice_off = np.empty(len(trial_on))
    trial_right_lever_on = np.empty(len(trial_on))
    trial_right_lever_off = np.empty(len(trial_on))
    trial_reward_pump_on = np.empty(len(trial_on))
    trial_reward_pump_off = np.empty(len(trial_on))
    trial_completed = np.empty(len(trial_on))
    trial_correct = np.empty(len(trial_on))
    trial_outcome_on = np.empty(len(trial_on))
    trial_outcome_off = np.empty(len(trial_on))
    trial_opto_first = np.empty(len(trial_on))
    trial_opto_last = np.empty(len(trial_on))
    trial_completed[:] = np.nan
    trial_correct[:] = np.nan
    trial_opto_first[:] = np.nan
    trial_opto_last[:] = np.nan
    trial_trial_on[:] = np.nan
    trial_trial_side = ['' for x in range(len(trial_on))]
    trial_opto_event = ['' for x in range(len(trial_on))]
    trial_opto_trial[:] = np.nan
    trial_trial_off[:] = np.nan
    trial_sample_on[:] = np.nan
    trial_sample_off[:] = np.nan
    trial_delay_on[:] = np.nan
    trial_delay_off[:] = np.nan
    trial_choice_on[:] = np.nan
    trial_choice_off[:] = np.nan
    trial_right_lever_on[:] = np.nan
    trial_right_lever_off[:] =  np.nan
    trial_reward_pump_on[:] = np.nan
    trial_reward_pump_off[:] = np.nan
    trial_outcome_on[:] = np.nan
    trial_outcome_off[:] = np.nan


    #Empty matrix for variables with variable length per trial
    trial_port_on = []
    trial_port_off = []
    trial_opto_on = []
    trial_opto_off = []
    trial_nosepoke_on = [] 
    trial_nosepoke_off = []

    # Fill trial vectors
    trial_trial_on = trial_on
    trial_trial_off = trial_off


    #assert len(trial_on) == len(trial_off) == len(sample_on) == len(sample_off) \
     #   , 'ERROR: Samples and trials dont match!'


    # Fill in trial vectors that require computation, giving 0.001 leeway for
    # some variables
    for t in range(len(trial_on)):

        #Variables that can only have one value per trial
        trial_sample_on[t] = sample_on[np.logical_and(sample_on>= trial_on[t] 
                      , sample_on<=trial_off[t])] if any(np.logical_and(sample_on
                        >= trial_on[t], sample_on <= trial_off[t])) else np.nan

        trial_sample_off[t] = sample_off[np.logical_and(sample_off>= trial_on[t] 
                      , sample_off<=trial_off[t])] if any(np.logical_and(sample_off
                        >= trial_on[t], sample_off <= trial_off[t])) else np.nan

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

        trial_outcome_on[t] = outcome_on[np.logical_and(outcome_on>= trial_on[t] 
                      , outcome_on<=trial_off[t])] if any(np.logical_and(outcome_on
                        >= trial_on[t], outcome_on <= trial_off[t])) else np.nan

        trial_outcome_off[t] = outcome_off[np.logical_and(outcome_off>= trial_on[t] 
                      , outcome_off<=trial_off[t])] if any(np.logical_and(outcome_off
                        >= trial_on[t], outcome_off <= trial_off[t])) else np.nan

        # Variables that can have more than on value per trial

        trial_port_on.append(reward_port_on[np.logical_and(reward_port_on>= trial_on[t] 
                      , reward_port_on<=trial_off[t])] if any(np.logical_and(reward_port_on
                        >= trial_on[t], reward_port_on <= trial_off[t])) else np.nan)

        trial_port_off.append(reward_port_off[np.logical_and(reward_port_off>= trial_on[t] 
                      , reward_port_off<=trial_off[t])] if any(np.logical_and(reward_port_off
                        >= trial_on[t], reward_port_off <= trial_off[t])) else np.nan)                                                                

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

        # Giving 1 ms leeway for syncing pulse error
        # Variables that require logic computation

        if math.isnan(trial_sample_on[t]) == True:
             trial_trial_side[t] = ''
        else:
            trial_trial_side[t] = 'R' if any(np.logical_and(right_lever_on>= (trial_sample_on[t] - 0.002)
                          , right_lever_on<=trial_sample_off[t])) else 'L'
        trial_opto_trial[t] = True if any(np.logical_and(opto_on>= trial_on[t], 
                            opto_on<=trial_off[t])) else False
        if trial_opto_trial[t] == True:
            if any(np.logical_and(opto_on>= trial_sample_on[t], 
                            opto_on<=trial_sample_off[t])):
                trial_opto_event[t] = 'S' 
            if any(np.logical_and(opto_on>= trial_delay_on[t], 
                            opto_on<=trial_delay_off[t])):
                trial_opto_event[t] =  'D'
            if any(np.logical_and(opto_on>= trial_choice_on[t], 
                            opto_on<=trial_choice_off[t])):
                trial_opto_event[t] =  'C'

        # Calculate vector of completed trials
        trial_completed[t] = not math.isnan(trial_outcome_on[t])


        # Calculates vector of correct trials
        trial_correct[t] = not math.isnan(trial_reward_pump_on[t])

        # Calculate vector with extremes of port,opto
        trial_opto_first[t] = min(trial_opto_on[t]) if not np.mean(np.isnan(trial_opto_on[t])) else np.nan
        trial_opto_last[t] = min(trial_opto_off[t]) if not np.mean(np.isnan(trial_opto_off[t])) else np.nan

    if save==True:
        np.save(output_path + '/' + '_trial_on.npy', trial_trial_on)
        np.save(output_path + '/' + '_trial_off.npy', trial_trial_off)
        np.save(output_path + '/' + '_trial_sample_on.npy', trial_sample_on)
        np.save(output_path + '/' + '_trial_sample_off.npy', trial_sample_off)
        np.save(output_path + '/' + '_trial_delay_on.npy', trial_delay_on)
        np.save(output_path + '/' + '_trial_delay_off.npy', trial_delay_off)
        np.save(output_path + '/' + '_trial_choice_on.npy', trial_choice_on)
        np.save(output_path + '/' + '_trial_choice_off.npy', trial_choice_off)
        np.save(output_path + '/' + '_trial_opto_event.npy', trial_opto_event)
        np.save(output_path + '/' + '_trial_reward_pump_on.npy', trial_reward_pump_on)
        np.save(output_path + '/' + '_trial_reward_pump_off.npy', trial_reward_pump_off)
        np.save(output_path + '/' + '_trial_outcome_on.npy', trial_outcome_on)
        np.save(output_path + '/' + '_trial_outcome_off.npy', trial_outcome_off)
        np.save(output_path + '/' + '_trial_port_on.npy', trial_port_on)
        np.save(output_path + '/' + '_trial_port_off.npy', trial_port_off)
        np.save(output_path + '/' + '_trial_opto_on.npy', trial_opto_on)
        np.save(output_path + '/' + '_trial_opto_off.npy', trial_opto_off)
        np.save(output_path + '/' + '_trial_nosepoke_on.npy', trial_nosepoke_on)
        np.save(output_path + '/' + '_trial_nosepoke_off.npy', trial_nosepoke_off)
        np.save(output_path + '/' + '_trial_trial_side.npy', trial_trial_side)
        np.save(output_path + '/' + '_trial_opto_trial.npy', trial_opto_trial)
        np.save(output_path + '/' + '_trial_completed.npy', trial_completed)
        np.save(output_path + '/' + '_trial_correct.npy', trial_correct)
        np.save(output_path + '/' + '_trial_opto_first.npy', trial_opto_first)
        np.save(output_path + '/' + '_trial_opto_last.npy', trial_opto_last)

def extract_camera_sync(sync, output_path=None, save=False, chmap=None):
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



def fix_keys(sync):
    '''
    Fixes keys of bunch objects to channels, polarirites and times
    :param sync : bunch object with channels, polarities and times
    :return : bunch file with standardized name

    '''
    sync['channels'] = sync.pop(sorted(sync)[0])
    sync['polarities'] = sync.pop(sorted(sync)[1])
    sync['times'] = sync.pop(sorted(sync)[2])
    
    return sync


if __name__ == "__main__":
    output_path=  str((sys.argv[1]))
    ks_path = Path(str(sys.argv[2]))
    _, sync  = extract_sync(output_path, save=True, force=False, ephys_files=None)
    synced = fix_keys(sync) # This will make sure indexing is correct
    extract_camera_sync(synced, output_path=output_path, save=True, chmap=None)
    extract_behaviour_sync(synced, output_path=output_path, save=True, chmap=None)
    spikes.ks2_to_alf(ks_path, Path(output_path), ampfactor=1, label=None, force=True)






