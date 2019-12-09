#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:53:22 2019

@author: ibladmin
"""

from ibllib.io import extractors
from pathlib import Path
from brainbox.core import Bunch
import matplotlib.pyplot as plt
import ibllib.plots as plots
import numpy as np

session = '/home/ibladmin/witten/Clare/Ephys/Neuropixel Ephys Data/360/20191204/raw_ephys_data000_1_g0'

sync  = extractors.ephys_fpga._get_main_probe_sync(session)

sync_chmap = {'trial_start': 0,
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
        delay = cut_odd_events(new_end,delay)
        choice = cut_odd_events(new_end,choice)
        outcome = cut_odd_events(new_end,outcome)
        opto = cut_odd_events(new_end,opto)
        right_lever = cut_odd_events(new_end,right_lever)
        nosepoke = cut_odd_events(new_end,nosepoke)
        reward_pump = cut_odd_events(new_end,reward_pump)
        reward_port = cut_odd_events(new_end,reward_port)
    
    events = {'trial': trial, 'delay': delay, 'choice': choice, 'outcome':
        outcome, 'opto': opto, 'right_lever': right_lever,'nosepoke': nosepoke,
        'reward_pump': reward_pump, 'reward_port': reward_port}
    
    #Assertion QC
    assert np.count_nonzero(trial['times']) % 2 ==0,'ERROR: Uneven trial fronts'
    assert np.count_nonzero(trial['times']) % 2 ==0,'ERROR: Uneven trial fronts'
    

def cut_odd_events(end_time, event):
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
    
    
   
    
    


    # stim off time is the first frame2ttl rise/fall after the trial start
    # does not apply for 1st trial
    ind = np.searchsorted(frame2ttl['times'], t_iti_in, side='left')
    t_stim_off = frame2ttl['times'][ind]
    t_stim_freeze = frame2ttl['times'][np.maximum(ind - 1, 0)]

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

    # stimOn_times: first fram2ttl change after trial start
    trials = Bunch({
        'ready_tone_in': _assign_events_to_trial(t_trial_start, t_ready_tone_in, take='first'),
        'error_tone_in': _assign_events_to_trial(t_trial_start, t_error_tone_in),
        'valve_open': _assign_events_to_trial(t_trial_start, t_valve_open),
        'stim_freeze': _assign_events_to_trial(t_trial_start, t_stim_freeze),
        'stimOn_times': _assign_events_to_trial(t_trial_start, frame2ttl['times'], take='first'),
        'iti_in': _assign_events_to_trial(t_trial_start, t_iti_in)
    })
    # goCue_times corresponds to the tone_in event
    trials['goCue_times'] = trials['ready_tone_in']
    # feedback times are valve open on good trials and error tone in on error trials
    trials['feedback_times'] = trials['valve_open']
    ind_err = np.isnan(trials['valve_open'])
    trials['feedback_times'][ind_err] = trials['error_tone_in'][ind_err]
    trials['intervals'] = np.c_[t_trial_start, trials['iti_in']]
    trials['response_times'] = trials['stimOn_times']

    if save and output_path:
        output_path = Path(output_path)
        np.save(output_path / '_ibl_trials.goCue_times.npy', trials['goCue_times'])
        np.save(output_path / '_ibl_trials.stimOn_times.npy', trials['stimOn_times'])
        np.save(output_path / '_ibl_trials.intervals.npy', trials['intervals'])
        np.save(output_path / '_ibl_trials.feedback_times.npy', trials['feedback_times'])
        np.save(output_path / '_ibl_trials.response_times.npy', trials['response_times'])
    return trials

extract_camera_sync(sync, output_path=session, save=True, chmap=sync_chmap)




