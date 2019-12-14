addpath(genpath('/home/ibladmin/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2')) %Kilosort directory
addpath(genpath('/home/ibladmin/Documents/MATLAB/Kilosort2')) % Iblscripts directory

RootH = '/home/ibladmin/Documents/kilosort'; %working directory for whitening matrix
neuro_folder = '/home/ibladmin/witten/Clare/Ephys/Neuropixel_Ephys_Data/378/';

files = dir(neuro_folder);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);

% Print folder names to command window.
for j=6:length(subFolders); %skip . and .. files
    sub = dir(strcat(neuro_folder,subFolders(j).name));
    sub_dirFlags = [sub.isdir];
    kilo_folders = sub(sub_dirFlags);
    for i = 3:length(kilo_folders);
        recording = dir(strcat(neuro_folder,subFolders(j).name,'/',kilo_folders(i).name));
        rec_dirFlags = [recording.isdir];
        final_folder = recording(rec_dirFlags);
        RootZ = strcat(neuro_folder,subFolders(j).name,'/',kilo_folders(i).name,'/',final_folder(3).name);
        run_ks2_ibl(RootZ,RootH);
    end
end
