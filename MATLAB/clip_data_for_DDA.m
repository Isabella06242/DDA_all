subject_IDs = {'EMU038'};%,'EMU039'};
references = {'bipolar'};
%alignments = {'outcome','win';...
%    'outcome','loss'};
alignments = {'inspection','full_single_opt_info';...
    'inspection','single_opt_first_inspection'};

% add needed function to path
addpath '/media/Data/Human_Intracranial_MAD/_toolbox/spectral-analysis'

%set input and output directories for data and spectrograms
datapath = '/media/Data/Human_Intracranial_MAD/';
data_base_dir = sprintf('%s1_formatted',datapath);
out_base_dir = '/media/Data/DDA/MATLAB_2025/data_for_DDA/';

% Set analysis window in seconds
time_window_seconds = 2;

% set number of points for mean subtraction
mean_subtract_window = 10;

% parameters for interictal spike removal
spike_amp_thresh = 5.8;       % amplitude (z-scored) threshold
spike_pad_seconds = 1;      % time (in seconds) to remove around spikes (both sides)

%loop through referencing schemes
for ref_num = 1:numel(references)
    reference = references{ref_num};
    if strcmpi('ground',reference)
        diff_ref_str = '';
    else
        diff_ref_str = ['_' reference];
    end
    out_dir = [out_base_dir '/' reference];
    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    %loop through alignment types
    for align_num = 1:size(alignments,1)
        alignment = alignments{align_num,1};
        disp(alignment);
        subset = alignments{align_num,2};
        disp(subset);

        %loop through subjects
        for subject_num = 1:numel(subject_IDs)
            subject_ID = subject_IDs{subject_num};
            if strcmpi('ground',reference)
                data_dir = [data_base_dir '/' subject_ID '/separate_channel_files'];
                if ~exist(data_dir,'dir'), mkdir(data_dir); end
            else
                data_dir = [data_base_dir '/' subject_ID '/separate_channel_files/' reference];
                if ~exist(data_dir,'dir'), mkdir(data_dir); end
            end
            switch subject_ID
                case 'EMU001'
                    session_numbers = [1 2 3];
                case 'EMU024'
                    session_numbers = [1 2 3];
                case 'EMU025'
                    session_numbers = [1 2];%3];
                case 'EMU030'
                    session_numbers = [1 2];
                case 'EMU036'
                    session_numbers = [1 3 4];%2 5];
                case 'EMU037'
                    session_numbers = [1 2 3 4];
                case 'EMU038'
                    session_numbers = 1;
                case 'EMU039'
                    session_numbers = [1 2 3 4];
                case 'EMU041'
                    session_numbers = [1 2 3 4 5 6 7 8 9];
                case 'EMU047'
                    session_numbers = 1;
                case 'EMU051'
                    session_numbers = 1;
            end

            %loop through sessions
            for sesnum = session_numbers
                load(sprintf('%s/%s/%s_MAD_SES%d_Setup%s.mat',data_base_dir,subject_ID,subject_ID,sesnum,diff_ref_str));
                load(sprintf('%s/%s/%s_MAD_SES%d_Raw.mat',data_base_dir,subject_ID,subject_ID,sesnum),'Fs','infos');

                if strcmpi(reference,'bipolar')
                    elec_ind = channel_ind(:,1);
                end

                % set sampling rate-dependent spectral parameters
                time_window_length = time_window_seconds*Fs;
                win = [time_window_length/2 time_window_length/2]; %time before and after alignment point in samples
                nwinl = round(win(1));
                nwinr = round(win(2));
    
                % set pad for data removal around interictal spikes based on sampling rate
                spike_pad = spike_pad_seconds*Fs;

                % get times for specified alignment points
                [align_times] = get_align_times(filters, trial_times, trial_words, alignment);
                [subset_times] = get_align_times(filters, trial_times, trial_words, subset);

                % remove NaN values--should come up with better solution
                align_times(isnan(align_times)) = [];
                subset_times(isnan(subset_times)) = [];

                % remove negative times--happens with multiple sessions on the same login--should remove this from trial_times
                align_times(align_times<0) = [];
                subset_times(subset_times<0) = [];

                % convert alignment times from seconds to samples
                align_times = round(align_times*Fs);
                subset_times = round(subset_times*Fs);

                % remove long flat periods in first two sessions from EMU001
                if strcmpi(subject_ID,'EMU001')&&sesnum<3
                    if sesnum == 1
                        begin_flat_1 = 786401;
                        end_flat_1 = 1171845;
                        begin_flat_2 = 1801256;
                        end_flat_2 = 2283692;
                    end
                    if sesnum == 2
                        begin_flat_1 = 1268592;
                        end_flat_1 = 1635461;
                        begin_flat_2 = 2948583;
                        end_flat_2 = 3414454;
                    end
                    align_times(align_times>begin_flat_1-win(1)&align_times<end_flat_1+win(2)) = [];
                    subset_times(subset_times>begin_flat_1-win(1)&subset_times<end_flat_1+win(2)) = [];
                    align_times(align_times>begin_flat_2-win(1)&align_times<end_flat_2+win(2)) = [];
                    subset_times(subset_times>begin_flat_2-win(1)&subset_times<end_flat_2+win(2)) = [];
                end

                % INDICES FOR SPECIFIC SUBSET OF ALIGNMENTS
                eval(sprintf('%s_SES%d_ind = find(ismember(align_times,subset_times));',subset,sesnum));

                 % store indices across sessions
                 if sesnum==min(session_numbers)
                     eval(sprintf('%s_ind = %s_SES%d_ind;',subset,subset,sesnum));
                     add_ind = numel(align_times);
                 else
                     eval(sprintf('%s_ind = [%s_ind; %s_SES%d_ind+add_ind];',subset,subset,subset,sesnum));
                     add_ind = add_ind+numel(align_times);
                 end

                % initialize list to store channels with files already saved
                skip_chan_list = [];
                for chnum = 1:numel(elec_ind)
                    if ~isfile(sprintf('%s/%s_%s_clipped%s_ch%03d.mat',out_dir,subject_ID,alignment,diff_ref_str,chnum))
                        if strcmpi(reference,'bipolar')
                            load(sprintf('%s/%s_MAD_SES%d_ch%03d-ch%03d.mat',data_dir,subject_ID,sesnum,channel_ind(chnum,1),channel_ind(chnum,2)));
                        else
                            load(sprintf('%s/%s_MAD_SES%d_ch%03d.mat',data_dir,subject_ID,sesnum,chnum));
                        end

                        % SUBTRACT MOVING AVERAGE
                        win_len = round(mean_subtract_window*Fs);
                        ma = movmean(data, win_len);
                        mean_subtracted = data - ma;

                        % FILTER 60 HZ NOISE
                        d = designfilt('bandstopiir','FilterOrder',2, ...
                            'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
                            'DesignMethod','butter','SampleRate',Fs);
                        d1 = designfilt('bandstopiir','FilterOrder',2, ...
                            'HalfPowerFrequency1',119,'HalfPowerFrequency2',121, ...
                            'DesignMethod','butter','SampleRate',Fs);
                        d2 = designfilt('bandstopiir','FilterOrder',2, ...
                            'HalfPowerFrequency1',179,'HalfPowerFrequency2',181, ...
                            'DesignMethod','butter','SampleRate',Fs);
                        filtered = filtfilt(d, mean_subtracted);
                        filtered = filtfilt(d1, filtered);
                        filtered = filtfilt(d2, filtered);

                        % Z-SCORE
                        zscored = filtered./std(filtered,0,1);

                        % clip short segments of data aligned to behavioral timepoints
                        data = nan(time_window_length,numel(align_times));
                        for event = 1:numel(align_times)
                            if align_times(event)+win(2)-1>size(zscored,1)
                                continue
                            elseif align_times(event)-win(1)<0
                                continue
                            else
                                data(:,event) = zscored(align_times(event)-win(1):align_times(event)+win(2)-1);
                            end
                        end
                        
                        % store data across sessions
                        if sesnum==min(session_numbers)
                            eval(sprintf('%s_ch%03d_data = data;',alignment,chnum));
                        else
                            eval(sprintf('%s_ch%03d_data = [%s_ch%03d_data, data];',alignment,chnum,alignment,chnum));
                        end

                        % clear data to save memory
                        clear data;
                    else
                        skip_chan_list = [skip_chan_list chnum];
                    end
                end
            end
            for chnum = 1:numel(elec_ind)
                if any(skip_chan_list==chnum)
                    save(sprintf('%s/%s_%s_clipped%s_ch%03d.mat',out_dir,subject_ID,alignment,diff_ref_str,chnum),sprintf('%s_ind',subset),'-append');
                else
                    % save clipped data 
                    save(sprintf('%s/%s_%s_clipped%s_ch%03d.mat',out_dir,subject_ID,alignment,diff_ref_str,chnum),sprintf('%s_ch%03d_data',alignment,chnum),sprintf('%s_ind',subset),'-v7.3');
               
                    %clear channel spectrograms to save memory
                    eval(sprintf('clear %s_ch%03d_data',alignment,chnum))
                end
            end
        end
    end
end