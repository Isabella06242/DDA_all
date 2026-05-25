subject_IDs = {'EMU038'};%,'EMU039'};
references = {'bipolar'};
comparisons = {...%'outcome','win','loss';
    'inspection','full_single_opt_info','single_opt_first_inspection'};

%set input and output directories for data and spectrograms
datapath = '/media/Data/Human_Intracranial_MAD/';
data_base_dir = sprintf('%s1_formatted',datapath);
out_base_dir = '/media/Data/DDA/MATLAB_2025/data_for_DDA/';

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

    %loop through subjects
    for subject_num = 1:numel(subject_IDs)
        subject_ID = subject_IDs{subject_num};
        if strcmpi(reference,'bipolar')
            load(sprintf('%s/%s/%s_MAD_SES%d_Setup%s.mat',data_base_dir,subject_ID,subject_ID,1,diff_ref_str),'channel_ind');
            elec_ind = channel_ind; clear channel_ind;
        else
            load(sprintf('%s/%s/%s_MAD_SES%d_Setup%s.mat',data_base_dir,subject_ID,subject_ID,1,diff_ref_str),'elec_ind');
        end

        for chnum = 1:size(elec_ind,1)

            %loop through alignment types
            for align_num = 1:size(comparisons,1)
                alignment = comparisons{align_num,1};
                disp(alignment);
                load(sprintf('%s/%s_%s_clipped%s_ch%03d.mat',out_dir,subject_ID,alignment,diff_ref_str,chnum));
                for comp_num = 1:2
                    subset = comparisons{align_num,comp_num+1};
                    disp(subset);

                    eval(sprintf('x = %s_ch%03d_data(:,%s_ind);',alignment,chnum,subset));

                    %%% TEMPORARY SLOPPY METHOD OF TRIMMING POST_ALIGNMENT DATA ONLY
                    x = x(2049:end,:);

                    %%%%% RUN DDA %%%%%
                    dm = 5;
                    eval(sprintf('%s_AF = nan(50,50,4);',subset));
                    for tau1 = 1:50
                        for tau2 = 1:50
                            TM = max([tau1 tau2]);
                            [AF] = run_DDE_model(x,dm,tau1,tau2,TM);
                            eval(sprintf('%s_AF(tau1,tau2,:) = AF;',subset));
                        end
                    end
                    %%%%%%%%%%%%%%%%%%%
                end
                labels = [1; -1];
                D = nan(50,50,2);
                W = nan(50,50,4);
                B = nan(50,50,1);
                for tau1 = 1:50
                    for tau2 = 1:50
                        eval(sprintf('features = [squeeze(%s_AF(tau1,tau2,:))'';squeeze(%s_AF(tau1,tau2,:))''];',comparisons{align_num,2},comparisons{align_num,3}));
                        %[d,~] = distance_from_hyperplane(features, labels);
                        [d,w,b] = compute_hyperplane_distances(features, labels);
                        D(tau1,tau2,:) = d;
                        W(tau1,tau2,:) = w;
                        B(tau1,tau2,:) = b;
                    end
                end
                D_diff = squeeze(D(:,:,1))-squeeze(D(:,:,2));
                [Y,I] = max(D_diff(:));
                %find best delays
                ll = 0;
                delays = [0 0];
                for tau1 = 1:50
                    for tau2 = 1:50
                        ll = ll+1;
                        if ll == I, delays = [tau1 tau2]; end
                    end
                end
                disp(delays)
                save(sprintf('%s_ch%03d_%s_%s.mat',subject_ID,chnum,comparisons{align_num,2},comparisons{align_num,3}),'delays','W','B','Y')
            end
        end
    end
end