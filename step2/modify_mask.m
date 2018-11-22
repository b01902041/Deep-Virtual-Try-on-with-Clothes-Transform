clear;
close all;
fclose all;
%%
output_mat_folder = fullfile('/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/fc8_mask');
save_result_folder = fullfile('/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/modified_mask');
if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end
output_dir = dir(fullfile(output_mat_folder, '*.mat'));
for i = 1 : numel(output_dir)
    if mod(i, 10) == 0
        fprintf(1, 'processing %d (%d)...\n', i, numel(output_dir));
    end
    data1 = load(fullfile(output_mat_folder, output_dir(i).name));
    raw_result = data1.data;
    %raw_result = permute(raw_result, [2 1 3]);
    %imshow(raw_result)
    raw_result(raw_result == 7) = 5;
    raw_result(raw_result == 6) = 5;
    raw_result(raw_result == 10) = 5;
    raw_result(raw_result == 11) = 5;
    %raw_result(raw_result == 12) = 9;
    
    data = raw_result;
    save(fullfile(save_result_folder, output_dir(i).name), 'data');
    data2 = load(fullfile(save_result_folder, output_dir(i).name));
end

      
