%% set folder name
segment_mat_folder = fullfile('/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/modified_mask_png');
segment_input_mat_folder = fullfile('/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/input_mask');
CAGAN_mask_folder = fullfile('/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/mask');
save_result_folder = fullfile('/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/combined_mask');
save_face_folder = fullfile('/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/face_mask');

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

if ~exist(save_face_folder, 'dir')
    mkdir(save_face_folder);
end

%% 
segment_dir = dir(fullfile(segment_mat_folder, '*.png'));
CAGAN_dir = dir(fullfile(CAGAN_mask_folder, '*.jpg'));

for i = 1 : numel(segment_dir)
    if mod(i, 10) == 0
        fprintf(1, 'processing %d (%d)...\n', i, numel(segment_dir));
    end
    
    segment_mask = imread(fullfile(segment_mat_folder, segment_dir(i).name));
    segment_input_mask = imread(fullfile(segment_input_mat_folder, segment_dir(i).name));
    CAGAN_mask = imread(fullfile(CAGAN_mask_folder, CAGAN_dir(i).name));
    segment_dir(i).name
    CAGAN_dir(i).name
    [row, column] = size(segment_mask);
%     size(CAGAN_mask)
    
%     min(segment_mask(:))
%     min(CAGAN_mask(:))
%     max(segment_mask(:))
%     max(CAGAN_mask(:))
    
    new_mask = zeros(row, column);
    face_mask = zeros(row, column);

    new_mask(segment_mask == 5) = 255;
    new_mask(segment_input_mask == 5) = 255;
    new_mask(CAGAN_mask > 200) = 255;
    new_mask(segment_mask == 14) = 0; %left arm
    new_mask(segment_mask == 15) = 0; %right arm
    new_mask(segment_mask == 1) = 0; %hat
    new_mask(segment_mask == 2) = 0; %hair
    new_mask(segment_mask == 13) = 0; %face
    
    new_mask(segment_input_mask == 1) = 0; %hat
    new_mask(segment_input_mask == 2) = 0; %hair
    new_mask(segment_input_mask == 13) = 0; %face
    
    face_mask(segment_mask == 1) = 255;
    face_mask(segment_mask == 2) = 255;
    face_mask(segment_mask == 13) = 255;
    face_mask(segment_input_mask == 1) = 255;
    face_mask(segment_input_mask == 2) = 255;
    face_mask(segment_input_mask == 13) = 255;
    
    
    %imshow(new_mask);
%     
%     raw_result(raw_result == 7) = 5;
%     raw_result(raw_result == 6) = 5;
%     raw_result(raw_result == 10) = 5;
%     %raw_result(raw_result == 12) = 9;
%     
%     data = raw_result;
    %fullfile(save_result_folder, CAGAN_dir(i).name)
    imwrite(new_mask, fullfile(save_result_folder, CAGAN_dir(i).name));
    imwrite(face_mask, fullfile(save_face_folder, CAGAN_dir(i).name));
%     data2 = load(fullfile(save_result_folder, segment_dir(i).name));
end

      
