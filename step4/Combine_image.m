CAGAN_input_folder = '/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/input_image';
CAGAN_output_folder = '/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/output_image';
%warping_shirt_folder = './testing_results_perceptualloss_256_conv5_mse_0.9999_0.0001/test_result';
warping_shirt_folder = '/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/warping_results/test_result';
alpha_mask_folder = '/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/warping_results/test_mask';
face_mask_folder = '/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/face_mask';
%save_folder = './testing_results_perceptualloss_256_conv5_mse_0.9999_0.0001/combine';
save_folder = '/Users/Claire/Documents/CSIEMaster/Research/CAGAN/testing_results_with_conditioning_data_5/combine_final';

CAGAN_input = dir(fullfile(CAGAN_input_folder, '*.jpg'));
CAGAN_output = dir(fullfile(CAGAN_output_folder, '*.jpg'));
warping_shirt = dir(fullfile(warping_shirt_folder, '*.jpg'));
alpha_mask = dir(fullfile(alpha_mask_folder, '*.jpg'));
face_mask = dir(fullfile(face_mask_folder, '*.jpg'));

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

for i = 1 : numel(CAGAN_output) 
    file_name = CAGAN_output(i).name;
    CAGAN_input_img = imread(fullfile(CAGAN_input_folder, CAGAN_input(i).name));
    CAGAN_output_img = imread(fullfile(CAGAN_output_folder, CAGAN_output(i).name));
    warping_shirt_img = imread(fullfile(warping_shirt_folder, warping_shirt(i).name));
    alpha_mask_img = imread(fullfile(alpha_mask_folder, alpha_mask(i).name));
    face_mask_img = imread(fullfile(face_mask_folder, face_mask(i).name));
    
    alpha_mask_img = alpha_mask_img / 255;
    face_mask_img = face_mask_img / 255;
    alpha_mask_array = cat(3, alpha_mask_img, alpha_mask_img, alpha_mask_img);
    
    %alpha_mask_array = uint8(ones(256, 256, 3));
    %alpha_mask_array(warping_shirt_img >= 230) = 0;
    
    face_mask_array = cat(3, face_mask_img, face_mask_img, face_mask_img);

    output = alpha_mask_array .* warping_shirt_img + (1 - alpha_mask_array - face_mask_array) .* CAGAN_output_img + face_mask_array .* CAGAN_input_img;
    imshow(output);
    imwrite(output, fullfile(save_folder, file_name));
end