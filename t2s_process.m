clear all
close all
clc

tenum = 9;

subj_folder = 'D:\Workspace_Jianxun\LE_Proj\LE_BOLD\le_bold\subj_TangFeiFei\';

t2s_filename = 't2s_m180_R.nii';


% tearr = [3.93, 7.84, 11.75, 15.66, 19.57, 23.48, 27.39, 31.3, 35.21];

% tearr = [4.03, 8.6, 13.17, 17.74, 22.31, 26.88, 31.45, 36.02, 40.59];

tearr = linspace(3.78, 40.9, 9);

t2s_path = fullfile(subj_folder, t2s_filename);

t2s_nii = load_untouch_nii(t2s_path);

t2s_img = double(t2s_nii.img);

[xres, yres, zres, pnum] = size(t2s_img);

measnum = pnum / tenum;

t2s_img = reshape(t2s_img, xres, yres, zres, tenum, measnum);

t2s_map = zeros(xres, yres, zres, measnum);

t2s_ec1 = zeros(xres, yres, zres, measnum);

sms_kernal = [1 3 1; 3 9 3; 1 3 1]/25;

disp(size(t2s_img))

for measidx = 1: measnum
    
    img_curr = t2s_img(:,:,:, :, measidx);
    
    ec1_curr = img_curr(:, :, :, 1);
    
    img_curr_filt = convn(img_curr, sms_kernal, 'same');
    
    msk_curr = auxil_asl_auto_msk(img_curr_filt(:,:,:,1), 0.35);
    
    tic
    map_curr = auxil_map_t2s_nlin(img_curr_filt, tearr, msk_curr);
    timestamp = toc;
    
    disp([measidx, timestamp])
    
    t2s_map(:, :, :, measidx) = map_curr;
    t2s_ec1(:, :, :, measidx) = ec1_curr;
    
end

imshow(mosaic(rot90(flip(t2s_map, 1))), [0 45]); colormap jet

save(fullfile(subj_folder, 't2s_map_L_1.mat'), 't2s_map');
save(fullfile(subj_folder, 't2s_ec1_L_1.mat'), 't2s_ec1');
