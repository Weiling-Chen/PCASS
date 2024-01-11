% Prepare three *.mat files before testing your own image
clc;
clear all;

% Replace the path with your image path
Test_img = imread('testimg/22_8_vdsr.png'); 

% Row position of test image's high-level feature in *.mat file
% 11_4_RCAN.png, k=1; 22_8_vdsr.png, k=2；
k=2;

% Quality prediction (w & w/o nonlinear fit)
[score, score_fit] = CTE_score(Test_img,k);
disp(['Quality Score: ',num2str(score)]);
disp(['Quality Score (nonlinear_fit): ',num2str(score_fit)]);