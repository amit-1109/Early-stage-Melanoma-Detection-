clc;
close all;
clear all;
warning off;

ada = dir('test cancer\*.jpg');

fea_skin11=[];

for ik = 1:length(ada)
    file = ada(ik).name;
    I=imread(fullfile('test cancer\',file));
    I = imresize(I,[256,256]);
% % figure, imshow(I); title(' Leaf Image');
% % Enhance Contrast
% I = imadjust(I,stretchlim(I));
% % figure, imshow(I);title('Contrast Enhanced');
% I_Otsu = im2bw(I,graythresh(I));
% % figure,imshow(I_Otsu);title('otsu');
% 
% HSV=rgb2hsv(I);
% % figure,imshow(HSV);
% 
% H=HSV(:,:,1);
% S=HSV(:,:,2);
% V=HSV(:,:,3);
% 
% % figure,imshow(H);
% % figure,imshow(S);
% % figure,imshow(V);
% 
% %% RGB to Gray conversion
% [m n o]=size(V);
% if o==3
%     gray=rgb2gray(V);
% else
%     gray=V;
% end
% % figure,imshow(gray);title('YELLOW CHANNEL -GRAY IMAGE');
% 
% 
% %%%%%%ADJUST THE CONTRAST OF THE GRAY CHANNEL IMAGE%%%%
% ad=imadjust(gray);
% % figure,imshow(ad);title('ADJUSTED GRAY IMAGE');
% 
% %%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
% bw=im2bw(gray,0.5);
% % figure,imshow(bw);title('BLACK AND WHITE IMAGE');
% 
% %%%%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
% bw=imcomplement(bw);
% % figure,imshow(bw);title('COMPLEMENT IMAGE');
% 
% bw=imfill(bw,'holes');
% % figure,imshow(bw),title('HOLES IMAGE');
% SE=strel('square',3);
% bw=imdilate(bw,SE);
% % figure,imshow(bw),title('DILATE');
% 
% bw=imfill(bw,'holes');
% 
% SE=strel('disk',10);
% bw=imopen(bw,SE);
% % figure,imshow(bw),title('open');
% bw=imclearborder(bw);
% % figure,imshow(bw),title('bwww');
% 
% % figure,imshow(S);
% % figure,imshow(V);
% 
% 
% % % % 
% % Extract Features
% cform = makecform('srgb2lab');
% % % % Apply the colorform
% lab_he = applycform(I ,cform);
% % figure,imshow(lab_he);
% % % Classify the colors in a*b* colorspace using K means clustering.
% % % Since the image has 3 colors create 3 clusters.
% % % Measure the distance using Euclidean Distance Metric.
% ab = double(lab_he(:,:,2:3));
% nrows = size(ab,1);
% ncols = size(ab,2);
% ab = reshape(ab,nrows*ncols,2);
% % % % %%%%%%%%%%    SEGMENTATION
% nColors = 3;
% [cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
%                                       'Replicates',3);
% % [cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% % % % Label every pixel in tha image using results from K means
% pixel_labels = reshape(cluster_idx,nrows,ncols);
% % figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');
% % % 
% % % % Create a blank cell array to store the results of clustering
% segmented_images = cell(1,3);
% % % % Create RGB label using pixel_labels
% rgb_label = repmat(pixel_labels,[1,1,3]);
% % % 
% for k = 1:nColors
%     colors = I;
%     colors(rgb_label ~= k) = 0;
%     segmented_images{k} = colors;
% end
% % figure, subplot(1,3,1);
% % imshow(segmented_images{1});title('Cluster 1'); subplot(1,3,2);imshow(segmented_images{2});title('Cluster 2');
% % subplot(1,3,3);imshow(segmented_images{3});title('Cluster 3');
% % set(gcf, 'Position', get(0,'Screensize'));
% % % % % Feature Extraction
% x ='2';
% i = str2double(x);
% % % Extract the features from the segmented image
% seg_img = segmented_images{i};
% % % Convert to grayscale if image is RGB
% if ndims(seg_img) == 3
%    img = rgb2gray(seg_img);
% end
% figure, imshow(seg_img); title('Gray Scale Image');
% % % % % Evaluate the disease affected area
% % black = im2bw(seg_img,graythresh(seg_img));
% % % figure
% % % imshow(black);
% % % figure, imshow(black);title('Black & White Image');
% % 
% % black=imfill(black,'holes');
% % % figure,imshow(black),title('bw image');
% % 
% % SE=strel('square',10);
% % 
% % black=imdilate(black,SE);
% % 
% % % figure,imshow(black),title('bw');
% % 
% % G=bw;
% 
% 
%  %%%FEATURE EXTRACTION BY GLCM(GRAY LEVEL CO-OCCURENCE MATRIX)%%%%
% g = graycomatrix(bw);
% stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
% Contrast = stats.Contrast;
% Correlation = stats.Correlation;
% Energy = stats.Energy;
% Homogeneity = stats.Homogeneity;
% Mean = mean2(bw);
% Standard_Deviation = std2(bw);
% Entropy = entropy(bw);
% % RMS = mean2(rms(G));
% % Skewness = skewness(G)
% Variance = mean2(var(double(bw)));
% a = sum(double(bw(:)));
% Smoothness = 1-(1/(1+a));
% Kurtosis = kurtosis(double(bw(:)));
% % Skewness = skewness(double(G(:)));
% 
% %%% Inverse Difference Movement%%%
% m = size(bw,1);
% n = size(bw,2);
% in_diff = 0;
% for i = 1:m
%     for j = 1:n
%         temp = bw(i,j)./(1+(i-j).^2);
%         in_diff = in_diff+temp;
%     end
% end
% IDM = double(in_diff);
% % S=regionprops(G,'ALL');
% % area=[S.Area];
% feat_disease1 = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy,Variance, Smoothness, Kurtosis,IDM];
% 
% 
% 
% 
% %% % Color Texture Feature Extraction(LBP) FOR TUMOR ALONE IMAGE%%%
% % % % step 1: Local Binary Patterns 
% feat_LBP = extractLBPFeatures(bw);
% grayImage = bw;
% localBinaryPatternImage1 = zeros(size(grayImage));
% [row col] = size(grayImage);
% for r = 2 : row - 1   
% 	for c = 2 : col - 1    
% 		centerPixel = grayImage(r, c);
% 		pixel7 = grayImage(r-1, c-1) > centerPixel;  
% 		pixel6 = grayImage(r-1, c) > centerPixel;   
% 		pixel5 = grayImage(r-1, c+1) > centerPixel;  
% 		pixel4 = grayImage(r, c+1) > centerPixel;     
% 		pixel3 = grayImage(r+1, c+1) > centerPixel;    
% 		pixel2 = grayImage(r+1, c) > centerPixel;      
% 		pixel1 = grayImage(r+1, c-1) > centerPixel;     
% 		pixel0 = grayImage(r, c-1) > centerPixel;       
% 		localBinaryPatternImage1(r, c) = uint8(...
% 			pixel7 * 2^7 + pixel6 * 2^6 + ...
% 			pixel5 * 2^5 + pixel4 * 2^4 + ...
% 			pixel3 * 2^3 + pixel2 * 2^2 + ...
% 			pixel1 * 2 + pixel0);
% 	end  
% end 
% 
% % figure,imshow(localBinaryPatternImage1),title('LBP TUMOR-ALONE IMAGE'); 
% 
% 
%  %%%FEATURE EXTRACTION BY GLCM(GRAY LEVEL CO-OCCURENCE MATRIX)%%%%
%  feat_LBP=im2double(feat_LBP);
% g1 = graycomatrix(feat_LBP);
% stats1 = graycoprops(g,'Contrast Correlation Energy Homogeneity');
% Contrast1 = stats1.Contrast;
% Correlation1 = stats1.Correlation;
% Energy1 = stats1.Energy;
% Homogeneity1 = stats1.Homogeneity;
% Mean1 = mean2(feat_LBP);
% Standard_Deviation1 = std2(feat_LBP);
% Entropy1 = entropy(feat_LBP);
% % RMS = mean2(rms(G));
% % Skewness = skewness(G)
% Variance1 = mean2(var(double(feat_LBP)));
% a1 = sum(double(feat_LBP(:)));
% Smoothness1 = 1-(1/(1+a));
% Kurtosis1 = kurtosis(double(feat_LBP(:)));
% % Skewness = skewness(double(G(:)));
% 
% %%% Inverse Difference Movement%%%
% m1 = size(feat_LBP,1);
% n1 = size(feat_LBP,2);
% in_diff = 0;
% for i = 1:m1
%     for j = 1:n1
%         temp = feat_LBP(i,j)./(1+(i-j).^2);
%         in_diff1 = in_diff+temp;
%     end
% end
% IDM1 = double(in_diff);
% feat_disease2 = [Contrast1,Correlation1,Energy1,Homogeneity1, Mean1, Standard_Deviation1, Entropy1,Variance1, Smoothness1, Kurtosis1,IDM1];
% 
% 
% 
% %%%%TO TAKE COLOUR FEATURES%%%%%%
% R = mean2(seg_img(:,:,1));
% G = mean2(seg_img(:,:,2));
% B = mean2(seg_img(:,:,3));
% Co_Fea = [R G B];
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);


r=im2double(R);
g=im2double(G);
b=im2double(B);
%%%%TO ENHANCE THE CONTRAST OF THE RESIZED IMAGE%%%%
I = imadjust(I,stretchlim(I));
% figure, imshow(I);title('CONTRAST ENHANCED IMAGE');

          %%%%%%%SEGMENTATION%%%%%%

%%%%PERFORM RGB TO HSV COLOUR TRANSFORMATION%%%%% 
HSV=rgb2hsv(I);
% figure,imshow(HSV),title('HSV COLOUR TRANSFORM IMAGE');
%%SEPARATE THREE CHANNELS%%%
H=HSV(:,:,1);
S=HSV(:,:,2);
V=HSV(:,:,3);

% figure,imshow(H),title('H-CHANNEL IMAGE');
% figure,imshow(S),title('S-CHANNEL IMAGE');
% figure,imshow(V),title('V-CHANNEL IMAGE');

% figure,
% subplot(1,3,1),imshow(H),title('H-CHANNEL');
% subplot(1,3,2),imshow(S),title('S-CHANNEL');
% subplot(1,3,3),imshow(V),title('V-CHANNEL');

%% PERFORM RGB TO GRAY CONVERSION ON THE V-CHANNEL IMAGE%%%%
[m n o]=size(V);
if o==3
    gray=rgb2gray(V);
else
    gray=V;
end
% figure,imshow(gray);title('V- CHANNEL GRAY IMAGE');

% % % % % % % % % % ad = fcnBPDFHE(gray); 


%%%%%%ADJUST THE CONTRAST OF THE GRAY CHANNEL IMAGE%%%%
ad=imadjust(gray);
% figure,imshow(ad);title('ADJUSTED GRAY IMAGE');

%%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
bw=im2bw(gray,0.5);
% figure,imshow(bw);title('BLACK AND WHITE IMAGE');

%%%%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
bw=imcomplement(bw);
% figure,imshow(bw);title('COMPLEMENT IMAGE');

%%%%TO PERFORM MORPHOLOGICAL OPERATIONS IN THE BW IMAGE%%%%
 %%FILL HOLES%%
bw=imfill(bw,'holes');
% figure,imshow(bw),title('HOLES IMAGE');
 %%DILATE OPERATION%%%
SE=strel('square',3);
bw=imdilate(bw,SE);
% figure,imshow(bw),title('DILATED IMAGE');

 %%AGAIN FILL HOLES%%
bw=imfill(bw,'holes');
  %%TO REMOVE THE SMALL OBJECTS TO PERFORM IMOPEN OPERATIONS%%%
SE=strel('disk',10);
bw=imopen(bw,SE);
% figure,imshow(bw),title('SMALL OBJECTS REMOVED IMAGE');
  %%TO CLEAR THE UNWANTED BORDERS%%%
bw=imclearborder(bw);
% figure,imshow(bw),title('BORDER CORRECTED IMAGE');
bw11=im2double(bw);


%% Masking
seg = bw11.*r;
seg2 = bw11.*g;
seg3 = bw11.*b;
seg1=cat(3,seg,seg2,seg3);
% figure,imshow(seg1);
% title('Segmentation output');
%  
R1 = seg1(:,:,1);
G1 = seg1(:,:,2);
B1 = seg1(:,:,3);





             %%%%%%%FEATURE EXTRACTION%%%%%%%%
             
%      %%%%FEATURE EXTRACTION BY GLCM(GREY LEVEL CO-OCCURENCE MATRIX)%%%%%
% 
g = graycomatrix(bw);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(bw);
Standard_Deviation = std2(bw);
Entropy = entropy(bw);
Variance = mean2(var(double(bw)));
a = sum(double(bw(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(bw(:)));


%%% Inverse Difference Movement%%%
m = size(bw,1);
n = size(bw,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = bw(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);


feat_disease1 = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy,Variance, Smoothness, Kurtosis,IDM];




%%%%%FEATURE EXTRACTION BY LBP(LOCAL BINARY PATTERN)%%%%
% % % step 1: Local Binary Patterns 
feat_LBP = extractLBPFeatures(bw);
grayImage = bw;
localBinaryPatternImage1 = zeros(size(grayImage));
[row col] = size(grayImage);
for r = 2 : row - 1   
	for c = 2 : col - 1    
		centerPixel = grayImage(r, c);
		pixel7 = grayImage(r-1, c-1) > centerPixel;  
		pixel6 = grayImage(r-1, c) > centerPixel;   
		pixel5 = grayImage(r-1, c+1) > centerPixel;  
		pixel4 = grayImage(r, c+1) > centerPixel;     
		pixel3 = grayImage(r+1, c+1) > centerPixel;    
		pixel2 = grayImage(r+1, c) > centerPixel;      
		pixel1 = grayImage(r+1, c-1) > centerPixel;     
		pixel0 = grayImage(r, c-1) > centerPixel;       
		localBinaryPatternImage1(r, c) = uint8(...
			pixel7 * 2^7 + pixel6 * 2^6 + ...
			pixel5 * 2^5 + pixel4 * 2^4 + ...
			pixel3 * 2^3 + pixel2 * 2^2 + ...
			pixel1 * 2 + pixel0);
	end  
end 

% figure,imshow(localBinaryPatternImage1),title('LBP-IMAGE'); 


 %%%TAKE IMPORTANT FEATURES FROM THE LBP BY USING GLCM%%%%
 feat_LBP=im2double(feat_LBP);
g1 = graycomatrix(feat_LBP);
stats1 = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast1 = stats1.Contrast;
Correlation1 = stats1.Correlation;
Energy1 = stats1.Energy;
Homogeneity1 = stats1.Homogeneity;
Mean1 = mean2(feat_LBP);
Standard_Deviation1 = std2(feat_LBP);
Entropy1 = entropy(feat_LBP);
Variance1 = mean2(var(double(feat_LBP)));
a1 = sum(double(feat_LBP(:)));
Smoothness1 = 1-(1/(1+a));
Kurtosis1 = kurtosis(double(feat_LBP(:)));
%%% Inverse Difference Movement%%%
m1 = size(feat_LBP,1);
n1 = size(feat_LBP,2);
in_diff = 0;
for i = 1:m1
    for j = 1:n1
        temp = feat_LBP(i,j)./(1+(i-j).^2);
        in_diff1 = in_diff+temp;
    end
end
IDM1 = double(in_diff);

feat_disease2 = [Contrast1,Correlation1,Energy1,Homogeneity1, Mean1, Standard_Deviation1, Entropy1,Variance1, Smoothness1, Kurtosis1,IDM1];


%%%% TAKE COLOUR FEATURES FROM THE K-MEANS OUTPUT IMAGE %%%%%%
R2 = mean2(seg1(:,:,1));
G2 = mean2(seg1(:,:,2));
B2 = mean2(seg1(:,:,3));
Co_Fea = [R2 G2 B2];

     
%%%%%COMBINE ALL THE FEATURES%%%%%%
% feat_tot=[Co_Fea feat_disease1 feat_disease2]
feat_tot=[Co_Fea feat_disease1 feat_disease2]
fea_skin11=[fea_skin11;feat_tot];

end

% ada = dir('NEVUS\*.jpg');
% for ik = 1:length(ada)
%     file = ada(ik).name;
%     I=imread(fullfile('NEVUS\',file));
%     I = imresize(I,[256,256]);
% % figure, imshow(I); title(' Leaf Image');
% % Enhance Contrast
% % I = imadjust(I,stretchlim(I));
% % % figure, imshow(I);title('Contrast Enhanced');
% % I_Otsu = im2bw(I,graythresh(I));
% % % figure,imshow(I_Otsu);title('otsu');
% % 
% % HSV=rgb2hsv(I);
% % % figure,imshow(HSV);
% % 
% % H=HSV(:,:,1);
% % S=HSV(:,:,2);
% % V=HSV(:,:,3);
% % 
% % % figure,imshow(H);
% % % figure,imshow(S);
% % % figure,imshow(V);
% % 
% % %% RGB to Gray conversion
% % [m n o]=size(V);
% % if o==3
% %     gray=rgb2gray(V);
% % else
% %     gray=V;
% % end
% % % figure,imshow(gray);title('YELLOW CHANNEL -GRAY IMAGE');
% % 
% % 
% % %%%%%%ADJUST THE CONTRAST OF THE GRAY CHANNEL IMAGE%%%%
% % ad=imadjust(gray);
% % % figure,imshow(ad);title('ADJUSTED GRAY IMAGE');
% % 
% % %%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
% % bw=im2bw(gray,0.5);
% % % figure,imshow(bw);title('BLACK AND WHITE IMAGE');
% % 
% % %%%%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
% % bw=imcomplement(bw);
% % % figure,imshow(bw);title('COMPLEMENT IMAGE');
% % 
% % bw=imfill(bw,'holes');
% % % figure,imshow(bw),title('HOLES IMAGE');
% % SE=strel('square',3);
% % bw=imdilate(bw,SE);
% % % figure,imshow(bw),title('DILATE');
% % 
% % bw=imfill(bw,'holes');
% % 
% % SE=strel('disk',10);
% % bw=imopen(bw,SE);
% % % figure,imshow(bw),title('open');
% % bw=imclearborder(bw);
% % % figure,imshow(bw),title('bwww');
% % 
% % % figure,imshow(S);
% % % figure,imshow(V);
% % 
% % 
% % % % % 
% % % Extract Features
% % cform = makecform('srgb2lab');
% % % % % Apply the colorform
% % lab_he = applycform(I ,cform);
% % % figure,imshow(lab_he);
% % % % Classify the colors in a*b* colorspace using K means clustering.
% % % % Since the image has 3 colors create 3 clusters.
% % % % Measure the distance using Euclidean Distance Metric.
% % ab = double(lab_he(:,:,2:3));
% % nrows = size(ab,1);
% % ncols = size(ab,2);
% % ab = reshape(ab,nrows*ncols,2);
% % % % % %%%%%%%%%%    SEGMENTATION
% % nColors = 3;
% % [cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
% %                                       'Replicates',3);
% % % [cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% % % % % Label every pixel in tha image using results from K means
% % pixel_labels = reshape(cluster_idx,nrows,ncols);
% % % figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');
% % % % 
% % % % % Create a blank cell array to store the results of clustering
% % segmented_images = cell(1,3);
% % % % % Create RGB label using pixel_labels
% % rgb_label = repmat(pixel_labels,[1,1,3]);
% % % % 
% % for k = 1:nColors
% %     colors = I;
% %     colors(rgb_label ~= k) = 0;
% %     segmented_images{k} = colors;
% % end
% % % figure, subplot(1,3,1);
% % % imshow(segmented_images{1});title('Cluster 1'); subplot(1,3,2);imshow(segmented_images{2});title('Cluster 2');
% % % subplot(1,3,3);imshow(segmented_images{3});title('Cluster 3');
% % % set(gcf, 'Position', get(0,'Screensize'));
% % % % % % Feature Extraction
% % x ='2';
% % i = str2double(x);
% % % % Extract the features from the segmented image
% % seg_img = segmented_images{i};
% % % % Convert to grayscale if image is RGB
% % if ndims(seg_img) == 3
% %    img = rgb2gray(seg_img);
% % end
% % figure, imshow(seg_img); title('Gray Scale Image');
% % % % % % Evaluate the disease affected area
% % % black = im2bw(seg_img,graythresh(seg_img));
% % % % figure
% % % % imshow(black);
% % % % figure, imshow(black);title('Black & White Image');
% % % 
% % % black=imfill(black,'holes');
% % % % figure,imshow(black),title('bw image');
% % % 
% % % SE=strel('square',10);
% % % 
% % % black=imdilate(black,SE);
% % % 
% % % % figure,imshow(black),title('bw');
% % % 
% % % G=bw;
% % 
% % 
% %  %%%FEATURE EXTRACTION BY GLCM(GRAY LEVEL CO-OCCURENCE MATRIX)%%%%
% % g = graycomatrix(bw);
% % stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
% % Contrast = stats.Contrast;
% % Correlation = stats.Correlation;
% % Energy = stats.Energy;
% % Homogeneity = stats.Homogeneity;
% % Mean = mean2(bw);
% % Standard_Deviation = std2(bw);
% % Entropy = entropy(bw);
% % % RMS = mean2(rms(G));
% % % Skewness = skewness(G)
% % Variance = mean2(var(double(bw)));
% % a = sum(double(bw(:)));
% % Smoothness = 1-(1/(1+a));
% % Kurtosis = kurtosis(double(bw(:)));
% % % Skewness = skewness(double(G(:)));
% % 
% % %%% Inverse Difference Movement%%%
% % m = size(bw,1);
% % n = size(bw,2);
% % in_diff = 0;
% % for i = 1:m
% %     for j = 1:n
% %         temp = bw(i,j)./(1+(i-j).^2);
% %         in_diff = in_diff+temp;
% %     end
% % end
% % IDM = double(in_diff);
% % % S=regionprops(G,'ALL');
% % % area=[S.Area];
% % feat_disease1 = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy,Variance, Smoothness, Kurtosis,IDM];
% % 
% % 
% % 
% % 
% % %% % Color Texture Feature Extraction(LBP) FOR TUMOR ALONE IMAGE%%%
% % % % % step 1: Local Binary Patterns 
% % feat_LBP = extractLBPFeatures(bw);
% % grayImage = bw;
% % localBinaryPatternImage1 = zeros(size(grayImage));
% % [row col] = size(grayImage);
% % for r = 2 : row - 1   
% % 	for c = 2 : col - 1    
% % 		centerPixel = grayImage(r, c);
% % 		pixel7 = grayImage(r-1, c-1) > centerPixel;  
% % 		pixel6 = grayImage(r-1, c) > centerPixel;   
% % 		pixel5 = grayImage(r-1, c+1) > centerPixel;  
% % 		pixel4 = grayImage(r, c+1) > centerPixel;     
% % 		pixel3 = grayImage(r+1, c+1) > centerPixel;    
% % 		pixel2 = grayImage(r+1, c) > centerPixel;      
% % 		pixel1 = grayImage(r+1, c-1) > centerPixel;     
% % 		pixel0 = grayImage(r, c-1) > centerPixel;       
% % 		localBinaryPatternImage1(r, c) = uint8(...
% % 			pixel7 * 2^7 + pixel6 * 2^6 + ...
% % 			pixel5 * 2^5 + pixel4 * 2^4 + ...
% % 			pixel3 * 2^3 + pixel2 * 2^2 + ...
% % 			pixel1 * 2 + pixel0);
% % 	end  
% % end 
% % 
% % % figure,imshow(localBinaryPatternImage1),title('LBP TUMOR-ALONE IMAGE'); 
% % 
% % 
% %  %%%FEATURE EXTRACTION BY GLCM(GRAY LEVEL CO-OCCURENCE MATRIX)%%%%
% %  feat_LBP=im2double(feat_LBP);
% % g1 = graycomatrix(feat_LBP);
% % stats1 = graycoprops(g,'Contrast Correlation Energy Homogeneity');
% % Contrast1 = stats1.Contrast;
% % Correlation1 = stats1.Correlation;
% % Energy1 = stats1.Energy;
% % Homogeneity1 = stats1.Homogeneity;
% % Mean1 = mean2(feat_LBP);
% % Standard_Deviation1 = std2(feat_LBP);
% % Entropy1 = entropy(feat_LBP);
% % % RMS = mean2(rms(G));
% % % Skewness = skewness(G)
% % Variance1 = mean2(var(double(feat_LBP)));
% % a1 = sum(double(feat_LBP(:)));
% % Smoothness1 = 1-(1/(1+a));
% % Kurtosis1 = kurtosis(double(feat_LBP(:)));
% % % Skewness = skewness(double(G(:)));
% % 
% % %%% Inverse Difference Movement%%%
% % m1 = size(feat_LBP,1);
% % n1 = size(feat_LBP,2);
% % in_diff = 0;
% % for i = 1:m1
% %     for j = 1:n1
% %         temp = feat_LBP(i,j)./(1+(i-j).^2);
% %         in_diff1 = in_diff+temp;
% %     end
% % end
% % IDM1 = double(in_diff);
% % feat_disease2 = [Contrast1,Correlation1,Energy1,Homogeneity1, Mean1, Standard_Deviation1, Entropy1,Variance1, Smoothness1, Kurtosis1,IDM1];
% % 
% % 
% % %%%%TO TAKE COLOUR FEATURES%%%%%%
% % R = mean2(seg_img(:,:,1));
% % G = mean2(seg_img(:,:,2));
% % B = mean2(seg_img(:,:,3));
% % Co_Fea = [R G B];
% R = I(:,:,1);
% G = I(:,:,2);
% B = I(:,:,3);
% 
% 
% r=im2double(R);
% g=im2double(G);
% b=im2double(B);
% %%%%TO ENHANCE THE CONTRAST OF THE RESIZED IMAGE%%%%
% I = imadjust(I,stretchlim(I));
% % figure, imshow(I);title('CONTRAST ENHANCED IMAGE');
% 
%           %%%%%%%SEGMENTATION%%%%%%
% 
% %%%%PERFORM RGB TO HSV COLOUR TRANSFORMATION%%%%% 
% HSV=rgb2hsv(I);
% % figure,imshow(HSV),title('HSV COLOUR TRANSFORM IMAGE');
% %%SEPARATE THREE CHANNELS%%%
% H=HSV(:,:,1);
% S=HSV(:,:,2);
% V=HSV(:,:,3);
% 
% % figure,imshow(H),title('H-CHANNEL IMAGE');
% % figure,imshow(S),title('S-CHANNEL IMAGE');
% % figure,imshow(V),title('V-CHANNEL IMAGE');
% % 
% % figure,
% % subplot(1,3,1),imshow(H),title('H-CHANNEL');
% % subplot(1,3,2),imshow(S),title('S-CHANNEL');
% % subplot(1,3,3),imshow(V),title('V-CHANNEL');
% 
% %% PERFORM RGB TO GRAY CONVERSION ON THE V-CHANNEL IMAGE%%%%
% [m n o]=size(V);
% if o==3
%     gray=rgb2gray(V);
% else
%     gray=V;
% end
% % figure,imshow(gray);title('V- CHANNEL GRAY IMAGE');
% 
% % % % % % % % % % % ad = fcnBPDFHE(gray); 
% 
% 
% %%%%%%ADJUST THE CONTRAST OF THE GRAY CHANNEL IMAGE%%%%
% ad=imadjust(gray);
% % figure,imshow(ad);title('ADJUSTED GRAY IMAGE');
% 
% %%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
% bw=im2bw(gray,0.5);
% % figure,imshow(bw);title('BLACK AND WHITE IMAGE');
% 
% %%%%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
% bw=imcomplement(bw);
% % figure,imshow(bw);title('COMPLEMENT IMAGE');
% 
% %%%%TO PERFORM MORPHOLOGICAL OPERATIONS IN THE BW IMAGE%%%%
%  %%FILL HOLES%%
% bw=imfill(bw,'holes');
% % figure,imshow(bw),title('HOLES IMAGE');
%  %%DILATE OPERATION%%%
% SE=strel('square',3);
% bw=imdilate(bw,SE);
% % figure,imshow(bw),title('DILATED IMAGE');
% 
%  %%AGAIN FILL HOLES%%
% bw=imfill(bw,'holes');
%   %%TO REMOVE THE SMALL OBJECTS TO PERFORM IMOPEN OPERATIONS%%%
% SE=strel('disk',10);
% bw=imopen(bw,SE);
% % figure,imshow(bw),title('SMALL OBJECTS REMOVED IMAGE');
%   %%TO CLEAR THE UNWANTED BORDERS%%%
% bw=imclearborder(bw);
% % figure,imshow(bw),title('BORDER CORRECTED IMAGE');
% bw11=im2double(bw);
% 
% 
% %% Masking
% seg = bw11.*r;
% seg2 = bw11.*g;
% seg3 = bw11.*b;
% seg1=cat(3,seg,seg2,seg3);
% % figure,imshow(seg1);
% % title('Segmentation output');
%  
% R1 = seg1(:,:,1);
% G1 = seg1(:,:,2);
% B1 = seg1(:,:,3);
% 
% 
% 
% 
% 
%              %%%%%%%FEATURE EXTRACTION%%%%%%%%
%              
% %      %%%%FEATURE EXTRACTION BY GLCM(GREY LEVEL CO-OCCURENCE MATRIX)%%%%%
% % 
% g = graycomatrix(bw);
% stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
% Contrast = stats.Contrast;
% Correlation = stats.Correlation;
% Energy = stats.Energy;
% Homogeneity = stats.Homogeneity;
% Mean = mean2(bw);
% Standard_Deviation = std2(bw);
% Entropy = entropy(bw);
% Variance = mean2(var(double(bw)));
% a = sum(double(bw(:)));
% Smoothness = 1-(1/(1+a));
% Kurtosis = kurtosis(double(bw(:)));
% 
% 
% %%% Inverse Difference Movement%%%
% m = size(bw,1);
% n = size(bw,2);
% in_diff = 0;
% for i = 1:m
%     for j = 1:n
%         temp = bw(i,j)./(1+(i-j).^2);
%         in_diff = in_diff+temp;
%     end
% end
% IDM = double(in_diff);
% 
% 
% feat_disease1 = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy,Variance, Smoothness, Kurtosis,IDM];
% 
% 
% 
% 
% %%%%%FEATURE EXTRACTION BY LBP(LOCAL BINARY PATTERN)%%%%
% % % % step 1: Local Binary Patterns 
% feat_LBP = extractLBPFeatures(bw);
% grayImage = bw;
% localBinaryPatternImage1 = zeros(size(grayImage));
% [row col] = size(grayImage);
% for r = 2 : row - 1   
% 	for c = 2 : col - 1    
% 		centerPixel = grayImage(r, c);
% 		pixel7 = grayImage(r-1, c-1) > centerPixel;  
% 		pixel6 = grayImage(r-1, c) > centerPixel;   
% 		pixel5 = grayImage(r-1, c+1) > centerPixel;  
% 		pixel4 = grayImage(r, c+1) > centerPixel;     
% 		pixel3 = grayImage(r+1, c+1) > centerPixel;    
% 		pixel2 = grayImage(r+1, c) > centerPixel;      
% 		pixel1 = grayImage(r+1, c-1) > centerPixel;     
% 		pixel0 = grayImage(r, c-1) > centerPixel;       
% 		localBinaryPatternImage1(r, c) = uint8(...
% 			pixel7 * 2^7 + pixel6 * 2^6 + ...
% 			pixel5 * 2^5 + pixel4 * 2^4 + ...
% 			pixel3 * 2^3 + pixel2 * 2^2 + ...
% 			pixel1 * 2 + pixel0);
% 	end  
% end 
% 
% % figure,imshow(localBinaryPatternImage1),title('LBP-IMAGE'); 
% 
% 
%  %%%TAKE IMPORTANT FEATURES FROM THE LBP BY USING GLCM%%%%
%  feat_LBP=im2double(feat_LBP);
% g1 = graycomatrix(feat_LBP);
% stats1 = graycoprops(g,'Contrast Correlation Energy Homogeneity');
% Contrast1 = stats1.Contrast;
% Correlation1 = stats1.Correlation;
% Energy1 = stats1.Energy;
% Homogeneity1 = stats1.Homogeneity;
% Mean1 = mean2(feat_LBP);
% Standard_Deviation1 = std2(feat_LBP);
% Entropy1 = entropy(feat_LBP);
% Variance1 = mean2(var(double(feat_LBP)));
% a1 = sum(double(feat_LBP(:)));
% Smoothness1 = 1-(1/(1+a));
% Kurtosis1 = kurtosis(double(feat_LBP(:)));
% %%% Inverse Difference Movement%%%
% m1 = size(feat_LBP,1);
% n1 = size(feat_LBP,2);
% in_diff = 0;
% for i = 1:m1
%     for j = 1:n1
%         temp = feat_LBP(i,j)./(1+(i-j).^2);
%         in_diff1 = in_diff+temp;
%     end
% end
% IDM1 = double(in_diff);
% 
% feat_disease2 = [Contrast1,Correlation1,Energy1,Homogeneity1, Mean1, Standard_Deviation1, Entropy1,Variance1, Smoothness1, Kurtosis1,IDM1];
% 
% 
% %%%% TAKE COLOUR FEATURES FROM THE K-MEANS OUTPUT IMAGE %%%%%%
% R2 = mean2(seg1(:,:,1));
% G2 = mean2(seg1(:,:,2));
% B2 = mean2(seg1(:,:,3));
% Co_Fea = [R2 G2 B2];
% 
%       %%%%%COMBINE ALL THE FEATURES%%%%%%
% % feat_tot=[Co_Fea feat_disease1 feat_disease2]
% feat_tot=[Co_Fea feat_disease1 feat_disease2];
% fea_skin1=[fea_skin1;feat_tot];
% end
% save featureskin.mat fea_skin
save featureskin11.mat fea_skin11