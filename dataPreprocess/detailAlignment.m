clear
clc
close all
for img = [8 9 16 17 19 20 21 22 23]
    originalpath = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\',num2str(img),'.tif'];
    distortedpath = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\match',num2str(img),'.tif'];
    savepath = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\alignmentMatch',num2str(img),'.tif'];
    %Step 1: Read Image
    original = mat2gray(imread(originalpath));
    original = imresize(original,0.25,'bicubic');
    distorted = mat2gray(imread(distortedpath));
%     figure;imshow(original);title('original');
    
    % Find Matching Features Between Images
    ptsOriginal  = detectSURFFeatures(original);
    ptsDistorted = detectSURFFeatures(distorted);
    [featuresOriginal,  validPtsOriginal]  = extractFeatures(original,  ptsOriginal);
    [featuresDistorted, validPtsDistorted] = extractFeatures(distorted, ptsDistorted);
    indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
    matchedOriginal  = validPtsOriginal(indexPairs(:,1));
    matchedDistorted = validPtsDistorted(indexPairs(:,2));
%     figure;
%     showMatchedFeatures(original,distorted,matchedOriginal,matchedDistorted);
%     title('Putatively matched points (including outliers)');
    
    %estimate transformation
    [tform, inlierDistorted, inlierOriginal] = estimateGeometricTransform(...
        matchedDistorted, matchedOriginal, 'similarity');
    figure;
%     showMatchedFeatures(original,distorted,inlierOriginal,inlierDistorted);
%     title('Matching points (inliers only)');
%     legend('ptsOriginal','ptsDistorted');
    
    %Solve for Scale and Angle
    Tinv  = tform.invert.T;
    
    ss = Tinv(2,1);
    sc = Tinv(1,1);
    scaleRecovered = sqrt(ss*ss + sc*sc)
    thetaRecovered = atan2(ss,sc)*180/pi
    
    %Recover the Original Image
    outputView = imref2d(size(original));
    recovered  = imwarp(distorted,tform,'OutputView',outputView);
    %figure, imshowpair(original,recovered,'montage')
    s = original-recovered;
    figure;
    subplot(1,3,1),imshow(original),title('original')
    subplot(1,3,2),imshow(recovered),title('recovered')
    subplot(1,3,3),imshow(s),title('subtraction')
    %save data
    distorted = imread(distortedpath);
    recovered  = imwarp(distorted,tform,'OutputView',outputView);
    imwrite(uint16(recovered), savepath);
    clear
end