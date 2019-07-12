%% correlation match %%
clear;
reource_p=mat2gray(imread('D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\1.tif'));
[m,n]=size(reource_p);
for img = [10 11]
    filepath = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\',num2str(img),'.tif'];
    savepath = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\match',num2str(img),'.tif'];
    reource_p_sub=mat2gray(imread(filepath));
    reource_p_sub=imresize(reource_p_sub,0.25,'bicubic');
    
    [m0,n0]=size(reource_p_sub);
    result=zeros(m-m0+1,n-n0+1);
    vec_sub = double( reource_p_sub(:) );
    norm_sub = norm( vec_sub );
    for i=1:m-m0+1
        for j=1:n-n0+1
            subMatr=reource_p(i:i+m0-1,j:j+n0-1);
            vec=double( subMatr(:) );
            result(i,j)=vec'*vec_sub / (norm(vec)*norm_sub+eps);
        end
    end
    %Find the most relevant location
    [iMaxPos,jMaxPos]=find( result==max( result(:)));
    
    figure,
    subplot(121);imshow(reource_p_sub),title('Matching template subimage');
    subplot(122);
    imshow(reource_p);
    title('Mark the matching area'),
    hold on
    plot(jMaxPos,iMaxPos,'*');%Draw the largest relevant point
    %Mark the matching area with a rectangular box
    plot([jMaxPos,jMaxPos+n0-1],[iMaxPos,iMaxPos]);
    plot([jMaxPos+n0-1,jMaxPos+n0-1],[iMaxPos,iMaxPos+m0-1]);
    plot([jMaxPos,jMaxPos+n0-1],[iMaxPos+m0-1,iMaxPos+m0-1]);
    plot([jMaxPos,jMaxPos],[iMaxPos,iMaxPos+m0-1]);
    matchMat = reource_p(iMaxPos:iMaxPos+m0-1,jMaxPos:jMaxPos+n0-1);
    imwrite(uint16(matchMat), savepath);
end