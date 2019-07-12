for img = [8 9 16 17 19 20 21 22 23]
    inputfile = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\alignmentMatch',num2str(img),'.tif'];
    inputImg = imread(inputfile);
    [m,n] = size(inputImg);
    for i = 1:64:m
        for j = 1:64:n
            savepath = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\patches\input\match',num2str(img),'-',num2str((i+63)/64),'-',num2str((j+63)/64),'.tif'];
            %         imwrite(uint16(A(i:i+255,j:j+255)),['patch' num2str(i) num2str(j) '.png'],'png');
            inputimg = inputImg(i:i+63,j:j+63);
            imwrite(mat2gray(inputimg),savepath);
        end
    end
    
    outputfile = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\',num2str(img),'.tif'];
    outputImg = imread(outputfile);
    [m,n] = size(outputImg);
    for w = 1:256:m
        for q = 1:256:n
            savepath = ['D:\LeeX\deep-learning-microscopy\dataPreprocess\group1\patches\output\match',num2str(img),'-',num2str((w+255)/256),'-',num2str((q+255)/256),'.tif'];
            %         imwrite(uint16(A(i:i+255,j:j+255)),['patch' num2str(i) num2str(j) '.png'],'png');
            outputImgimg = outputImg(w:w+255,q:q+255);
            imwrite(mat2gray(outputImgimg),savepath);
        end
    end
    
end