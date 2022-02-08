clear all;
close all;
% clc;
ImgSer=3;
FPath=['D:\CE\IM\3D Scanner\codesOr\printrun\pics\test2\'];
load cameraParams5;

fs=mean(cameraParams5.FocalLength);
u0=cameraParams5.PrincipalPoint;
% u0 = [uu0(2),uu0(1)];
%fs = [ffs(2),ffs(1)];
R=cameraParams5.ImageSize(1);
C=cameraParams5.ImageSize(2);

%hamun right===================
load LsrPlnRight28mehr;
Vn=LsrPRight(1:3)';
d=LsrPRight(4);

load LsrPlnLeft28mehr;
VnL=LsrPLeft(1:3)';
dL=LsrPLeft(4);

%cam=webcam(3);
%cam.ExposureMode='manual';
%cam.Exposure=-4;

%bias and step of moving down the x axis
step = 40;
bias = 0;
P=[];
%preview(cam);
for i=60:148
    %f=snapshot(cam);
    
    FileName=[FPath,'dast',num2str(i),'.jpg'];
    %real=[FPath,'img',num2str(i),'.bmp'];
    f=imread(FileName);
    f = imrotate(f,270);
    g=f(:,:,1);
    %ff=imread(real);

    ZL=zeros(1,C);

     I3=g;
     avg = mean(I3(:));

     BW = im2bw(f,0.15);
     www=zeros(R,C); 

     % It changes the sensetivity of red color detection
     www(:,:)=uint8(3);


     g=I3.*uint8(www);
    %  I5=f(:,:,1)+I4;
    %  I5(:,:,2)=f(:,:,2);
    %  I5(:,:,3)=f(:,:,3);
    %  imshow(I5)
    temp2=double(imregionalmax(g));
    temp2=double(im2bw(g,0.9));% i convert to this to see result
    %%strel function is for morphology 
    %%and with 'line' means create line shape with length 11 in degree 90
    erodedBW = imerode(temp2,strel('line',21,90));
%     if(i==9)
%         pause
%     end
%     erodedBW = imerode(erodedBW,strel('line',21,90));

    %blu=imgaussfilt(temp2,[16 1]);

%     figure,imshow(erodedBW)
%     pause
  %  figure,imshow(temp2)
    temp=zeros(R,C);
    for r=1:R
        %temp(r,:)=imregionalmax(g(r,:));
        [M,mP]=max(erodedBW(r,:));

        %L=g(r,:);
        %commnet this on 4/08/99
%         g(r,:)=ZL;
%         if M ~= 0
% %         if M > 20
%             g(r,mP)=1;
%         end
        %end comment
    %     if M>80
    %         g(r,mP)=255;
    %     end
        
        ii=mP;
        while(erodedBW(r,ii)~=0 && ii<480)
            erodedBW(r,ii)=0;
            ii=ii+1;
        end
        
        %add this part on 04/08/99
        g(r,:)=ZL;
        if mP ~= ii
            center = floor((mP + ii)/2);
%         if M > 20
            g(r,mP)=1;
        end
        %end this part
        
        [M,mP]=max(erodedBW(r,:));
        ii=mP;
        while(erodedBW(r,ii)~=0 && ii<480)
            erodedBW(r,ii)=0;
            ii=ii+1;
        end
        if mP ~= ii
            center = floor((mP + ii)/2);
%         if M > 20
            g(r,mP)=1;
        end
    %     if M>80
    %         g(r,mP)=255;
    %     end

    end
    % figure;
   % imshow(g)

    
    D=zeros(R,C);
    w=[];
    firstSaw = 1;
    dim = 1;
    
    for r=1:R
%         if sum(g(r,:))~=2
%             continue
%         end
        for c=1:C
            if g(r,c)==1 && firstSaw==0%%left lsr
%                 p=([c r]-u0)/fs;
%                 z=-dL/(VnL*[p 1]');
%                 xy=z*p;
%                 xy(dim)= xy(dim)+bias*step;
%                 %z= z+bias*step;
%                 P=[P; xy z];
%                 D(r,c)=z;
%                 w=[w;r c];
%                 firstSaw = 1;
            elseif g(r,c)==1 && firstSaw==1%%right lsr
                p=([c r]-u0)/fs;
                z=-d/(Vn*[p 1]');
                xy=z*p;
                xy(dim)= xy(dim)+bias*step;                
                %z= z+bias*step;
                P=[P; xy z];
                D(r,c)=z;
                w=[w;r c];
                firstSaw = 1;
            end
        end
    end
    plot(sum(D'),'.')
    bias=bias+1;
%imshow([mat2gray(D);mat2gray(g)])
end



figure,
plot3(P(:,1),P(:,2),P(:,3),'.')

size1=size(P(:,1));
size2=size(P(:,2));
size3=size(P(:,3));
newX=zeros(size1(1),size2(1));
newY=zeros(size2(1),size3(1));
newZ=zeros(size1(1),size3(1));