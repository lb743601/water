clc;
clear all;
for i=0:255
% input_path = 'E:\【采集】可见光-采集数据\水面倒影\3_out\3_out\'; 
I1=(im2double(imread(['E:\2023.6.18\3_out\1\', num2str(i), '.png'])));%I1=rgb2gray(I1);
I2=(im2double(imread(['E:\2023.6.18\3_out\2\', num2str(i) , '.png'])));%I2=rgb2gray(I2);
I3=(im2double(imread(['E:\2023.6.18\3_out\3\' , num2str(i), '.png'])));%I3=rgb2gray(I3);
I4=(im2double(imread(['E:\2023.6.18\3_out\4\' , num2str(i) , '.png'])));%I4=rgb2gray(I4);
s0=(1/2).*(I1+I2+I3+I4);
s1=(I1-I3);s2=(I2-I4);
DoLP = sqrt(s1.^2 + s2.^2)./s0; %DoLP(isnan(DoLP)) = 0;DoLP_disp=lin2rgb(DoLP);
AoLP = 0.5*atan2(s2,s1);
%%
P_R=0.65;
I_D=(I1+I3).*(1-sqrt(s1.^2 + s2.^2)./(s0.*(P_R)));
I_D=(I_D-min(I_D(:)))./(max(I_D(:))-min(I_D(:)));
% figure,imshow(s0,[]);
% figure,imshow(I_D,[]);
imwrite(I_D,['.\Reflection\3\', num2str(i), '.png']); 
end