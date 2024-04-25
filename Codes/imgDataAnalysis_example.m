%=========================================================================%
% Handong Global University
% Course    : Mechatronics Integrated Projects
% About     : Image data analysis
% Created   : 2022.03.14
%=========================================================================%
clc; clear; close all;
cd C:\UNIVERSE_GP76\HGU\2022\CAPSTONE\Data\city_1_0\Navtech_Polar

intensity_read = imread('000001.png');
imshow(intensity_read);
slope_r = 100/(size(intensity_read, 1) - 1);
slope_deg = 360/(size(intensity_read, 2) - 1);
itc_r = -slope_r;
itc_deg = -slope_deg;
% Polar coordinates calculations
for i = 1 : size(intensity_read, 1)
    r(i) = slope_r*i+itc_r;
end
for i = 1 : size(intensity_read, 2)
    deg(i) = slope_deg*i+itc_deg;
end

for j = 1 : size(intensity_read, 2)

    for i = 1 : size(intensity_read, 1)
        xpos(i,j) = r(i)*cosd(deg(j));
        ypos(i,j) = r(i)*sind(deg(j));
         
    end
end

figure(2), plot(xpos);
figure(3), plot(ypos);
figure(2), imshow(xpos);
figure(3), imshow(ypos);


%im2 = imread('000001.png');
