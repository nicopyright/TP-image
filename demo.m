% Demo Lecture et Affichage Image Couleur
%
clear all; close all;
%
%  lecture et affichage d'une image: 
nom=input('Entrer le nom image avec le format jpg, bmp ou tif :','s'); % l'image doit être dans le répertoire
a=imread(nom);
figure(1);
subplot(2,2,1);
imagesc(a);
%
% transformation image en niveau de gris:
rouge=double(a(:,:,1));
vert=double(a(:,:,2));
bleu=double(a(:,:,3));
subplot(2,2,2);
imagesc(rouge, [min(min(rouge)) max(max(rouge))]);
subplot(2,2,3);
imagesc(vert, [min(min(vert)) max(max(vert))]);
subplot(2,2,4);
imagesc(bleu, [min(min(bleu)) max(max(bleu))]);
colormap(gray);

%
% Transformation image couleur en une image niveaux de gris:
%
image=double(rgb2gray(a));
figure(2);
imagesc(image, [min(min(image)) max(max(image))]);
colormap(gray);



