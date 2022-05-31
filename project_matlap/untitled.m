function varargout = untitled(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @untitled_OpeningFcn, ...
                   'gui_OutputFcn',  @untitled_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end


function untitled_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
function varargout = untitled_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

i=imread('imge3.jpg');
imshow(i);
% --------------------------------------------------------------------
function operation_Callback(hObject, eventdata, handles)
% hObject    handle to operation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function addition_Callback(hObject, eventdata, handles)
% hObject    handle to addition (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
j=imread('image2.jpg');
result=imresize(i,[300,300]);
results=imresize(j,[300,300]);
add_image=imadd(result,results);
subplot(3,2,3),imshow(i),title("image1")
subplot(3,2,4),imshow(j),title("image2")
subplot(3,2,5),imshow(add_image),title("img1+img2")

% --------------------------------------------------------------------
function subtraction_Callback(hObject, eventdata, handles)
% hObject    handle to subtraction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
j=imread('image2.jpg');
result=imresize(i,[400,300]);
results=imresize(j,[400,300]);
sub_image=imsubtract(result,results);
subplot(3,2,3),imshow(i),title("image1")
subplot(3,2,4),imshow(j),title("image2")
subplot(3,2,5),imshow(sub_image),title("img1-img2")



% --------------------------------------------------------------------
function multiply_Callback(hObject, eventdata, handles)
% hObject    handle to multiply (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
j=imread('image2.jpg');
result=imresize(i,[400,300]);
results=imresize(j,[400,300]);
multi_image=immultiply(result,results);
subplot(3,2,3),imshow(i),title("image1")
subplot(3,2,4),imshow(j),title("image2")
subplot(3,2,5),imshow(multi_image),title("img1*img2")

% --------------------------------------------------------------------
function divide_Callback(hObject, eventdata, handles)
% hObject    handle to divide (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
j=imread('image2.jpg');
result=imresize(i,[400,300]);
results=imresize(j,[400,300]);
div_image=imdivide(result,results);
subplot(3,2,3),imshow(i),title("image1")
subplot(3,2,4),imshow(j),title("image2")
subplot(3,2,5),imshow(div_image),title("img1/img2")


% --------------------------------------------------------------------
function Complement_Callback(hObject, eventdata, handles)
% hObject    handle to Complement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% i=imread('image1');
% y=imcomplement(i);
% subplot(2,2,3),imshow(i);
% subplot(2,2,4),imshow(y);
p1=imread('image1.jpg')
subplot(2,2,3),imshow(p1),title('Original Image')
z= imcomplement(p1);
subplot(2,2,4),imshow(z),title('Complement Image')
 % --------------------------------------------------------------------
function rgb_to_gray_Callback(hObject, eventdata, handles)
% hObject    handle to rgb_to_gray (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
gray=rgb2gray(i);
subplot(2,2,3),imshow(i),title('Original Image')
subplot(2,2,4),imshow(gray),title('gray Image')
% --------------------------------------------------------------------
function rgb_to_binary_Callback(hObject, eventdata, handles)
% hObject    handle to rgb_to_binary (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
binary=im2bw(i);
subplot(2,2,3),imshow(i),title('Original Image')
subplot(2,2,4),imshow(binary),title('binary Image')


% --------------------------------------------------------------------
function color_image_operation_Callback(hObject, eventdata, handles)
% hObject    handle to color_image_operation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function changing_lighting_Callback(hObject, eventdata, handles)
% hObject    handle to changing_lighting (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I=imread('image1.jpg');
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
R=R+40;
G=G+50;
B=B+50;
LIGHT=cat(3,G,R,B);
subplot(2,2,3),imshow(I),title('orignal img')
subplot(2,2,4),imshow(LIGHT),title('LIGHT img')
% --------------------------------------------------------------------
function swapping_Callback(hObject, ~, handles)
% hObject    handle to swapping (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I=imread('image1.jpg');
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
GRB=cat(3,G,R,B);
BGR=cat(3,B,G,R);
RBG=cat(3,R,B,G);
subplot(1,3,1),imshow(GRB);
subplot(1,3,2),imshow(BGR);
subplot(1,3,3),imshow(RBG);

% --------------------------------------------------------------------
function Eliminating_Callback(hObject, eventdata, handles)
% hObject    handle to Eliminating (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I=imread('image1.jpg');
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);

eleminate_B=I;
eleminate_B(:,:,3)=0;


eleminate_G=I;
eleminate_G(:,:,2)=0;

eleminate_R=I;
eleminate_R(:,:,1)=0;

subplot(1,3,1),imshow(eleminate_B);
subplot(1,3,2),imshow(eleminate_G);
subplot(1,3,3),imshow(eleminate_R);


% --------------------------------------------------------------------
function histogram_Callback(hObject, eventdata, handles)
% hObject    handle to histogram (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Histogram_calculation_Callback(hObject, eventdata, handles)
% hObject    handle to Histogram_calculation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i = imread('image1.jpg');
gray = rgb2gray(i);
R = i(:,:,1);
G = i(:,:,2);
B = i(:,:,3);
hR = imhist(R);
hG = imhist(G);
hB = imhist(B);
subplot(3,3,4), imshow(gray), title('gray Image')
subplot(3,3,5), imhist(gray), title('histogram for gray')
subplot(3,3,7), bar(hR,'r'), title('Red histogram')
subplot(3,3,8), bar(hG,'g'), title('Green histogram')
subplot(3,3,9), bar(hB,'b'), title('Blue histogram')


% --------------------------------------------------------------------
function Histogram_Stretching_Callback(hObject, eventdata, handles)
% hObject    handle to Histogram_Stretching (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i = imread('image2.jpg');
gray=rgb2gray(i);
stretch = imadjust(gray);
subplot(3,2,3),imshow(gray),title('original image');
subplot(3,2,4),imhist(gray),title('original Histogram');
subplot(3,2,5),imshow(stretch), title('After stretching');
subplot(3,2,6),imhist(stretch),title('stretching Histogram');


% --------------------------------------------------------------------
function Histogram_Equalization_Callback(hObject, eventdata, handles)
% hObject    handle to Histogram_Equalization (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i = imread('image2.jpg');
gray=rgb2gray(i);
Equal = histeq(gray);
subplot(3,2,3),imshow(gray),title('original image');
subplot(3,2,4),imhist(gray),title('original Histogram');
subplot(3,2,5),imshow(Equal), title('After Equalization');
subplot(3,2,6),imhist(Equal),title('Equalization Histogram');


% --------------------------------------------------------------------
function Neighborhood_processing_Callback(hObject, eventdata, handles)
% hObject    handle to Neighborhood_processing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Linear_filter_Callback(hObject, eventdata, handles)
% hObject    handle to Linear_filter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function nonlinear_filter_Callback(hObject, eventdata, handles)
% hObject    handle to nonlinear_filter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Maximum_Callback(hObject, eventdata, handles)
% hObject    handle to Maximum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
X=imread('image1.jpg');
gray=rgb2gray(X);
y=nlfilter(gray,[3 3], 'max(x(:))');
subplot(2,2,3),imshow(gray),title('orignal image')
subplot(2,2,4),imshow(y),title('maximum filter')
% --------------------------------------------------------------------
function Minimum_Callback(hObject, eventdata, handles)
% hObject    handle to Minimum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
X=imread('image1.jpg');
gray=rgb2gray(X);
y=nlfilter(gray,[3 3], 'min(x(:))');
subplot(2,2,3),imshow(gray),title('orignal image')
subplot(2,2,4),imshow(y),title('minimam filter')

% --------------------------------------------------------------------
function Rank_Order_Callback(hObject, eventdata, handles)
% hObject    handle to Rank_Order (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
gray=rgb2gray(i);
j = ordfilt2(gray, 1, ones(3)) ;
subplot(2,2,3),imshow(gray),title('orignal image')
subplot(2,2,4),imshow(j),title('rank filter')
% --------------------------------------------------------------------
function Median_Callback(hObject, eventdata, handles)
% hObject    handle to Median (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
X=imread('image1.jpg');
gray=rgb2gray(X);
y=nlfilter(gray,[3 3], 'median(x(:))');
subplot(2,2,3),imshow(gray),title('orignal image')
subplot(2,2,4),imshow(y),title('median filter')

% --------------------------------------------------------------------
function Average_Callback(hObject, eventdata, handles)
% hObject    handle to Average (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i = imread('image2.jpg');
h = fspecial('average',5);
j = imfilter(i , h);
subplot(2,2,3),imshow(i),title('orignal')
subplot(2,2,4),imshow(j),title('average')


% --------------------------------------------------------------------
function Laplacian_Callback(hObject, eventdata, handles)
% hObject    handle to Laplacian (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i = imread('image2.jpg');
h = fspecial('laplacian');
j = imfilter(i , h);
subplot(2,2,3),imshow(i), title('Original Image')
subplot(2,2,4),imshow(j), title('HPF')


% --------------------------------------------------------------------
function Color_Space_Callback(hObject, eventdata, handles)
% hObject    handle to Color_Space (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function rgb_Callback(hObject, eventdata, handles)
% hObject    handle to rgb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I=imread('image1.jpg');
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
x=cat(3,R,G,B);
subplot(3,2,3), imshow(x), title('orginal Image')
subplot(3,2,4), imshow(R), title('R Channel')
subplot(3,2,5), imshow(G), title('G Channel')
subplot(3,2,6), imshow(B), title('B Channel')

% --------------------------------------------------------------------
function HSV_Callback(hObject, eventdata, handles)
% hObject    handle to HSV (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i=imread('image1.jpg');
HSV = rgb2hsv(i);
H = HSV(:,:,1);
S = HSV(:,:,2);
V = HSV(:,:,3);
subplot(3,2,3), imshow(HSV), title('HSV Image')
subplot(3,2,4), imshow(H), title('H Channel')
subplot(3,2,5), imshow(S), title('S Channel')
subplot(3,2,6), imshow(V), title('V Channel')

% --------------------------------------------------------------------
function HSI_Callback(hObject, eventdata, handles)
% hObject    handle to HSI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

i=imread('image1.jpg');
F=im2double(i);
r=F(:,:,1);
g=F(:,:,2);
b=F(:,:,3);
x1= 0.5*((r-g)+(r-b));
x2 = sqrt((r-g).^2 +(r-b).*(g-b));
th=acos(x1./x2);
rule = (b <= g);
H = (rule .* th) + ((1-rule).*(2 .* pi - th));
S=1 - (3.*(min(min(r,g),b)))./(r+g+b);
I=(r+g+b)/3;
hsi=cat(3,H,S,I);
subplot(2,2,3),imshow(i),title('orginal image')
subplot(2,2,4),imshow(hsi),title('HSI image')

% --------------------------------------------------------------------
function YCrCb_Callback(hObject, eventdata, handles)
% hObject    handle to YCrCb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
i = imread('image2.jpg');
YCbCr = rgb2ycbcr(i);
Y = YCbCr(:,:,1);
Cb = YCbCr(:,:,2);
Cr = YCbCr(:,:,3);
subplot(3,2,3), imshow(YCbCr), title('YCbCr Image')
subplot(3,2,4), imshow(Y), title('Y Channel')
subplot(3,2,5), imshow(Cb), title('Cb Channel')
subplot(3,2,6), imshow(Cr), title('Cr Channel')


% --------------------------------------------------------------------
function Edge_Detection_Callback(hObject, eventdata, handles)
% hObject    handle to Edge_Detection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Untitled_7_Callback(hObject, eventdata, handles)
% hObject    handle to Untitled_7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Sobel_Callback(hObject, eventdata, handles)
% hObject    handle to Sobel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image2.jpg');
gray=rgb2gray(I);
edgeS=edge(gray,'sobel');
subplot(2,2,3),imshow(I),title('original')
subplot(2,2,4),imshow(edgeS),title('sobel operator')
% --------------------------------------------------------------------
function prewit_Callback(hObject, eventdata, handles)
% hObject    handle to prewit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image2.jpg');
gray=rgb2gray(I);
edgeP=edge(gray,'prewitt');
subplot(2,2,3),imshow(I),title('original')
subplot(2,2,4),imshow(edgeP),title('prewitt operator')

% --------------------------------------------------------------------
function Roberts_Callback(hObject, eventdata, handles)
% hObject    handle to Roberts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image2.jpg');
gray=rgb2gray(I);
edgeR=edge(gray, 'roberts');
subplot(2,2,3),imshow(I)
title('original')
subplot(2,2,4),imshow(edgeR)
title('roberts operator')


% --- Executes on selection change in popupmenu1.


% --------------------------------------------------------------------
function Transformation_Callback(hObject, eventdata, handles)
% hObject    handle to Transformation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function one_image_Callback(hObject, eventdata, handles)
% hObject    handle to one_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function two_image_Callback(hObject, eventdata, handles)
% hObject    handle to two_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Push_Right_Callback(hObject, eventdata, handles)
% hObject    handle to Push_Right (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image1.jpg');

J = imread('image2.jpg');

[R, C, x]= size(I);

New = imresize(J,[R,C]);

for c = 1 : C
New(:, 1:c , :) = I(:, 1:c , :);
imshow(New);
end
% --------------------------------------------------------------------
function Push_left_Callback(hObject, eventdata, handles)
% hObject    handle to Push_left (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

I = imread('image1.jpg');
J = imread('image2.jpg');

[R, C, x]= size(I);

New = imresize(J,[R,C]);

for c = C : -1 : 1
New(:, c: C , :) = I(:, c:C , :);
imshow(New);
end

% --------------------------------------------------------------------
function Push_Down_Callback(hObject, eventdata, handles)
% hObject    handle to Push_Down (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image1.jpg');
J = imread('image2.jpg');

[R, C, x]= size(I);

New = imresize(J,[R,C]);

for r = 1 : R
New(1:r, : , :) = I(1:r, : , :);
imshow(New);
end

% --------------------------------------------------------------------
function Push_up_Callback(hObject, eventdata, handles)
% hObject    handle to Push_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

I = imread('image1.jpg');
J = imread('image2.jpg');

[R, C, x]= size(I);

New = imresize(J,[R,C]);

for r = R : -1 : 1
New(r:R, : , :) = I(r:R, : , :);
imshow(New);
end

% --------------------------------------------------------------------
function spilit_out_Callback(hObject, eventdata, handles)
% hObject    handle to spilit_out (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
image1=imread('image1.jpg');
image2=imread('image2.jpg');
[R,C,N]=size(image1);
image2=imresize(image2,[R,C]);
for h=1:C/2
    image1(:,C/2-h+1:C/2,:)=image2(:,C/2-h+1:C/2,:);
    image1(:,C/2:C/2+h,:)=image2(:,C/2:C/2+h,:);
    imshow(image1);
end

% --------------------------------------------------------------------
function spilit_up_Callback(hObject, eventdata, handles)
% hObject    handle to spilit_up (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
image1=imread('image1.jpg');
image2=imread('image2.jpg');
[R,C,N]=size(image1);
image2=imresize(image2,[R,C]);
for h=1:R/2
    image1(R/2-h+1:R/2,:,:)=image2(R/2-h+1:R/2,:,:);
    image1(R/2:R/2+h,:,:)=image2(R/2:R/2+h,:,:);
    imshow(image1);
end

% --------------------------------------------------------------------
function Scale_Callback(hObject, eventdata, handles)
% hObject    handle to Scale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image1.jpg');

x = imresize(I, [200,200]);

subplot(2,2,3), imshow(I),title('orginal image')
subplot(2,2,4), imshow(x),title('scale')

% --------------------------------------------------------------------
function Rotate_Callback(hObject, eventdata, handles)
% hObject    handle to Rotate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image2.jpg');

Angle = -90;
x = imrotate(I, Angle);

subplot(2,2,3), imshow(I),title('orginal image')
subplot(2,2,4), imshow(x),title('rotate')

% --------------------------------------------------------------------
function Reflection_Callback(hObject, eventdata, handles)
% hObject    handle to Reflection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('image2.jpg');

% Flib Vertical
x1 = flipud(I);

% Flib Horizontal
x2 = fliplr(I);

subplot(3,2,3:4), imshow(I) , title('Original')
subplot(3,2,5), imshow(x1) , title('Flib Vertical')
subplot(3,2,6), imshow(x2) , title('Flib Horizontal')


% --------------------------------------------------------------------
function Mathematical_Morphology_Callback(hObject, eventdata, handles)
% hObject    handle to Mathematical_Morphology (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Dilation_Callback(hObject, eventdata, handles)
% hObject    handle to Dilation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im=imread('noise.jpg');
sq=ones(2,2);
td=imdilate(im, sq);
subplot(2,2,3),imshow(im),title('orginal image')
subplot(2,2,4),imshow(td),title('Dilation image')

% --------------------------------------------------------------------
function Erosion_Callback(hObject, eventdata, handles)
% hObject    handle to Erosion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im=imread('noise.jpg');
sq=ones(2,2);
td=imerode(im, sq);
subplot(2,2,3),imshow(im),title('orginal image')
subplot(2,2,4),imshow(td),title('Erosion image')

% --------------------------------------------------------------------
function Opening_Callback(hObject, eventdata, handles)
% hObject    handle to Opening (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

im=imread('noise.jpg');
im=double(im);
im=imnoise(im,'salt & pepper',.01);
sq=ones(2,2);
td=imopen(im, sq);
subplot(1,2,1),imshow(im),title('orginal image')
subplot(1,2,2),imshow(td),title('Opening image')
% --------------------------------------------------------------------
function Closing_Callback(hObject, eventdata, handles)
% hObject    handle to Closing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im=imread('noise.jpg');
im=double(~im);
im=imnoise(im,'salt & pepper',.01);
sq=ones(2,2);
td=imclose(im, sq);
subplot(1,2,1),imshow(im),title('orginal image')
subplot(1,2,2),imshow(td),title('Closing image')
