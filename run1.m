%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% method (optional) -- Number Defines Reconstruction Algorithm To Use
% 0.)   OMP
% 1.)   CoSaMP
% 2.)   IHT
% 3.)   SP
% 4.)   L1 Dantzig Pd
% 5.)   L1 Decode Pd
% 6.)   L1 Eq Pd--BP
% 7.)   L1 Qc Logbarrier--BPDN
% 8.)   TVAL3
% 9.)   TV Dantzig Logbarrier
% 10.)   TV Eq Logbarrier
% 11.)   TV Qc Logbarrier
% 12.)   Bayesian Compressive Sensing
% 13.)   IRLS
% 14.)   Split Bregman

% R (optional) -- Measurement Basis Used to Sample Image
% 0.) gauss matrix
% 1.) Bernoulli matrix
% 2.) part Hadamard matrix
% 3.) The SparseRandom matrix
% 4.) Toeplitz matrix
% 5.) part Fourier matrix

% ww (optional) -- Representation basis.
% 0.) DCT 
% 1.) Hadamard
% 2.) DB-8 Wavelet
% 3.) DFT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear;close all

addpath('greedy algorithm' )           % greedy algorithm
addpath('other algorithms' )
addpath('l1_magic' )
addpath('TVAL3' )
addpath('images' )
addpath(['l1_magic' filesep 'Measurements']) % l1-magic
addpath(['l1_magic' filesep 'Optimization']) % l1-magic
addpath(['TVAL3' filesep 'Utilities']) % TVAL3
addpath(['TVAL3' filesep 'Solver']) % TVAL3
addpath(['TVAL3' filesep 'Fast_Walsh_Hadamard_Transform']) % TVAL3

%%  Read the file   
X=imread('cameraman.tif');
X=imresize(X,[64,64],'nearest');
image_size=size(X);
dimension=numel(image_size);
if dimension==3
X=rgb2gray(X);
end
X=double(X);
[a,b]=size(X);
percent = input('sampling rate：');%Setting of sampling rate
mmea = input('Whether to add noise or not:'); % 1-add noise   0-not add noise
if mmea==1
    mea = input('Please enter noise level：');%Setting of noise level
end
r = input('Please select a measurement matrix：');
psi = input('Please select a representation basis：');
M=round(a^2*percent);
method=input('Please select a reconstruction algorithm:');

%% Generation of representation basis
sze=a^2;
if psi == 0 % DCT matrix
%     mat_dct_1d=zeros(a^2,b^2);  % building the DCT basis (corresponding to each column)
%     for k=0:1:(a^2)-1
%         dct_1d=cos([0:1:(a^2)-1]'*k*pi/(b^2));
%         if k>0
%             dct_1d=dct_1d-mean(dct_1d); 
%         end
%     mat_dct_1d(:,k+1)=dct_1d/norm(dct_1d);
%     end
% ww=mat_dct_1d;
    ww = dctmtx(sze);
elseif psi == 1
%     ww = ifwht(eye(sze), sze, 'dyadic')';
    ww = (1/sze)*ifwht(eye(sze), sze, 'dyadic')';
elseif psi == 2
    [ww,~,~,longs] = wmpdictionary(sze, 'lstcpt', {'db8'});
    ww=full(ww);
elseif psi == 3
    ww = fft(eye(sze))/sqrt(sze);
    ww = ww';
%     ww=full(ww);
end

X1=X(:);

%% Create a sample pattern
N=a^2;
if r==0
%  gauss matrix
    R= randn(M,a^2);
    R = R/sqrt(M);
%     R=orth(R')';    

elseif r==1
% Generate Bernoulli matrix
    R = randi([0,1],M,a^2);%If your MATLAB version is too low,please use randint instead  
    R(R==0) = -1;
%     R=orth(R')';  
    
elseif r==2
% Generate part Hadamard matrix
    L_t = max(M,N);%Maybe L_t does not meet requirement of function hadamard  
    L_t1 = (12 - mod(L_t,12)) + L_t;  
    L_t2 = (20 - mod(L_t,20)) + L_t;   
    L_t3 = 2^ceil(log2(L_t));  
    L = min([L_t1,L_t2,L_t3]);%Get the minimum L    
    R = [];  
    Phi_t = hadamard(L);  
    RowIndex = randperm(L);  
    Phi_t_r = Phi_t(RowIndex(1:M),:);  
    ColIndex = randperm(L);  
    R = Phi_t_r(:,ColIndex(1:N));  
        

elseif r==3
% Generate SparseRandom matrix
    Phi = zeros(M,N);
    for ii = 1:N
        ColIdx = randperm(M);
        Phi(ColIdx(1:16),ii) = 1;
    end
    R=Phi;
    
elseif r==4
% Generate Toeplitz matrix
    u = randi([0,1],1,2*a^2-1);
    u(u==0) = -1;   
    Phi_t = toeplitz(u(a^2:end),fliplr(u(1:a^2)));
    R = Phi_t(1:M,:);
    
elseif r==5
% Generate part Fourier matrix
    Phi_t = fft(eye(N,N))/sqrt(N);%Fourier matrix
    RowIndex = randperm(N);
    Phi = Phi_t(RowIndex(1:M),:);%Select M rows randomly
    %normalization
    for ii = 1:N
        Phi(:,ii) = Phi(:,ii)/norm(Phi(:,ii));
    end
    R=Phi;
    

end
%%  Measurement

% X1=ww'*X1;
Y=R*X1;

%% Add noise
if mmea==1
    Y=awgn(Y,mea,'measured');
end
%%  Reconstruct image
A=R*ww;
s2=A'*Y;
s3=R'*Y;

tic;
if method==0
    rec=cs_omp1(Y,A,a^2,round(length(Y)/4));
    X3=ww*rec';

elseif method==1
    rec=CS_CoSaMP(Y,A,round(length(Y)/4));
    X3=ww*rec;
    
elseif method==2
    rec=hard_l0_Mterm(Y,A,a^2,round(length(Y)/4));
    X3=ww*rec;
    
elseif method==3
    rec=SP(Y,A,200);
    X3=ww*rec;

elseif method == 4          % L1 Method
    rec = l1dantzig_pd(s2,A,[],Y,5e-3,20);
    X3=ww*rec;
    
elseif method==5
    % large scale
    gfun = @(z) A*z;
    gtfun = @(z) A'*z;
    rec = l1decode_pd(s2, gfun, gtfun, Y, 1e-3, 25, 1e-8, 200);
%     rec = l1decode_pd(s2,A,[],Y,5e-3,20);
    X3=ww*rec;
    
elseif method==6
    % minimize l1 with equality constraints (l1-magic library)
    rec = l1eq_pd(s2,A,[],Y,5e-3,20);
    X3=ww*rec;
    
elseif method==7
    % minimize l1 with equality constraints (l1-magic library)
    rec = l1qc_logbarrier(s2,A,[],Y,5e-3,20);
    X3=ww*rec;
    
elseif method==8
    clear opts
    opts.mu = 2^8;%（2^4）~（2^13）
    opts.beta = 2^5;%（2^4）~（2^13）
    % opts.mu0 = 2^4;      % trigger continuation shceme
    % opts.beta0 = 2^-2;    % trigger continuation scheme
    opts.tol = 1.e-6; %1.e-2    determine the solution accuracy. Their smaller values result in a longer elapsed time and usually a better solution quality.
    % opts.tol_inn = 1.e-3;   %determine the solution accuracy. Their smaller values result in a longer elapsed time and usually a better solution quality.
    opts.maxit = 300;  % Total iterations
    opts.maxcnt = 10;  % Maximum external iterations
    opts.TVnorm = 1;      %Anisotropy (1 norm) and isotropy (2 norm) transformation
    opts.nonneg = true;   %Switch to non-negative model (true is non-negative)
    % opts.TVL2 = true;   %switch for TV/L2 models
    % opts.isreal = false;    % switch for real signals/images
    % opts.init = 1;  %1: A*b,,  0: 0,, U0: U0

    [rec, out] = TVAL3(R,Y,a^2,1,opts);
    X3=rec;
    
elseif method==9
    % minimize l1 with equality constraints (l1-magic library)
    rec = tvdantzig_logbarrier(s3,R,[],Y, 5e-3, 1e-3, 5, 1e-8, 1500);
    X3=rec;
    
elseif method==10
    % minimize l1 with equality constraints (l1-magic library)
    % large scale
%     Afun = @(z) A*z;
%     Atfun = @(z) A'*z;
%     rec = tveq_logbarrier(s2(:,i),Afun,Atfun,Y(:,i),1e-3, 5, 1e-8, 200);
    rec = tveq_logbarrier(s2,A,[],Y,1e-3, 5, 1e-8, 200);
    X3=ww*rec;
    
elseif method==11
    % minimize l1 with equality constraints (l1-magic library)
    rec = tvqc_logbarrier(s2,A,[],Y,1e-3, 5, 1e-8, 200);
    X3=ww*rec;
    
elseif method==12
%     s2 = pinv(A)*Y;
    sigma2 =  std(Y)^2/1e6;
    eta = 1e-8;
    [weights,used,sigma2,errbars] = BCS_fast_rvm(A,Y,sigma2,eta);
    rec = zeros(size(A,2),1);
    rec(used) = weights;
    X3=ww*rec;
    
elseif method==13
    rec=cs_irls(Y,A,a^2);
    X3=ww*rec;
    
elseif method==14
    rec=CS_SBIL1(Y,A,1,1,200);
    X3=ww*rec;
end
X4=reshape(X3,a,b);
toc;
%% Showing
figure;
subplot(121),imshow(X,[]);
title('(a)Original image');

if psi==3
    X4=flipud(X4);
    subplot(122),imshow(abs(X4),[]);
    title('(b)Reconstructed image');
else
    subplot(122),imshow(X4,[]);
    title('(b)Reconstructed image');
end

%% Image quality evaluation
I_R=X4;
I=X;
I_R=(I_R-min(I_R(:)))/(max(I_R(:))-min(I_R(:)));
I=(I-min(I(:)))/(max(I(:))-min(I(:)));
SSIM=ssim(I,abs(I_R));
PSNR=psnr(I_R,I);