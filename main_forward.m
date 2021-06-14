clc;
clear all;
close all;

format short;

%% parameters definition
lambra = 606;% [nm] % the peak emission wavelength of mScarlet-I
%     lambra =540;  % [nm] % the peak emission wavelength of EYFP

% Scattering Length compuation
a=29.11*10^(-3); b=3.28; % costant coeficient for brain tissue: Ref: EUNJUNG MIN et al, BIOMEDICAL OPTICS EXPRESS,Vol. 8, No. 3, 2017. 
sl_em=1/(a*(lambra/500)^(-b));

NA = 0.95;  % numerical aperture
r_ex =  1.22*lambra*1e-3/NA;   % excitaion resolution
    
exp_t_factor = 3;  % The constant power coeficient to make sure that the synthetic TFM images are in the same power level as experiments
dataPath_root = './PSTPM_data/';  % data root
files = dir([dataPath_root '**/*.tif*']);   % data files are tif
z_start=1; % The start depth in PSTPM image
z_end=64; % The end depth in PSTPM image 
NN_x=64;NN_y=64;  % The synthetic TFM images in x and y dimension; changed to 128*128 in the manuscript
NN_z=z_end-z_start+1; % The total number of depths
NN_t=round(length(files)*2); % total number of data, 10 means that for each tif, we randomly generate 10 stacks by randomly chose the center points.
NN_p=round(250/170*800); % total number of pixels in TFM：the pixel size of PSTPM is 250nm, wheras the pixel size of TFM is 170nm 
dx_gt = (0.25*800)/NN_p;% The TFM pixel size
dx = dx_gt;   % [um] TF image's pixel size
nn_z=round(NN_t/(length(files))); % number of patters generated from each tif: should be integer
Count_nn_tem=1;  
np=100;    % the maximum photon value used for normarlization
inx_tem1=100; % Avoid use boundary pixels of the tif PSTPM images;

%% Loop PSTPM files to generate training data.
for i=1:length(files)
    fname = fullfile(files(i).folder,files(i).name);
    info = imfinfo(fname);
    Nz = numel(info);
    kk=1;  % the z dimension
    z=z_start  % The initialization of z
    for mm=64:-1:1
        j=mm;  % loop for depth：Red Channel image
        sPSF = sim_get_modeled_sPSF(z,sl_em,dx,round(0.5*NN_p),r_ex);   % [um] simulated PSF
        
        I_temp1 = single(imread(fname,j));  % Red channel
        I_temp2 = imresize(I_temp1,[NN_p NN_p]);
        I_temp=exp_t_factor*round(I_temp2);  % match power in exp
        
        I_temp3=I_temp;
        
        J_temp = conv2(I_temp3,sPSF,'same');  % scale magnification
        J_temp(J_temp<0)=0;
        
        J_temp = poissrnd(J_temp);
        J_temp= J_temp;
        J_temp=round(J_temp);
        
        I_temp(I_temp<0)=0;
        I_temp(I_temp>np)=np;
        
        J_temp(J_temp<0)=0;
        J_temp(J_temp>np)=np;
        
        [x_inx,y_inx]=find(I_temp>0.2*np);
        Inx_tem=(x_inx>inx_tem1).*(y_inx>inx_tem1).*(x_inx<NN_p-inx_tem1).*(y_inx<NN_p-inx_tem1);
        AA=find(Inx_tem>0.1);
        x_inx_rand=x_inx(AA);
        y_inx_rand=y_inx(AA);
        len=length(x_inx_rand)-1; % Avoid zero indx
        
        for nn=Count_nn_tem:Count_nn_tem+nn_z-1  % loop to randomly select NN_x*NN_y small patterns from NN_p*NN_p large pattern
            rand('state', nn);
            Ind_x=x_inx_rand(round(len*rand)+1);
            rand('state', nn);
            Ind_y=y_inx_rand(round(len*rand)+1);
            Ind_xr=Ind_x-round(NN_x/2):Ind_x-round(NN_x/2)+NN_x-1;  % Inx range
            Ind_yr=Ind_y-round(NN_y/2):Ind_y-round(NN_y/2)+NN_y-1;
            temp_1=I_temp(Ind_xr,Ind_yr);
            temp_2=J_temp(Ind_xr,Ind_yr);

            I_out(kk,:,:,nn)=single((temp_1).'); % for h5py loading in python
            I_in(kk,:,:,nn)=single((temp_2).');  % for h5py loading in python
        end
        kk=kk+1;
        if kk>NN_z
            Count_nn_tem=nn+1;  % if all z direction has counted, then start count at maximum point
        end
        z = z+1
    end
    
end
%% display one example
figure; imagesc(squeeze((I_out(1,:,:,1)))); colormap hot; colorbar;title('GT')
figure; imagesc(squeeze((I_in(1,:,:,1)))); colormap hot; colorbar;title('In')

%% Save mat data for training purpose 
M_in=min(min(min(min(I_in))));
M_out=min(min(min(min(I_out))));
M_coef=min([M_in,M_out]);
I_in=I_in-M_coef;
I_out=I_out-M_coef;
save('Training_SpineN_in','I_in')
save('Training_SpineN_out','I_out')
clear all;













