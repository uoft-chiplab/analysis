scanWithImages
function out = scanWithImages(s)
% Usage: Save in current day's folder under unique name for specific run.
% Enter corresponding filenames and specify evaluation and output
% parameters in the preamble. Hit F5 or press 'run' to run. ASCII data will
% be written to .dat files for further usage. Can also be called from a
% batch with input s.
%
% Built for scanEval - V2.0.0 (October 2012)
%
%   gerty.m (optional) -- silly but fun.
%   idlmwrite.m -- writing ASCII data files with headers
%   scanread.m 
%   scanInitialize.m
%   scanProcessResults.m
%   scanSaveResults.m
%   scanOutputResults.m
%   scanGetNamePos.m
%   scanCheckStructure.m
%   initializeCameraChip.m -- initializes camera chip structure
%   chipPixisImages.m -- loads pixis images saved by camera program
% And any other function used in the evaluation loop.
%
% Make sure files 'scan*.m' belong to the right version of the scanEval
% package. Note that this little package is subject to continuous beta
% testing, improvement and change. Compatibility between different versions
% is intended, though not always pertained.
tic
if ~exist('s', 'var') % for direct evaluation
    % SCAN FILES AND TARGET LOCATION
    filenames = mfilename;  % Takes current filename. Used to be given manually: filenames = 'C_TShots';
    param='HF AM'; %scan parameter/s
    param2='HF AM img2';
    s.imgsubpath = './imgs/';
    s.eigenclean = 1;  
    s.rescalebyblanks =0; % doesn't do anything to FF
    s.subtractbg = 0;
    
    % RANGE OF EVALUATION/ CYCLE OFFSET
    s.eval.excludelist=[]; %Include file number of images you don't want to include
    s.eval.first = 0; s.eval.last = 1000;  
    % OUTPUT SPECIFICATIONS
    s.out.scannames = {param,param2};% parameter names (gives default names to unspecified)
    s.out.plotWhich = {{'cyc',{'ROIsum','LiNfit'}},{param,{'ROIsum','LiToTF','LiNfit','ToTFcalc'}}}; % which parameters to plot
    s.out.addnames = {'Nfit','BZ-2','BZ-1','BZ0','BZ1','BZ2','Excited', ...
        'c5','c7','c9','fraction95','fraction75','fraction97','sum95', 'ODmean', 'refmean' ...
        }; % names of additional columns
    s.out.groupby = ''; % name (or index) of variable to group data by (nested scan overrides this)    
    s.out.talk = 0; % show images and fits? (turn off to speed up eval)
    s.out.gerty = 0; % a trip to the moon?
    s.out.individualFiles = 1; % save individual files for each individual grouping value?
    s.out.niceimage = ''; %I forget what this does
    s.out.average = 0; % average data (be careful, this does not work with multiple scan parameters
    s.out.report_std = 1; % 0=report se-td unc of mean, 1=report std dev KX Mar 27 2021
    %KJ Mar 2021, Added method to get plot number from current directory. Must be bypassed to run from G or H drive 
    
    a=pwd; 
    smon=a(14:15); 
    %this assumes the usual D:Data/year/ format
    sday=a(25:26); %this depends on the month currently which is not ideal
    srun= num2str(double(upper(filenames(1)))-double('A')+1); %takes letter from run name
    if str2double(srun)<10
        s.out.fig = str2double([smon sday '0' srun]); % MMDDXX
       
    else 
        s.out.fig = str2double([smon sday srun]); % MMDDXX
    end
    s.out.fig = datenum(datetime("today")); % MMDDXX, XX is the numeric value for the letter.
    
    s.path = './'; 
    s.scanfile = strcat(filenames, '.mscan'); % automatically checks for '.scan2' as well
    s.incscanfile = 'commands/incremental.txt'; % concatenated to s.path
    s.parameter1file = strcat(filenames, '.txt');
    s.parameter2file = ''; % enter empty string for one-parameter scans 
    %Use to override the default in case of scan's going past Z for example

    
%   s.out.plotWhich = {{'dummy',{'LiNfit','G_ctr_x','G_ctr_y','G_sigma_x','G_sigma_y','G_ar','fCtr1','fWidth1'}},{'cyc',{'ROIsum','LiNfit'}}}; % which parameters to plot
%   s.out.plotWhich = {{'VVA',{'c5','c9','fraction95','sum95'}},{'cyc',{'sum95','ROIsum'}}}; % which parameters to plot
%   s.out.plotWhich = {{'Delay',{'c5','c9','fraction95','ROIsum','two2D_cv1','two2D_cv2','two2D_cvd','two2D_ch1','two2D_ch2','two2D_chd'}},{'cyc',{'sum95','ROIsum'}}}; % which parameters to plot
%   s.out.plotWhich = {{'FB',{'two2D_cv1','two2D_cv2','two2D_cvd','two2D_ch1','two2D_ch2','two2D_chd','c5','c9'}},{'cyc',{'sum95','ROIsum'}}}; % which parameters to plot
%   s.out.plotWhich = {{'dummy',{'fCtr1','fWidth1','ROIsum'}},{'cyc',{'ROIsum'}}}; % which parameters to plot
%   s.out.plotWhich= {{'pulse time (us)',{'BZ0','fWidth1','ROIsum'}},{'cyc',{'ROIsum'}}}; % which parameters to plot
%   s.out.plotWhich = {{'Latt (V)',{'BZ0','BZ1','BZ2','Excited','ROIsum'}},{'cyc',{'ROIsum'}}}; % which parameters to plot
%   s.out.plotWhich = {{'ODT1',{'Tx','Ty','LiNfit','LiToTF','ToTFcalc'}},{'cyc',{'ROIsum','LiToTF','LiNfit'}}}; % which parameters to plot
%   s.out.plotWhich = {{'dummy',{'Tx','Ty','LiNfit','LiToTF','ToTFcalc','G_ctr_x','G_ctr_y','G_ar','G_sigma_x'}},{'cyc',{'ROIsum','LiToTF','LiNfit'}}}; % which parameters to plot

    % EVALUATION SETTINGS

    s.eval.reevaluate = 1;
    s.eval.autoSkipBadImages = 1;
       
    % region of interest
    s.eval.ROI = [37 250 350 750]; % rowI rowF colI colF
    %s.eval.ROI = [28 250 250 850]; % rowI rowF colI colF
    % 8 <= rowI < rowF <= 256
    % colI < colF < 1024?
    %s.eval.ROI = [80 190 400 700];s.eval.ROI = [40 100 620 680];%s.eval.ROI = [20 170 400 625];
    s.ecbox = [30 200 90 250];
    % box that encompasses atoms for eigenclean method (within ROI)
     
    % Marginal fit parameters and settings (use for simple Gaussian
    % analysis)
    s.eval.fit.initial = [0 5 101 125 0 5 135];
    s.eval.fit.numpars = length(s.eval.fit.initial);
    s.eval.fit.fix  = [[5 6 7]; ... % parameter #
        [0 10 120]];  % value
    s.eval.fit.limits = [[ 4  7]; ... % parameter #
        [0  0]; ... % lower bound
        [500  500]]; % upper bound
    s.eval.fit.names = {'fBg', 'fA1', 'fWidth1', 'fCtr1', ...
        'fA2', 'fWidth2', 'fCtr2'}; % names of the fit parameters
    s.eval.fit.sumdir = 1;% 1 for horizontal marginal (ie. sum along vertical), 2 for vert.
    s.eval.fit.do = 0;  % whether to do the fit or not
    
    capExpr = '(\d{2})([a-zA-Z]+)(\d{4})\W([A-Z])'; 
    capStr = regexp(pwd, capExpr,'tokens'); %i.e: (29)(September)(2023)\(F)
    monthchar = num2str(month(datetime(char(capStr{:}(2)), 'InputFormat', 'MMMM')),'%.2d'); %'September' -> '09'
    
    %%%%%%%%%% KX: OCT 2 2023 %%%%%%%%%
    if s.eigenclean
        %s.outfile = strcat(filenames, '_e.dat'); 
        s.outfile = strcat(char(capStr{:}(3)), '-', monthchar, '-', char(capStr{:}(1)), '_', char(capStr{:}(4)), '_e.dat');
        s.monitorfile = strcat(filenames, '_e.mat');
    else
        %s.outfile = strcat(filenames, '.dat');
        s.outfile = strcat(char(capStr{:}(3)), '-', monthchar, '-', char(capStr{:}(1)), '_', char(capStr{:}(4)), '.dat');
        s.monitorfile = strcat(filenames, '.mat');
    end 
    
    s.imgmask = strcat(filenames, '_\d{4}\.mat');% Regexp for image mask
    s.blanksPath = '../B_TBlanks/imgs/'; % Only used if a fermi fit is requested. And not eigencleaning.
    s.blanksImgMask = 'B_TBlanks_\d{4}'; % Regexp that matches only image file name without '.mat' 
   %% 
    % Two 2D gaussian fit (constrained by boxes)
    s.eval.two2Dfit.names = {'two2D_bg',...
        'two2D_a1', 'two2D_sh1', 'two2D_sv1',...% horiz and vert sigma
        'two2D_ch1', 'two2D_cv1',...% horiz and vert centre
        'two2D_a2', 'two2D_sh2', 'two2D_sv2',...
        'two2D_ch2', 'two2D_cv2',...
        'two2D_chd', 'two2D_cvd'};% Difference ch1-ch2 and cv1-cv2
    s.eval.two2Dfit.initialValues = [0  3 5 5 100 100  3 5 5 100 200];
    s.eval.two2Dfit.fix = []; % [[1,0]]?
    s.eval.two2Dfit.limits = [];
    s.eval.two2Dfit.constrainToBoxes = 1;% Constrain G1 to 5/2 box and G2 to 9/2 box independent of box autoadjust(overrides the center initial values)
    s.eval.two2Dfit.do = 0;
    %%
    % 3D Fermi Fit parameters and settings    
    s.eval.fermifit3d.eigenclean = s.eigenclean;
    s.eval.fermifit3d.outputNames = {'Tx','Ty','LiToTF','LiNfit','TFcalc','ToTFcalc','G_ctr_x','G_ctr_y','G_sigma_x','G_sigma_y','G_ar','G_Tx','G_Ty','G_ToTFcalc'};
    s.eval.fermifit3d.doFit = 1;
    %s.eval.fermifit3d.areaToFit = [1, 202, 6, 239];% Within the ROI. [sLow, sHigh, lLow, lHigh]
    s.eval.fermifit3d.areaToFit = [20 -20 50 -130]+[1, s.eval.ROI(2)-s.eval.ROI(1)+1, 1, s.eval.ROI(4)-s.eval.ROI(3)+1];
    %s.eval.fermifit3d.areaToFit = [10 -10 20 -20]+[1, s.eval.ROI(2)-s.eval.ROI(1)+1, 1, s.eval.ROI(4)-s.eval.ROI(3)+1];
    s.eval.fermifit3d.TOF = 6.6E-3;% TOF in milliseconds.
    s.eval.fermifit3d.polarization = 1;% 1 if spin polarized, 0.5 if two spin species?
    s.eval.fermifit3d.is2D = 0;% Fitting a 2D gas is still experimental.
    s.eval.fermifit3d.pixis = 'z_cam';% Either 'z_cam' for Pixis 2 (ie. axial imaging) or 'x_cam' for Pixis 1 (ie. radial imaging).
    % K ODT1/2=0.2/0.4 values
    
    % 2021-02-11
    s.eval.fermifit3d.trapFreqY = 384; %453; % 400;%441;      %401(5) - Trap frequency in Y direction in Hz not Rad/s.
    s.eval.fermifit3d.trapFreqX = 162; %169.1; % 148;%172.1;    %172.1(6) - Trap frequency in X direction in Hz not Rad/s.
    s.eval.fermifit3d.trapFreqZ = 450; %441;% 400;%457;      %457(1) - Trap frequency in Z direction in Hz not Rad/s
    
    % 2021-12-03
   % s.eval.fermifit3d.trapFreqY = 406*1.41; %473; %406; %(7) % 400;%441;      %401(5) - Trap frequency in Y direction in Hz not Rad/s.
   % s.eval.fermifit3d.trapFreqX = 161.5*1.22; %229.0; %161.5; %(0.5) % 148;%172.1;    %172.1(6) - Trap frequency in X direction in Hz not Rad/s.
    %s.eval.fermifit3d.trapFreqZ = 431*1.41; %(3)% 400;%457;      %457(1) - Trap frequency in Z direction in Hz not Rad/s
 %%    %%   
   %s.eval.fermifit3d.trapFreqY = 680;% Trap frequency in Y direction in Hz not Rad/s.
   % s.eval.fermifit3d.trapFreqX = 120;% Trap frequency in X direction in Hz not Rad/s.
   % s.eval.fermifit3d.trapFreqZ = 520;
%     % RB in chip trap only trap frequencies
%     s.eval.fermifit3d.trapFreqY = 362.5;% Trap frequency in Y direction in Hz not Rad/s. 731.713
%     s.eval.fermifit3d.trapFreqX = 10.254;% Trap frequency in X direction in Hz not Rad/s. 165.183
%     s.eval.fermifit3d.trapFreqZ = 364.07; 
    % K in chip trap only trap frequencies
%     s.eval.fermifit3d.trapFreqY = 543;% Trap frequency in Y direction in Hz not Rad/s. 731.713
%     s.eval.fermifit3d.trapFreqX = 10.68;% Trap frequency in X direction in Hz not Rad/s. 165.183
%     s.eval.fermifit3d.trapFreqZ = 545;
    
    
    % This is a bit of a cludgey way around the immutability of s.eval at
    % the moment. The s.skipped field should be loaded from the monitor
    % file without changes.
    s.skipped.autoSkippedImages = [];
    s.skipped.fermifit3d.fitFailedImages = [];
    
    s.physicalConstants.hbar = 1.054571800E-34;% Reduced Planck's constant in J*s. Error is (13). From 2014 CODATA.
    s.physicalConstants.kB = 1.3806488E-23;% Boltzmann's constant in J/K. Error is (13). From 2014 CODATA.
    s.physicalConstants.K40mass = 39.96399848 * 1.660538921E-27;% Mass of 40K in kg. Errors are (21)u and (73)kg/u.
    
    % box-count parameters and settings
    
    % BZ zone boxes adjustment variables
    bzcx = 15;% Center of BZ0
    bzcy = 6;% Top of BZ boxes
    bzw = 2;% Width of BZ1,2,-1,2. BZ0 is 2*bzw pixels in size.
    bzh = 7;% Approx height of BZ boxes
    % SG boxes adjustment variables
    n5y = 25; %Top of 5/2 box
    n5x = 100; %Left edge of 5/2 box;
    n5w = 120; %Width of 5/2, 7/2, 9/2 box
    n5h = 2; %Height of 5/2 and 9/2 boxes
    n7h = 150; %Height of 7/2 box
    % Background box adjustment
    bgdx = 330;
    bgdy = 40;
    s.eval.box.boxes = [
        % BZ boxes
        [bzcx-bzw-bzw-bzw, bzcx-bzw-bzw-1, bzcy, bzh+bzcy]
        [bzcx-bzw-bzw, bzcx-bzw-1, bzcy, bzh+bzcy]
        [bzcx-bzw, bzcx+bzw-1, bzcy, bzh+bzcy]
        [bzcx+bzw, bzcx+bzw+bzw-1, bzcy, bzh+bzcy]
        [bzcx+bzw+bzw, bzcx+bzw+bzw+bzw-1, bzcy, bzh+bzcy]
        % background box
        [0+bgdx, 40+bgdx, 0+bgdy, 100+bgdy]
        % -5/2, -7/2, and -9/2 FB scan count boxes
        [n5x, n5x+n5w, n5y, n5y+n5h-1]
        [n5x, n5x+n5w, n5y+n5h, n5y+n5h+n7h-1]
        [n5x, n5x+n5w, n5y+n5h+n7h, n5y+n5h+n7h+n5h-1]
        ];
    
    % autoadjust == 0 means do not auto adjust N5 (N7) N9 boxes
    % autoadjust == 1 means use marginal fit to shift boxes
    % autoadjust == 2 means use two2Dfit to center boxes on clouds
    s.eval.box.autoadjust = 0;
    
    s.eval.box.numboxes = 9;
    s.eval.box.names = {'box1','box2','box3','box4','box5','box6','box7','box8','box9'};
    
    % Auto add names if doing different kinds of fits
    % The order of appending to s.out.addnames must be the same as the
    % order of appending to each p(j).res at the end of evaluation.
    if( s.eval.fermifit3d.doFit )
        s.out.addnames = [s.out.addnames s.eval.fermifit3d.outputNames];
    end
    if( s.eval.two2Dfit.do || s.eval.box.autoadjust == 2 )
        %If you change the Fermi fitter to do this as well, you must keep
        %the order in this preamble the same as the order in p(j).res
        s.out.addnames = [s.out.addnames s.eval.two2Dfit.names];
    end

end
%%
% Get going ...
disp(' '); disp('This is the chip experiment''s scan monitor!')

[s,p] = scanInitialize(s);
s = scanCheckStructure(s); % SO FAR ONLY A DUMMY!

if (s.eval.fermifit3d.doFit && ~s.eigenclean)
    disp('Create averaged blank images.')
    %   at the moment, always create new background image
    [s.eval.fermifit3d.avgBlanksAtomImage, s.eval.fermifit3d.avgBlanksRefImage] = ...
        createBlanksFermiFitImage(s.blanksPath, s.blanksImgMask, s.eval.ROI, s.eval.fermifit3d.areaToFit, s.out.talk,1);
    
    disp('Creation of averaged blanks images complete.')
end

%% EVALUATION *** EVALUATION *** EVALUATION *** EVALUATION
sigmares = 3*766.7e-9^2/2/pi;
pxArea = 2.7e-6*2.7e-6; % ROUGH ESTIMATE ONLY!
C = pxArea/sigmares; %used for ROIsum and 22d gaussian fit

% These checks could go into scanInitialize.m
if isempty(s.evaluated); clear('p'); end
if s.eval.reevaluate
    s.skipped.autoSkippedImages = [];
    s.skipped.fermifit3d.fitFailedImages = [];
end
skip = 0;
j = 1;
while j <= length(s.filelist)
    
    %         if any(s.evaluated == s.it(j))...
    %                 || any(s.skipped.autoSkippedImages == s.it(j))...
    %                 || any(s.skipped.fermifit3d.fitFailedImages == s.it(j))
    % At the moment, always check the bad images and failed fermi fits.
    if any(s.evaluated == s.it(j))
        if (skip == 0)
            fprintf('    Skipping (already evaluated): %g',s.it(j));
            skip = 1;
        else
            fprintf(', %g',s.it(j));
        end
        if s.out.gerty; gerty('dizzy'); end
        %            if s.out.gerty; gerty('ahm'); end
        %pause(2);
        j = j + 1;
        continue
    else
        if (skip == 1)
            fprintf('!\n');
            skip = 0;
        end
        if s.out.gerty; gerty(((-1)^j+1)/2+8); end
    end
    
    % initialize this image's data structure
    q = struct('imgfile', [s.path s.filelist(j).name], ...
        'camname', 'Pixis Chip Axial', ...
        'TOF', 6.6e-3, ...
        'talk', s.out.talk, ... %here you can turn talk off if you want to speed up evaluation
        'eigenclean', s.eigenclean, ...
        'ecbox',s.ecbox, ...
        'ROI', s.eval.ROI, ...
        'it', s.it(j), ...
        'res', s.scanlist(s.scanlist(:,1)==s.it(j),:), ...
        'spectator', 'GERTY');
    if isfield(s, 'imgsubpath')
        q.imgfile = [s.path s.imgsubpath s.filelist(j).name];
    end

    % load image using camera settings
    img = chipPixisImages(q);
    
    % cut down to ROI and create vertical / horizontal pixelsums
    roi = img.roi; 
    clear('img');
    roi.v = 1:size(roi.OD,2); 
    roi.h = 1:size(roi.OD,1);
    roi.vsum = sum(roi.OD,1); 
    roi.hsum = sum(roi.OD,2);
    
    % if requested, plot current image and pixelsums
    if s.out.talk
        figure(2); clf;
        subplot(3,3,[2 3 5 6]); 
        imagesc(roi.OD); title('OD and Marginals');
        subplot(3,3,[8 9]); 
        plot(roi.v,roi.vsum,'-k'); 
        xlim([1 max(roi.v)]);
        subplot(3,3,[1 4]); 
        plot(roi.hsum,-roi.h,'-k'); 
        ylim([-max(roi.h) 1]);
        title(num2str(s.it(j)))
        if s.eval.fermifit3d.doFit
            subplot(3,3,[2 3 5 6]);
            atf = s.eval.fermifit3d.areaToFit;
            rectangle('Position',[atf(3), atf(1), atf([4,2]) - atf([3,1])], ...
                'EdgeColor',[1 1 1]);
            clear('atf');
            pause(0.1);
        end
    end
    
    if s.eval.autoSkipBadImages
        % 1000 is an arbitrary large number
        if median(roi.at(:)) > median(roi.ref(:)) + 800 || median(roi.at(:)) < median(roi.ref(:)) - 800
            fprintf('Discarding bad image, index %g. (shutter failure)\n', s.it(j))
            fprintf('Median(roi.at(:)) = %g and median(roi.ref(:)) = %g\n', median(roi.at(:)), median(roi.ref(:)))
            % Record this image as skipped. If we are trying these
            % again then don't add a duplicate.
            if ~any(s.skipped.autoSkippedImages == s.it(j))
                s.skipped.autoSkippedImages = [s.skipped.autoSkippedImages, s.it(j)];
            end
            % Now to carefully remove evidence of this image.
            s.filelist(j) = [];
            s.par(j) = [];
            s.it(j) = [];
            % We must also reduce s.numImages by 1 for
            % scanProcessResults.m to be happy. We will also use this
            % in the loop that collects the results.
            s.numImages = s.numImages - 1;
            % Also indicate to the user that we are skipping this
            % image.
            if s.out.talk
                figure(2); subplot(3,3,[2 3 5 6]);
                title('(Auto skipped)');% Not appending to title since this should be the first thing.
            end
            continue% j remains the same but s.filelist etc. are now shifted.
        end
    end
    
    % here starts the individual evaluation
    
    % FITTING MULTIPLE GAUSSIANS
    q.img = roi.OD;
    
    % Initialize fitting of gaussians to marginal distbns
    q.initialValues = s.eval.fit.initial;
    q.fix           = s.eval.fit.fix;
    q.limits        = s.eval.fit.limits;
    q.sumdir        = s.eval.fit.sumdir;
    
    % If we are doing box autoadjust of N5 and N9 boxes
    if( s.eval.box.autoadjust == 1 )% If we are using marginal distbns
        s.eval.fit.do = 1;
        if( q.sumdir == 2 )% vertical shifting
            % Assume the centers of each cloud will be in the center of
            % each box.
            % Use first gaussian for upper box N5 and second for lower.
            % ASSUMES N5 N7 N9 are the 7, 8, 9 boxes in the list
            %s.eval.box.boxes(7, 3:4)
            %s.eval.box.boxes(9, 3:4)
            q.initialValues = [0 2 30 mean(s.eval.box.boxes(7, 3:4))...
                2 30 mean(s.eval.box.boxes(9, 3:4))];
            %
            q.fix = [];
            q.limits = [[4 7];
                [s.eval.box.boxes(7,3) s.eval.box.boxes(9,3)];
                [s.eval.box.boxes(7,4) s.eval.box.boxes(9,4)]];
            %q.initialValues
        elseif( q.sumdir == 1 )% horizontal shifting
            % TBD
            error('  Horizontal only (marginal) box shift not implemented yet')
        else
            error('  Incorrect sum direction');
        end
    elseif( s.eval.box.autoadjust == 2 )% If we are using two2Dfit
        s.eval.two2Dfit.do = 1;
        q.two2Dfit.fix = [];
        %s.eval.box.boxes(7, 1:4)
        %s.eval.box.boxes(9, 1:4)
        % ASSUMES N5 N7 N9 are the 7, 8, 9 boxes in the list and G1 is
        % for N5 and G2 is for N9
        q.two2Dfit.initialValues = [0  0.4 30 10 ...% bg, a1, sh1, sv1
            mean(s.eval.box.boxes(7, 1:2))...% ch1
            mean(s.eval.box.boxes(7, 3:4))...% cv1
            0.4 30 10 ...% a2, sh2, sv2
            mean(s.eval.box.boxes(9, 1:2))...% ch2
            mean(s.eval.box.boxes(9, 3:4))...% cv2
            ];
        %q.two2Dfit.initialValues
        q.two2Dfit.limits = [[5 6  10 11];% ch1 cv1 ch2 cv2
            [s.eval.box.boxes(7,1) s.eval.box.boxes(7,3) s.eval.box.boxes(9,1) s.eval.box.boxes(9,3)];
            [s.eval.box.boxes(7,2) s.eval.box.boxes(7,4) s.eval.box.boxes(9,2) s.eval.box.boxes(9,4)]
            ];
        %q.two2Dfit.limits
    end
    
    if ( s.eval.fit.do )
        q = fitTwoGaussians(q,s.out.talk);
    else
        q.fit = q.initialValues;
    end
    
    q.NLfit = C*abs(q.fit(2)*q.fit(3)*sqrt(pi/2));
    q.NRfit = C*abs(q.fit(5)*q.fit(6)*sqrt(pi/2));
    q.Nfit = q.NLfit + q.NRfit;
     
    if( s.eval.two2Dfit.do )
        q = two2DGaussFitFn(q, s.out.talk);
        %q.two2Dfit.fitResult
    end
    
    %%%%%%%%%%%%%
    
    q.ROIsum = C*sum(sum(roi.OD));
    q.ROIODmean = mean(roi.OD, "all");
    roi.refminusbg = roi.ref - roi.bg;
    q.ROIrefmean = mean(roi.refminusbg, "all");
    
    if s.eval.fermifit3d.doFit
        fprintf('Start fermi fitting for image %g\n', s.it(j))
        
        q.TOF = [];% Don't need to, but insure the TOF value from the camera settings file is not used.
        
        atf = s.eval.fermifit3d.areaToFit;
        q.fermifit3d.fitOutput = ...
            fermi_fit_scan_monitor(...
            roi.at(atf(1):atf(2), atf(3):atf(4)),...
            roi.ref(atf(1):atf(2), atf(3):atf(4)),...
            roi.bg(atf(1):atf(2), atf(3):atf(4)),...
            s.eval.fermifit3d, s.out.talk, s.physicalConstants);
        clear('atf');
        
        if ~q.fermifit3d.fitOutput.fitSuccess
            % The Fermi fit did not succeed, so for now we kill this
            % image. First record that fitting this image failed. If we
            % are trying these again then don't add a duplicate.
            if ~any(s.skipped.fermifit3d.fitFailedImages == s.it(j))
                s.skipped.fermifit3d.fitFailedImages = [s.skipped.fermifit3d.fitFailedImages s.it(j)];
            end
            % And then remove evidence of this image.
            s.filelist(j) = [];
            s.par(j) = [];
            s.it(j) = [];
            s.numImages = s.numImages - 1;
            % Finally, use the figure to tell the user that this fit
            % failed.
            if s.out.talk
                figure(2); subplot(3,3,[2 3 5 6]);
                title([get(get(gca,'Title'),'String') '(Fermi fit failed)']);
            end
            continue% skip all other evaluation (harsh, but it works)
        end
        
        disp('Finished fermi fitting.')
    end
    
    % ADAPTIVE SHIFTING OF BOXES
    if( s.eval.box.autoadjust == 1 )% marginal fit to one direction
        if( q.sumdir == 2 )% vertical
            q.ctrs = q.fit([4 7]);
            q.amps = q.fit([2 5]);
            %q.ctrs
            %q.amps
            [~, idx] = sort(q.amps);
            pos = q.ctrs(idx(end));
            
            boxshift = 0;
            for i = 1:size(s.eval.box.boxes,1)
                %s.eval.box.boxes(i,:)
                if ( (pos>s.eval.box.boxes(i,3)) && (pos<s.eval.box.boxes(i,4)) )
                    boxshift = round(pos-mean(s.eval.box.boxes(i,3:4)));
                end
            end
            %boxshift
            
            %                 % Hacky fix
            %                 %   Only fixes c9 box (9th one)
            %                 %   Will do something unknown if box is entirely outside of ROI
            %                 if boxshift + s.eval.box.boxes(9,4) >= s.eval.ROI(2) - s.eval.ROI(1)
            %                     s.eval.box.boxes(9,4) = s.eval.ROI(2) - s.eval.ROI(1);
            %                 end
            
            q.boxes = s.eval.box.boxes;
            % Only shift the N5 N7 N9 boxes vertically
            q.boxes(7:9,3:4) = q.boxes(7:9,3:4) + boxshift;
        elseif( q.sumdir == 1 )% horizontal
            q.ctrs = q.fit([4 7]);
            q.amps = q.fit([2 5]);
            %q.ctrs
            %q.amps
            [~, idx] = sort(q.amps);
            pos = q.ctrs(idx(end));
            
            boxshift = 0;
            for i = 1:size(s.eval.box.boxes,1)
                %s.eval.box.boxes(i,:)
                if ( (pos>s.eval.box.boxes(i,1)) && (pos<s.eval.box.boxes(i,2)) )
                    boxshift = round(pos-mean(s.eval.box.boxes(i,1:2)));
                end
            end
            %boxshift
            
            q.boxes = s.eval.box.boxes;
            % Only shift the N5 N7 N9 boxes horizontally
            q.boxes(7:9,1:2) = q.boxes(7:9,1:2) + boxshift;
        end
    elseif( s.eval.box.autoadjust == 2 )
        
        if( q.two2Dfit.fitResult.a1 > q.two2Dfit.fitResult.a2 )
            % calc box shifts based on n5
            bsh = round(q.two2Dfit.fitResult.ch1 - mean(s.eval.box.boxes(7,1:2)));
            bsv = round(q.two2Dfit.fitResult.cv1 - mean(s.eval.box.boxes(7,3:4)));
        else
            % calc box shifts based on n9
            bsh = round(q.two2Dfit.fitResult.ch2 - mean(s.eval.box.boxes(9,1:2)));
            bsv = round(q.two2Dfit.fitResult.cv2 - mean(s.eval.box.boxes(9,3:4)));
        end
        
        % bsh and bsv should be the amounts to shift the boxes
        % horizontally and vertically to center the boxes on the
        % clouds.
        q.boxes = s.eval.box.boxes;
        q.boxes(7:9,1:2) = q.boxes(7:9,1:2) + bsh;
        q.boxes(7:9,3:4) = q.boxes(7:9,3:4) + bsv;
        
        % Additionally shift boxes to the left to capture half the
        % cloud.
        % ASSUMES the N5 N7 N9 boxes are the same width.
        %halfWidth = round((s.eval.box.boxes(7,2) - s.eval.box.boxes(7,1))/2);
        %q.boxes(7:9,1:2) = q.boxes(7:9,1:2) - halfWidth;
        
    elseif( s.eval.box.autoadjust == 0 )
        q.boxes = s.eval.box.boxes;
    end

    % COUNTING POPULATION IN BOXES
    q.boxnames = s.eval.box.names;
    q = boxCount(q, s.out.talk, s.eval.box.autoadjust);
    % boxCount(q, s.out.talk) draws all boxes. To keep backwards
    % compatibility boxCount(q, s.out.talk, 1) will not draw any boxes
    % and so we must do that here.
    % ASSUMES N5 N7 N9 boxes are 7,8,9 in list of boxes.
    if s.out.talk && s.eval.box.autoadjust
        figure(2);
        subplot(3,3,[2 3 5 6]);
        % First draw old 7,8,9 boxes in gray
        for drawIndex = 7:9
            rectangle('Position', [s.eval.box.boxes(drawIndex,[1 3]) ...
                s.eval.box.boxes(drawIndex,[2 4]) - s.eval.box.boxes(drawIndex,[1 3])], ...
                'EdgeColor', [0.7 0.7 0.7]);
        end
        % Second, draw new 7,8,9 (and all other boxes) in white
        for drawIndex = 1:size(q.boxes, 1)
            rectangle('Position',[q.boxes(drawIndex,[1 3]) ...
                q.boxes(drawIndex,[2 4]) - q.boxes(drawIndex,[1 3])], ...
                'EdgeColor',[1 1 1]);
        end
    end
    
    q.density = C*q.density;
    q.cnt = C*q.cnt;
    
    if s.out.talk
        figure(2); % show images
        drawnow; pause(0.3);
    end
    
    p(j) = q;
    s.evaluated = [s.evaluated s.it(j)]; % mark image as evaluated
    
    j = j + 1;
end

%     if (skip == 1)
%         fprintf('!\n');
%         skip = 0;
%     end

% building results row -- now in a separate loop
for j = 1:length(s.filelist)
    NLL = (p(j).density(1)-p(j).density(6))*p(j).size(1);
    NL = (p(j).density(2)-p(j).density(6))*p(j).size(2);
    NC = (p(j).density(3)-p(j).density(6))*p(j).size(3);
    NR = (p(j).density(4)-p(j).density(6))*p(j).size(4);
    NRR = (p(j).density(5)-p(j).density(6))*p(j).size(5);
    
    N5 = (p(j).density(7)-p(j).density(6))*p(j).size(7);
    N7 = (p(j).density(8)-p(j).density(6))*p(j).size(8);
    N9 = (p(j).density(9)-p(j).density(6))*p(j).size(9);
    
    % Fermi fit results
    %p(j).fermifit3d.fitOutput.LiAreaFitParams
    if( s.eval.fermifit3d.doFit )
        T_x = p(j).fermifit3d.fitOutput.LiAreaFitParams.T_x/1E-9;% Horizontal temp in nK.
        T_y = p(j).fermifit3d.fitOutput.LiAreaFitParams.T_y/1E-9;% Vertical temp in nK.
        ToTf = p(j).fermifit3d.fitOutput.LiAreaFitParams.ToTf;% T/T_F from the fugacity.
        nfit = p(j).fermifit3d.fitOutput.LiAreaFitParams.nfit;% Number from the Li fit to the area image.
        % Calculate the Fermi temperature (in Kelvin) from the gaussian fit.
        % The 2*pi here accounts for the trap freqs being in Hz not rad/s.
        TFcalc = 2*pi*s.physicalConstants.hbar*(6*nfit)^(1/3)*(s.eval.fermifit3d.trapFreqX*s.eval.fermifit3d.trapFreqY*s.eval.fermifit3d.trapFreqZ)^(1/3)/s.physicalConstants.kB;
        ToTFcalc = sqrt(T_x*T_y)/(TFcalc/1E-9);% Li fit with trap freq. Fermi temp.
        G_ctr_x = p(j).fermifit3d.fitOutput.LiAreaFitParams.G_mean_x;
        G_ctr_y = p(j).fermifit3d.fitOutput.LiAreaFitParams.G_mean_y;
        G_sigma_x = p(j).fermifit3d.fitOutput.LiAreaFitParams.G_sigma_x;
        G_sigma_y = p(j).fermifit3d.fitOutput.LiAreaFitParams.G_sigma_y;
        G_ar = p(j).fermifit3d.fitOutput.LiAreaFitParams.G_sigma_x/p(j).fermifit3d.fitOutput.LiAreaFitParams.G_sigma_y;
        G_Tx = p(j).fermifit3d.fitOutput.LiAreaFitParams.G_Tx/1E-9;% nK
        G_Ty = p(j).fermifit3d.fitOutput.LiAreaFitParams.G_Ty/1E-9;% nK
        % Calculate T/TF from only gaussian fit parameters.
        G_ToTFcalc = sqrt(G_Tx*G_Ty)/(TFcalc/1E-9); % Gauss Tx Ty with Li TFcalc
        fermifit3dWriteArray = [T_x T_y ToTf nfit TFcalc ToTFcalc ...
                                G_ctr_x G_ctr_y G_sigma_x G_sigma_y G_ar...
                                G_Tx G_Ty G_ToTFcalc];
                            
    else
        fermifit3dWriteArray = [];
    end
    
    % Two 2D gaussians fit
    if( s.eval.two2Dfit.do || s.eval.box.autoadjust == 2 )
        two2DfitWriteArray = [p(j).two2Dfit.fitResult.bg...
            p(j).two2Dfit.fitResult.a1...
            p(j).two2Dfit.fitResult.sh1...
            p(j).two2Dfit.fitResult.sv1...
            p(j).two2Dfit.fitResult.ch1...
            p(j).two2Dfit.fitResult.cv1...
            p(j).two2Dfit.fitResult.a2...
            p(j).two2Dfit.fitResult.sh2...
            p(j).two2Dfit.fitResult.sv2...
            p(j).two2Dfit.fitResult.ch2...
            p(j).two2Dfit.fitResult.cv2...
            p(j).two2Dfit.fitResult.ch1 -  p(j).two2Dfit.fitResult.ch2...
            p(j).two2Dfit.fitResult.cv1 -  p(j).two2Dfit.fitResult.cv2...
            ];
    else
        two2DfitWriteArray = [];
    end
    %two2DfitWriteArray
    
    % At the moment, the order of this array must manually match the order of the names cell array in scanProcessResults.m
    % Standard output arrangement is to have the cycle number as the first column,
    % followed by the scan parameters, followed by the cycle number, and then
    % the ROI sum.
    fudgeFactor95 = 1.2;% N5 = fdgfct * N9  0.94
    p(j).res = [
        s.scanlist(s.scanlist(:,1)==s.it(j),:) s.it(j) p(j).ROIsum ...
        p(j).fit ...% Array containing the seven results of the two gaussian fits to either the vertical or horiz projection of the 2d image.
        p(j).cnt ...% Array containting the raw OD sum in each box.
        p(j).Nfit ...% Number of atoms from dual gaussian fit to vert/horiz projection.
        NLL NL NC NR NRR ...% Background box density corrected OD sum in each of the Brillioun zone boxes.
        (NLL+NL+NR+NRR)/(NLL+NL+NC+NR+NRR) ...
        N5 N7 N9 ...% Background box density corrected OD sum in each of the SSI boxes.
        N5/(fudgeFactor95*N9+N5) ...% Fraction of counts in the -5/2 box only considering the -9/2 and -5/2 boxes.
        N5/(1.75*N7+N5) ...
        1.75*N7/(1.75*N7+0.95*N9) ...
        N5+fudgeFactor95*N9 ...
        p(j).ROIODmean...
        p(j).ROIrefmean ...
        fermifit3dWriteArray...% Fermi fit results
        two2DfitWriteArray...% two2Dfit results
        ]; %consider the density (counts/(box size)) instead of the total counts
    
    % At the moment the following line sums 'fA1' and 'fWidth1' and puts it in the last column of the data file.
    % p(j).res = [p(j).res p(j).res(6)+p(j).res(7)];
    
    %p(j).res
end

%% THINGS THAT NEED TO BE DONE AT THE END

% Collecting evaluation results and averaging if requested
s = scanProcessResults(s, p);

% Specifically for fermi fit results, we now show the average of each
% group specified with s.out.groupby.
% s.res is a struct array with length equal to the number of groupby
% options
if s.eval.fermifit3d.doFit
    pm = char(177);% Plus over minus character.
    % Get indicies of things we want to take averages of
    TxIndex = scanGetNamePos('Tx', s.names);
    TyIndex = scanGetNamePos('Ty', s.names);
    NIndex = scanGetNamePos('LiNfit', s.names);
    LiToTFIndex = scanGetNamePos('LiToTF', s.names);
    GaToTFIndex = scanGetNamePos('ToTFcalc', s.names); %NOTE: This definition is weird!
    GToTFIndex = scanGetNamePos('G_ToTFcalc', s.names);
    for j = 1:length(s.res)
        if( isempty(s.out.groupby) )% If there is a nonempty groupby
            if (s.out.report_std)
                fprintf('\nThe Fermi fit averages with std deviation are:\n')
            else
                fprintf('\nThe Fermi fit averages with std uncert of mean are:\n')
            end
        else
            if (s.out.report_std) 
                fprintf('\nThe Fermi fit averages with std deviation for %s = %g are:\n', s.names{s.out.groupby}, s.res(j).values(1,s.out.groupby))
            else
                fprintf('\nThe Fermi fit averages with std uncert of mean for %s = %g are:\n', s.names{s.out.groupby}, s.res(j).values(1,s.out.groupby))
            end
        end
        
        if (s.out.report_std)
            % Averages with standard deviation (N-1)
            fprintf(['Tx: %.4g ' pm ' %.2g, '...
                'Ty: %.4g ' pm ' %.2g, '...
                'N: %.5g ' pm ' %.2g, '...
                'LiToTF: %.3g ' pm ' %.2g, '...
                'ToTFcalc: %.3g ' pm ' %.2g\n'...
                'G_ToTFcalc: %.3g ' pm ' %.2g\n'],...
                mean(s.res(j).values(:,TxIndex)), std(s.res(j).values(:,TxIndex)),...
                mean(s.res(j).values(:,TyIndex)), std(s.res(j).values(:,TyIndex)),...
                mean(s.res(j).values(:,NIndex)), std(s.res(j).values(:,NIndex)),...
                mean(s.res(j).values(:,LiToTFIndex)), std(s.res(j).values(:,LiToTFIndex)),...
                mean(s.res(j).values(:,GaToTFIndex)), std(s.res(j).values(:,GaToTFIndex)),...
                mean(s.res(j).values(:,GToTFIndex)), std(s.res(j).values(:,GToTFIndex))...
                )
        else 
            % Averages with standard uncertainty of the mean
            fprintf(['Tx: %.4g ' pm ' %.2g, '...
                'Ty: %.4g ' pm ' %.2g, '...
                'N: %.5g ' pm ' %.3g, '...
                'LiToTF: %.4g ' pm ' %.2g, '...
                'ToTFcalc: %.4g ' pm ' %.2g\n'],...
                mean(s.res(j).values(:,TxIndex)), std(s.res(j).values(:,TxIndex))/sqrt(length(s.res(j).values(:,TxIndex))),...
                mean(s.res(j).values(:,TyIndex)), std(s.res(j).values(:,TyIndex))/sqrt(length(s.res(j).values(:,TyIndex))),...
                mean(s.res(j).values(:,NIndex)), std(s.res(j).values(:,NIndex))/sqrt(length(s.res(j).values(:,NIndex))),...
                mean(s.res(j).values(:,LiToTFIndex)), std(s.res(j).values(:,LiToTFIndex))/sqrt(length(s.res(j).values(:,LiToTFIndex))),...
                mean(s.res(j).values(:,GaToTFIndex)), std(s.res(j).values(:,GaToTFIndex))/sqrt(length(s.res(j).values(:,GaToTFIndex)))...
                )
        end
    end
end

if( s.eval.box.autoadjust == 2 )% can easily rewrite this for mean of each group (just no point yet since can't adjust boxes by looking at scan parameters)
    pm = char(177);% Plus over minus character.
    dh = zeros(length(p),1);%[];
    dv = zeros(length(p),1);%[];
    for j = 1:length(p)
        if( p(j).two2Dfit.fitResult.a1 > 0.1 && p(j).two2Dfit.fitResult.a2 > 0.1 )
            dh(j) = (p(j).two2Dfit.fitResult.ch1 - p(j).two2Dfit.fitResult.ch2);
            dv(j) = (p(j).two2Dfit.fitResult.cv1 - p(j).two2Dfit.fitResult.cv2);
        end
    end
    %dh
    %dv
    if( isempty(dh) || isempty(dv) )
        fprintf('\nNo images with two sufficiently defined clouds to report means.\n')
    else
        fprintf('\nThe mean and median cloud separations (N5 - N9) are:\n')
        fprintf(['Horz mean: %g ' pm ' %g, and median: %g\n'], mean(dh), std(dh), median(dh))
        fprintf(['Vert mean: %g ' pm ' %g, and median: %g\n'], mean(dv), std(dv), median(dv))
    end
end

if ~isempty(s.out.groupby)
    s.lastGrpParam = p(end).res(scanGetNamePos(s.out.groupby,s.names));
end

% Plotting results
scanOutputResults(s);

%scanPlotStats(s);
% plot histograms

disp('Auto skipped images:');
%disp(s.skipped.autoSkippedImages);
fprintf('    ');
for j = 1:length(s.skipped.autoSkippedImages)
    fprintf('%g ', s.skipped.autoSkippedImages(j));
end
if s.eval.fermifit3d.doFit
    disp('Fermi fit failed images:');
    %disp(s.skipped.fermifit3d.fitFailedImages);
    fprintf('    ');
    for j = 1:length(s.skipped.fermifit3d.fitFailedImages)
        fprintf('%g ', s.skipped.fermifit3d.fitFailedImages(j));
    end
end

% Saving results and monitor parameters
scanSaveResults(s, p);

if ~isempty(s.out.niceimage)
    scanNiceImages(s,p,s.out.niceimage,0);
end

% Say goodbye ...
disp('  Scan evaluation successful!')
if s.out.gerty
    gerty('american');
    pause(0.2); gerty(round(16+rand(1)));
    pause(0.2+0.3*rand(1)); gerty('bunny');
end

%catch % error during evaluation makes gerty sad!
%    disp('  Error in scan evaluation!')
%   disp('  Possible compatibility issues? Type ''help scanListOfChanges'' to see ... .')
%   if s.out.gerty; gerty(ceil(2*rand+2)); end
%%    rethrow(lasterror);
out=42;
toc
disp('Meaning of Life:')
end
