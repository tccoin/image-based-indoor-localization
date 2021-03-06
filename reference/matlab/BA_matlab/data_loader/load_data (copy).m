function [pose_ID_groudtruth,pose_ID_predict,groundtruth,history,cameraParams,filepath,filepath_history] = ...
              load_data(dataset_name,subdataset)
%===========================================%
%   Load data set
%   Input:      dataset_name:       Name of the data set (TUM,7scenes,nvidia)
%               subdataset:         data set subname 
%   Output:     pose_ID_groudtruth: Ground truth pose ID
%               pose_ID_predict:    Prediction image ID
%               groundtruth:        Groud truth poses, containing robotpose
%                                   and orientation (3*3)
%               history:        	history poses, containing robotpose
%                                   and orientation (3*3)
%               cameraParams:       Camera parameter
%===========================================%



switch dataset_name
    case 'TUM'
        disp(['Loading TUM data set ' subdataset]);
       %%
       driverpath = '/home/link/Projects/image-based-indoor-localization/data/ethl1_local';
       TUM_camera_ID = str2double(subdataset(1));
       switch TUM_camera_ID
           case 1
               load('TUM/cameraParams_RGB_TUM_1.mat');
           case 2
               load('TUM/cameraParams_RGB_TUM_2.mat');
           otherwise
               error('TUM data set error')
       end
       filepath = [driverpath '/sequences/01/image_'];
       posepath = [driverpath '/01.txt'];
       filepath_history = [driverpath '/sequences/00/image_'];
       posepath_history = [driverpath '/00.txt'];
       
       FILENAME_BASE = [driverpath '/ethl1_local_train__siamese_FXPAL_output'];
        
        
        for i = 1 : 2
            filename = [FILENAME_BASE '_' int2str(i-1) '.h5py'];
            pose_ID_groudtruth{i} = h5read(filename,'/posenet_x_label');
            pose_ID_predict{i} = h5read(filename,'/posenet_x_predicted');
        end
       
       fid = fopen(posepath);
       groundtruth.robotpose = zeros(1000,3);
       groundtruth.orientation = zeros(3*1000,3);
       i = 1;
       while 1
           tline = fgetl(fid);
           if ~ischar(tline),   break,   end  % exit at end of file
           ln = sscanf(tline,'%s',1); % line type
           if(isempty(ln))
               break;
           end
           
           mtl_name = split(tline);
           groundtruth.robotpose(i,:) = [str2double(mtl_name{2,1}) str2double(mtl_name{3,1}) str2double(mtl_name{4,1})];
           tmp = [str2double(mtl_name{8,1}) str2double(mtl_name{5,1}) str2double(mtl_name{6,1}) str2double(mtl_name{7,1})  ];
           groundtruth.orientation(3*i-2:3*i,:)=quat2rotm(tmp);
           i = i + 1;
           
       end
       fclose(fid);
       
       
       
       
       fid = fopen(posepath_history);
       history.robotpose = zeros(1000,3);
       history.orientation = zeros(3*1000,3);
       i = 1;
       while 1
           tline = fgetl(fid);
           if ~ischar(tline),   break,   end  % exit at end of file
           ln = sscanf(tline,'%s',1); % line type
           if(isempty(ln))
               break;
           end
           
           mtl_name = split(tline);
           history.robotpose(i,:) = [str2double(mtl_name{2,1}) str2double(mtl_name{3,1}) str2double(mtl_name{4,1})];
           tmp = [str2double(mtl_name{8,1}) str2double(mtl_name{5,1}) str2double(mtl_name{6,1}) str2double(mtl_name{7,1}) ];
           history.orientation(3*i-2:3*i,:)=quat2rotm(tmp);
           
           i = i + 1;
       end
       fclose(fid);

    case '7scenes'
        %%
        driverpath = '/home/link/Projects/image-based-indoor-localization/data/';
        disp(['Loading 7scenes data set ' subdataset]);
        load('7scenes/cameraParams_kinect_7scenes.mat');
        filepath = [driverpath subdataset '\sequences\test\image_'];
        posepath = [driverpath subdataset '\test.txt'];
        filepath_history = [driverpath subdataset '\sequences\train\image_'];
        posepath_history = [driverpath subdataset '\train.txt'];
        
        
        % 0 for train
%         filename = [FILENAME_BASE '_train.h5py'];
%         pose_ID_groudtruth{0} = h5read(filename,'/posenet_x_label');
%         pose_ID_predict{0} = h5read(filename,'/posenet_x_predicted');
        filename = '/home/link/Projects/image-based-indoor-localization/model/chess/cdf/chess_output.h5py';
        pose_ID_groudtruth{1} = h5read(filename,'/posenet_x_label');
        pose_ID_predict{1} = h5read(filename,'/posenet_x_predicted');
        
        
        
        fid = fopen(posepath);
        groundtruth.robotpose = zeros(1000,3);
        groundtruth.orientation = zeros(3*1000,3);
        i = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline),   break,   end  % exit at end of file
            ln = sscanf(tline,'%s',1); % line type
            if(isempty(ln))
                break;
            end
            
            mtl_name = split(tline);
            groundtruth.robotpose(i,:) = [str2double(mtl_name{2,1}) str2double(mtl_name{3,1}) str2double(mtl_name{4,1})];
            orientation = [str2double(mtl_name{5,1}) str2double(mtl_name{6,1}) str2double(mtl_name{7,1});
                str2double(mtl_name{8,1}) str2double(mtl_name{9,1}) str2double(mtl_name{10,1});
                str2double(mtl_name{11,1}) str2double(mtl_name{12,1}) str2double(mtl_name{13,1});];
            groundtruth.orientation(3*i-2:3*i,:) = orientation;
            i = i + 1;
            
        end
        fclose(fid);
        
        
        
        
        fid = fopen(posepath_history);
        % parse .obj file
        history.robotpose = zeros(1000,3);
        history.orientation = zeros(3*1000,3);
        i = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline),   break,   end  % exit at end of file
            ln = sscanf(tline,'%s',1); % line type
            if(isempty(ln))
                break;
            end
            
            mtl_name = split(tline);
            history.robotpose(i,:) = [str2double(mtl_name{2,1}) str2double(mtl_name{3,1}) str2double(mtl_name{4,1})];
            orientation = [str2double(mtl_name{5,1}) str2double(mtl_name{6,1}) str2double(mtl_name{7,1});
                str2double(mtl_name{8,1}) str2double(mtl_name{9,1}) str2double(mtl_name{10,1});
                str2double(mtl_name{11,1}) str2double(mtl_name{12,1}) str2double(mtl_name{13,1});];
            history.orientation(3*i-2:3*i,:) = orientation;
           
            i = i + 1;
        end
        fclose(fid);

    case 'nvidia'
        addpath(genpath('npy-matlab'));
        disp(['Loading nvidia data set ' subdataset]);
        driverpath = 'F:\dataset\NVIDIA\indoor_dataset\';
        filepath = [driverpath subdataset '\sequences\01\image_'];
        posepath = [driverpath subdataset '\01.txt'];
        filepath_history = [driverpath subdataset '\sequences\00\image_'];
        posepath_history = [driverpath subdataset '\00.txt'];
        
        
        K = readNPY([driverpath subdataset '\02_long_K.npy']);
        tmp=K(1,3);K(1,3)=K(2,3);K(2,3)=tmp;
        cameraParams = cameraParameters('IntrinsicMatrix',K'); 
        
        FILENAME_BASE = [driverpath subdataset '\posenet_training_output\cdf\chess_train__siamese_FXPAL_output'];
        
        for i = 1 : 2
            filename = [FILENAME_BASE '_' int2str(i-1) '.h5py'];
            pose_ID_groudtruth{i} = h5read(filename,'/posenet_x_label');
            pose_ID_predict{i} = h5read(filename,'/posenet_x_predicted');
        end
        
        fid = fopen(posepath);
        groundtruth.robotpose = zeros(1000,3);
        groundtruth.orientation = zeros(3*1000,3);
        i = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline),   break,   end  % exit at end of file
            ln = sscanf(tline,'%s',1); % line type
            if(isempty(ln))
                break;
            end
            
            mtl_name = split(tline);
            groundtruth.robotpose(i,:) = [str2double(mtl_name{2,1}) str2double(mtl_name{3,1}) str2double(mtl_name{4,1})];
            tmp = [str2double(mtl_name{8,1}) -str2double(mtl_name{5,1}) str2double(mtl_name{6,1}) str2double(mtl_name{7,1})  ];
            groundtruth.orientation(3*i-2:3*i,:)=quat2rotm(tmp)';
            i = i + 1;
        end
        fclose(fid);
        
        
        
        
        fid = fopen(posepath_history);
        % parse .obj file
        history.robotpose = zeros(1000,3);
        history.orientation = zeros(3*1000,3);
        i = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline),   break,   end  % exit at end of file
            ln = sscanf(tline,'%s',1); % line type
            if(isempty(ln))
                break;
            end
            
            mtl_name = split(tline);
            history.robotpose(i,:) = [str2double(mtl_name{2,1}) str2double(mtl_name{3,1}) str2double(mtl_name{4,1})];
            tmp = [str2double(mtl_name{8,1}) -str2double(mtl_name{5,1}) str2double(mtl_name{6,1}) str2double(mtl_name{7,1}) ];
            history.orientation(3*i-2:3*i,:)=quat2rotm(tmp)';
            
            i = i + 1;
        end
        fclose(fid);
    otherwise
        error('Dataset not found');
end


end

