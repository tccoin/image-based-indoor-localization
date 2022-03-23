%===========================================%
%   Splitting the Cambridge dataset
%   Method:     
%   Author: Jingwei 4 April 2020
%===========================================%
clc;clear all;close all;
%   Set parameter
dataset_dir = '\\palnas2\jsong\dataset_TUM\rgbd_dataset_freiburg2_large_with_loop';
filename = [dataset_dir  '\rgb.txt'];
posename = [dataset_dir  '\groundtruth.txt'];
num_testimage = 100;
% test_txt  = [dataset_dir  '\dataset_test.txt'];
index_train = 0;
index_test  = 1;

%   I Load training and testing folder index
[ folder_train,filename_train,trajectory_train ] = ...
                        data_folder_index( filename,posename );
folder_test = folder_train(end-num_testimage+1:end,:);folder_train(end-num_testimage+1:end,:)=[];
filename_test = filename_train(end-num_testimage+1:end,:);filename_train(end-num_testimage+1:end,:)=[];
trajectory_test = trajectory_train(end-num_testimage+1:end,:);trajectory_train(end-num_testimage+1:end,:)=[];
% [ folder_test, filename_test, trajectory_test  ] = ...
%                         data_folder_index( test_txt  );

%   II. Put training data in 00 and testing data in 01 folder
[status, message, messageid] = rmdir([dataset_dir '\sequences\'], 's');
mkdir([dataset_dir '\sequences\' num2str(index_train,'%02d')]);
mkdir([dataset_dir '\sequences\' num2str(index_test ,'%02d')]);
for i = 1 : size(filename_train,1)
    source = [dataset_dir '\' folder_train{i} '\' filename_train(i,:)];
    target = [dataset_dir '\sequences\' num2str(index_train,'%02d')  '\image_'  num2str(i-1,'%010d') '.png'];
    [status,message,messageId] = copyfile(source, target, 'f');
    if(status~=1)
        error('Wrong copying');
    end
end
for i = 1 : size(filename_test,1)
    source = [dataset_dir '\' folder_test{i} '\' filename_test(i,:)];
    target = [dataset_dir '\sequences\' num2str(index_test,'%02d')  '\image_'  num2str(i-1,'%010d') '.png'];
    [status,message,messageId] = copyfile(source, target, 'f');
    if(status~=1)
        error('Wrong copying');
    end
end


%   III. Put trajectory in the folder
filename = [dataset_dir '\' num2str(index_train,'%02d') '.txt'];
delete filename;
file = fopen (filename, 'wt');
for i = 1 : size(trajectory_train,1)
    linetxt{1} = ['image_'  num2str(i-1,'%010d') '.png'];
    linetxt{2} = num2str(trajectory_train(i,1));
    linetxt{3} = num2str(trajectory_train(i,2));
    linetxt{4} = num2str(trajectory_train(i,3));
    linetxt{5} = num2str(trajectory_train(i,4));
    linetxt{6} = num2str(trajectory_train(i,5));
    linetxt{7} = num2str(trajectory_train(i,6));
    linetxt{8} = num2str(trajectory_train(i,7));
    fprintf(file,'%s\n',strjoin(linetxt));
end
fclose(file); 
filename = [dataset_dir '\' num2str(index_test,'%02d') '.txt'];
delete filename;
file = fopen (filename, 'wt');
for i = 1 : size(trajectory_test,1)
    linetxt{1} = ['image_'  num2str(i-1,'%010d') '.png'];
    linetxt{2} = num2str(trajectory_test(i,1));
    linetxt{3} = num2str(trajectory_test(i,2));
    linetxt{4} = num2str(trajectory_test(i,3));
    linetxt{5} = num2str(trajectory_test(i,4));
    linetxt{6} = num2str(trajectory_test(i,5));
    linetxt{7} = num2str(trajectory_test(i,6));
    linetxt{8} = num2str(trajectory_test(i,7));
    fprintf(file,'%s\n',strjoin(linetxt));
end
fclose(file); 


function [ folder,filename,trajectory ] = data_folder_index( filename,posename )
%   I   Open all files
file1 = fopen (filename, 'rt');
folder      = {};
filename    = [];
timestamp_file = [];
while feof(file1) ~= 1

    linetxt = fgetl(file1);
    linetxt = strsplit(linetxt);
    if(size(linetxt,2) ~= 2)
        continue
    end
    tmp = strsplit(linetxt{2},'/');
    folder   = [folder  ; tmp{1}];
    filename = [filename; tmp{2}];
    timestamp_file = [timestamp_file; str2num(linetxt{1,1})];
end
fclose(file1);
file2 = fopen (posename, 'rt');
trajectory_tmp  = [];
timestamp_pose = [];
while feof(file2) ~= 1

    linetxt = fgetl(file2);
    linetxt = strsplit(linetxt);
    if(size(linetxt,2) ~= 8)
        continue
    end
    timestamp_pose = [timestamp_pose; str2num(linetxt{1,1})];
    position_tmp = [str2double(linetxt{2}) str2double(linetxt{3}) str2double(linetxt{4})...
                    str2double(linetxt{5}) str2double(linetxt{6}) str2double(linetxt{7}) str2double(linetxt{8})];
    trajectory_tmp = [trajectory_tmp;position_tmp];
end
fclose(file2);
% %   II.  Modify ind = ind + 10000*f(folder) 
% folder_namelist = unique(folder);
% for i = 1 : size(fileind,1)
%     ind_name = 0;
%     for j = 1 : size(folder_namelist,1)
%         if(strcmp(folder_namelist{j},folder{i}))
%             ind_name = j;
%             break;
%         end
%     end
%     if(ind_name == 0)
%         error('Name is wrong');
%     end
%     fileind(i) = fileind(i) + 10000*ind_name;
% end
% 
% %   III   Sort the index and permutate the files
% [B,I] = sort(fileind);
% folder = folder(I,:);
% filename = filename(I,:);
% trajectory = trajectory(I,:);
%   II.   Synchronize the time stamp
trajectory = [];
for i = 1 : size(timestamp_file,1)
    tmp_min = inf;
    ind = 1;
    for j = 1 : size(timestamp_pose,1)
        timelag = abs(timestamp_file(i)-timestamp_pose(j));
        if(timelag < tmp_min)
            tmp_min = timelag;
            ind = j;
        end
    end
    trajectory = [trajectory;trajectory_tmp(ind,:)];
end
end