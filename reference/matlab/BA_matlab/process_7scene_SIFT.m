function [ observe_ith,param] = process_7scene_SIFT(history,cameraParams,observe_ith,filepath,filepath_history,num_image,gap,locID,locID_init,param_keyframe, i_idx)
%GETIMAGE Summary of this function goes here


%   Select images for triangulation
ind_neighbor = selectimage(history.orientation,history.robotpose,locID_init,param_keyframe.bm,param_keyframe.sigma,param_keyframe.alpha_m,num_image,param_keyframe.max_range);


num_image_left = (num_image - 1) / 2;
nImgLst = cell(2*num_image_left+2,1);
nImgIndList = zeros(2*num_image_left+2,1);
nImgLst{1} = [filepath num2str(locID-1,'%010.0f') '.png'];
nImgIndList(1) = locID;

for i = 1 : size(ind_neighbor,1)
    nImgLst{i+1} = [filepath_history num2str(ind_neighbor(i)-1,'%010.0f') '.png'];
    nImgIndList(i+1) = ind_neighbor(i);
end

num_image = size(nImgLst,1) - 1;
IPrev = rgb2gray(imread(nImgLst{1}));
IPrev = undistortImage(IPrev, cameraParams);
%[featuresPrev,pointsPrev] = extractSIFTFeature(IPrev);
pointsPrev = detectORBFeatures(IPrev);
[featuresPrev,pointsPrev] = extractFeatures(IPrev,pointsPrev);

% pointsPrev = detectSURFFeatures(IPrev, 'MetricThreshold', 500,'NumScaleLevels',6,'NumOctaves',4);
Orient = observe_ith.orientation;
robotpose = observe_ith.robotpose;
viewID = [];
vSet = imageviewset;
vSet = addView(vSet, 1, rigid3d(Orient',robotpose),'Points',pointsPrev, 'Features', featuresPrev, 'Points', pointsPrev);
for viewId = 2:num_image+1
    if(~exist(nImgLst{viewId},'file'))
        continue;
    end
    I = imread(nImgLst{viewId});
    I = rgb2gray(I);
    I = undistortImage(I, cameraParams);

    %[features,points] = extractSIFTFeature(I);
    points = detectORBFeatures(I);
    [features,points] = extractFeatures(I,points);
    Orient = history.orientation(3*nImgIndList(viewId)-2:3*nImgIndList(viewId),:);
    robotpose = history.robotpose(nImgIndList(viewId),:);

    pairsIdx = matchFeatures(featuresPrev,features,'Method','Approximate','MaxRatio',0.8,'Unique',true);
    %[pairsIdx, scores] = vl_ubcmatch(featuresPrev', features');
    %pairsIdx = pairsIdx';
    
    %   Add RANSAC
    matchedPoints1 = pointsPrev(pairsIdx(:,1),:);
    matchedPoints2 = points(pairsIdx(:,2),:);
    if(size(matchedPoints1,1) > 10)
        vSet = addView(vSet,viewId, rigid3d(Orient',robotpose),'Points',points, 'Features', features);
        vSet = addConnection(vSet,1,viewId,'Matches',pairsIdx);
        viewID = [viewID;viewId];
    end  
end

tracks = findTracks(vSet);
cameraPoses = poses(vSet);



  [xyzPoints, reprojectionErrors, validIndex] = triangulateMultiview(tracks,cameraPoses,cameraParams);
  idx = reprojectionErrors < 30 & validIndex & xyzPoints(:,3)<4;


multiview_idx = [];
for i=1:size(tracks,2)
    if(size(tracks(1,i).ViewIds,2)>3)
        multiview_idx = [multiview_idx i];
    end
end

  %reprojectionErrors(multiview_idx)
  disp('point number: ')
  size(xyzPoints(idx, :), 1)
  pcshow(xyzPoints(idx, :), 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 100);
  pcshow(xyzPoints(idx, :), 'MarkerSize', 100);
  hold on
  plotCamera(cameraPoses, 'Size', 0.05);
  hold off

%   Filter outlier
xyzPoints = xyzPoints(idx,:);

% disp('############');
% disp(xyzPoints);
% disp('############');
filename = ['data/neighbor_id/' int2str(i_idx) '.csv'];
writematrix(ind_neighbor, filename);

filename = ['data/feature_matches_3d/' int2str(i_idx) '.csv'];
writematrix(xyzPoints, filename);
tracks1 = tracks(:,idx);

N = size(tracks1,2);
Views = ones(N, num_image+1) * -1;
Points = ones(N, (num_image+1)*2) * -1;
for i =1 : N
    M = size(tracks1(1,i).ViewIds,2);
    Views(i, 1:M) = tracks1(1,i).ViewIds(1, :);
    Points(i, 1:2*M) = reshape(tracks1(1,i).Points', [1, 2*M]);
end
filename = ['data/viewIds_matches/' int2str(i_idx) '.csv'];
writematrix(Views, filename);
filename = ['data/points_matches/' int2str(i_idx) '.csv'];
writematrix(Points, filename);


observe_ith.pts3D = xyzPoints;
observe_ith.pts2D = zeros(size(xyzPoints,1),2);
param.K = cameraParams.IntrinsicMatrix';
param.num_features = size(xyzPoints,1);
for i =1 : size(tracks1,2)
    observe_ith.pts2D(i,:) = [tracks1(1,i).Points(1,2) tracks1(1,i).Points(1,1)];
end
filename = ['data/feature_matches_2d/' int2str(i_idx) '.csv'];
writematrix(observe_ith.pts2D, filename);

end

