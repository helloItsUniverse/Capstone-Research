%% Download Lidar Dataset

doTraining = false;

outputFolder = fullfile('C:\UNIVERSE_GP76\HGU\2022\CAPSTONE','Pandaset');

lidarURL = ['https://ssd.mathworks.com/supportfiles/lidar/data/' ...
    'Pandaset_LidarData.tar.gz'];
helperDownloadPandasetData(outputFolder,lidarURL);

% Depending on your Internet connection, the download process can take some time.


%% Load Data

% Create a file datastore to load the PCD files from the specified path using the pcread function.
path = fullfile(outputFolder,'Lidar');
lidarData = fileDatastore(path,'ReadFcn',@(x) pcread(x));

% Load the 3-D bounding box labels of the car and truck objects.
gtPath = fullfile(outputFolder,'Cuboids','PandaSetLidarGroundTruth.mat');
data = load(gtPath,'lidarGtLabels');
Labels = timetable2table(data.lidarGtLabels);
boxLabels = Labels(:,2:3);

% Display the full-view point cloud.
figure
ptCld = read(lidarData);
ax = pcshow(ptCld.Location);
set(ax,'XLim',[-50 50],'YLim',[-40 40]);
zoom(ax,2.5);
axis off;

% Reset lidar data
reset(lidarData);


%% Preprocess Data

xMin = 0.0;     % Minimum value along X-axis.
yMin = -39.68;  % Minimum value along Y-axis.
zMin = -5.0;    % Minimum value along Z-axis.
xMax = 69.12;   % Maximum value along X-axis.
yMax = 39.68;   % Maximum value along Y-axis.
zMax = 5.0;     % Maximum value along Z-axis.
xStep = 0.16;   % Resolution along X-axis.
yStep = 0.16;   % Resolution along Y-axis.
dsFactor = 2.0; % Downsampling factor.

% Calculate the dimensions for the pseudo-image.
Xn = round(((xMax - xMin) / xStep));
Yn = round(((yMax - yMin) / yStep));

% Define point cloud parameters.
pointCloudRange = [xMin,xMax,yMin,yMax,zMin,zMax];
voxelSize = [xStep,yStep];

% Crop the front view from the input full-view point cloud.
% Select the box labels that are inside the ROI specified by gridParams.
[croppedPointCloudObj,processedLabels] = cropFrontViewFromLidarData(...
    lidarData,boxLabels,pointCloudRange);

% Display the cropped point cloud and the ground truth box labels 
% using the helperDisplay3DBoxesOverlaidPointCloud helper function 
% defined at the end of the example.
pc = croppedPointCloudObj{1,1};
gtLabelsCar = processedLabels.Car{1};
gtLabelsTruck = processedLabels.Truck{1};

helperDisplay3DBoxesOverlaidPointCloud(pc.Location,gtLabelsCar,...
   'green',gtLabelsTruck,'magenta','Cropped Point Cloud');

% Reset lidar data
reset(lidarData);


%% Create Datastore Objects for Training

% Split the data set into training and test sets. Select 70% of the data for training the network and the rest for evaluation.
rng(1);
shuffledIndices = randperm(size(processedLabels,1));
idx = floor(0.7 * length(shuffledIndices));

trainData = croppedPointCloudObj(shuffledIndices(1:idx),:);
testData = croppedPointCloudObj(shuffledIndices(idx+1:end),:);

trainLabels = processedLabels(shuffledIndices(1:idx),:);
testLabels = processedLabels(shuffledIndices(idx+1:end),:);

writeFiles = true;
dataLocation = fullfile(outputFolder,'InputData');
[trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
    dataLocation,writeFiles);

% Create a file datastore using fileDatastore to load PCD files using the pcread function.
lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));

% Create a box label datastore using boxLabelDatastore for loading the 3-D bounding box labels.
bds = boxLabelDatastore(trainLabels);

% Use the combine function to combine the point clouds and 3-D bounding box labels into a single datastore for training.
cds = combine(lds,bds);


%% Data Augmentation

% Read and display a point cloud before augmentation
% using the helperDisplay3DBoxesOverlaidPointCloud helper function,
% defined at the end of the example..
augData = read(cds);
augptCld = augData{1,1};
augLabels = augData{1,2};
augClass = augData{1,3};

labelsCar = augLabels(augClass=='Car',:);
labelsTruck = augLabels(augClass=='Truck',:);

helperDisplay3DBoxesOverlaidPointCloud(augptCld.Location,labelsCar,'green',...
    labelsTruck,'magenta','Before Data Augmentation');

% Reset cds
reset(cds);


% Use the sampleGroundTruthObjectsFromLidarData helper function, 
% attached to this example as a supporting file, 
% to extract all the ground truth bounding boxes from the training data.
classNames = {'Car','Truck'};
sampleLocation = fullfile(tempdir,'GTsamples');
[sampledGTData,indices] = sampleGroundTruthObjectsFromLidarData(cds,classNames,...
    'MinPoints',20,'sampleLocation',sampleLocation);

% Use the augmentGroundTruthObjectsToLidarData helper function, 
% attached to this example as a supporting file, 
% to randomly add a fixed number of car and truck class objects to every point cloud. 
% Use the transform function to apply the ground truth and custom data augmentations to the training data.
numObjects = [10,10];
cdsAugmented = transform(cds,@(x) augmentGroundTruthObjectsToLidarData(x,...
    sampledGTData,indices,classNames,numObjects));


% In addition, apply the following data augmentations to every point cloud.
% - Random flipping along the x-axis
% - Random scaling by 5 percent
% - Random rotation along the z-axis from [-pi/4, pi/4]
% - Random translation by [0.2, 0.2, 0.1] meters along the x-, y-, and z-axis respectively
cdsAugmented = transform(cdsAugmented,@(x) augmentData(x));

% Display an augmented point cloud along with ground truth augmented boxes 
% using the helperDisplay3DBoxesOverlaidPointCloud helper function, defined at the end of the example.
augData = read(cdsAugmented);
augptCld = augData{1,1};
augLabels = augData{1,2};
augClass = augData{1,3};

labelsCar = augLabels(augClass=='Car',:);
labelsTruck = augLabels(augClass=='Truck',:);

helperDisplay3DBoxesOverlaidPointCloud(augptCld.Location,labelsCar,'green',...
    labelsTruck,'magenta','After Data Augmentation');

% Reset cdsAugmented
reset(cdsAugmented);



%% Create PointPillars Object Detector

% Define number of prominent pillars.
P = 12000; 

% Define number of points per pillar.
N = 100;   

% Estimate anchor boxes from training data.
anchorBoxes = calculateAnchorsPointPillars(trainLabels);
classNames = trainLabels.Properties.VariableNames;

% Define the PointPillars detector.
detector = pointPillarsObjectDetector(pointCloudRange,classNames,anchorBoxes,...
    'VoxelSize',voxelSize,'NumPillars',P,'NumPointsPerPillar',N);



%% Train Pointpillars Object Detector

executionEnvironment = "auto";
if canUseParallelPool
    dispatchInBackground = true;
else
    dispatchInBackground = false;
end

options = trainingOptions('adam',...
    'Plots',"none",...
    'MaxEpochs',60,...
    'MiniBatchSize',3,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'LearnRateSchedule',"piecewise",...
    'InitialLearnRate',0.0002,...
    'LearnRateDropPeriod',15,...
    'LearnRateDropFactor',0.8,...
    'ExecutionEnvironment',executionEnvironment,...
    'DispatchInBackground',dispatchInBackground,...
    'BatchNormalizationStatistics','moving',...
    'ResetInputNormalization',false,...
    'CheckpointPath',tempdir);

% Use trainPointPillarsObjectDetector function to train the PointPillars object detector if doTraining is true. 
% Otherwise, load the pretrained detector.
if doTraining    
    [detector,info] = trainPointPillarsObjectDetector(cdsAugmented,detector,options);
else
    pretrainedDetector = load('pretrainedPointPillarsDetector.mat','detector');
    detector = pretrainedDetector.detector;
end



%% Generate Detections

% Use the trained network to detect objects in the test data:
% - Read the point cloud from the test data.
% - Run the detector on the test point cloud to get the predicted bounding boxes and confidence scores.
% - Display the point cloud with bounding boxes using the helperDisplay3DBoxesOverlaidPointCloud helper function, defined at the end of the example.

ptCloud = testData{45,1};
gtLabels = testLabels(45,:);

% Specify the confidence threshold to use only detections with
% confidence scores above this value.
confidenceThreshold = 0.5;
[box,score,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);

boxlabelsCar = box(labels'=='Car',:);
boxlabelsTruck = box(labels'=='Truck',:);

% Display the predictions on the point cloud.
helperDisplay3DBoxesOverlaidPointCloud(ptCloud.Location,boxlabelsCar,'green',...
    boxlabelsTruck,'magenta','Predicted Bounding Boxes');



%% Evaluate Detector Using Test Set

% Evaluate the trained object detector on a large set of point cloud data to measure the performance.

numInputs = 50;

% Generate rotated rectangles from the cuboid labels.
bds = boxLabelDatastore(testLabels(1:numInputs,:));
groundTruthData = transform(bds,@(x) createRotRect(x));

% Set the threshold values.
nmsPositiveIoUThreshold = 0.5;
confidenceThreshold = 0.25;

detectionResults = detect(detector,testData(1:numInputs,:),...
    'Threshold',confidenceThreshold);

% Convert to rotated rectangles format for calculating metrics
for i = 1:height(detectionResults)
    box = detectionResults.Boxes{i};
    detectionResults.Boxes{i} = box(:,[1,2,4,5,7]);
end

metrics = evaluateDetectionAOS(detectionResults,groundTruthData,...
    nmsPositiveIoUThreshold);
disp(metrics(:,1:2))



%% Helper Functions

function helperDownloadPandasetData(outputFolder,lidarURL)
% Download the data set from the given URL to the output folder.

    lidarDataTarFile = fullfile(outputFolder,'Pandaset_LidarData.tar.gz');
    
    if ~exist(lidarDataTarFile,'file')
        mkdir(outputFolder);
        
        disp('Downloading PandaSet Lidar driving data (5.2 GB)...');
        websave(lidarDataTarFile,lidarURL);
        untar(lidarDataTarFile,outputFolder);
    end
    
    % Extract the file.
    if (~exist(fullfile(outputFolder,'Lidar'),'dir'))...
            &&(~exist(fullfile(outputFolder,'Cuboids'),'dir'))
        untar(lidarDataTarFile,outputFolder);
    end

end

function helperDisplay3DBoxesOverlaidPointCloud(ptCld,labelsCar,carColor,...
    labelsTruck,truckColor,titleForFigure)
% Display the point cloud with different colored bounding boxes for different
% classes.
    figure;
    ax = pcshow(ptCld);
    showShape('cuboid',labelsCar,'Parent',ax,'Opacity',0.1,...
        'Color',carColor,'LineWidth',0.5);
    hold on;
    showShape('cuboid',labelsTruck,'Parent',ax,'Opacity',0.1,...
        'Color',truckColor,'LineWidth',0.5);
    title(titleForFigure);
    zoom(ax,1.5);
end