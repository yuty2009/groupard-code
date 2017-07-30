clc
clear

datapath = 'F:\bcicompetition\bci2005\II\';

% 6 by 6  matrix
matrix=['ABCDEF','GHIJKL','MNOPQR','STUVWX','YZ1234','56789_'];

subject = 'Subject_B';
fileTrain = [datapath subject '_Train.mat'];
fileTest = [datapath subject '_Test.mat'];
fileTrueLabel = 'true_labels_b.txt';

fs = 240;
order = 10;
fstop1 = 0;    % First Stopband Frequency
fpass1 = 0.5;  % First Passband Frequency
fpass2 = 20;   % Second Passband Frequency
fstop2 = 21;   % Second Stopband Frequency
wstop1 = 1;    % First Stopband Weight
wpass  = 1;    % Passband Weight
wstop2 = 2;    % Second Stopband Weight
dens  = 20;     % Density Factor
b  = firpm(order, [0 fstop1 fpass1 fpass2 fstop2 fs/2]/(fs/2), [0 0 1 1 0 ...
    0], [wstop1 wpass wstop2], {dens});
Hd = dfilt.dffir(b);

[dataRaw, targetRaw] = readTrainData1(fileTrain, Hd);

numTrials = length(dataRaw);
[numChars, numRepeats, numSamples, numChannels] = size(dataRaw{1});

channelAvailable = [1:64];%[34 4 11 18 48:54 57:59 61:63 ]; % 
numChannels = length(channelAvailable);

segmentSelected = [0 0.6*fs];
numSamples = segmentSelected(2) - segmentSelected(1);
dfs = 6;
numSamplesUsed = numSamples/dfs;
dims = [numChars, numRepeats, numSamplesUsed, numChannels];

[dataTrain, targetTrain] = extractFeature1(dataRaw, targetRaw, channelAvailable, segmentSelected);

featureTrain = [];
labelTrain = [];
for trial = 1:numTrials
    dataTrial = dataTrain{trial};
    featureTrial = reshape(dataTrial, numChars*numRepeats, numSamplesUsed, numChannels);
    featureTrain = cat(1, featureTrain, featureTrial);

    targetIndex = strfind(matrix, targetTrain(trial));
    targetRow = floor((targetIndex-1)/6) + 1;
    targetCol = targetIndex - (targetRow-1)*6;
    labelTrial = zeros(numChars,1);
    labelTrial([targetCol,targetRow+6]) = 1;
    labelTrain = cat(1,labelTrain,repmat(labelTrial,numRepeats,1));
end
featureTrain = single(featureTrain);
labelTrain = single(labelTrain);

[data target] = readTestData1(fileTest, fileTrueLabel, Hd);
[dataTest, targetTest] = extractFeature1(data, target, channelAvailable, segmentSelected);
clear data;
clear target;

numTrials = length(dataTest);
featureTest = [];
labelTest = [];
for trial = 1:numTrials
    dataTrial = dataTest{trial};
    featureTrial = reshape(dataTrial, numChars*numRepeats, numSamplesUsed, numChannels);
    featureTest = cat(1, featureTest, featureTrial);

    targetIndex = strfind(matrix, targetTest(trial));
    targetRow = floor((targetIndex-1)/6) + 1;
    targetCol = targetIndex - (targetRow-1)*6;
    labelTrial = zeros(numChars,1);
    labelTrial([targetCol,targetRow+6]) = 1;
    labelTest = cat(1,labelTest,repmat(labelTrial,numRepeats,1));
end
featureTest = single(featureTest);
labelTest = single(labelTest);

save(['data/' subject],'featureTrain','labelTrain','targetTrain','featureTest','labelTest','targetTest','dims');