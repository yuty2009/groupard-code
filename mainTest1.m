clc
clear

% 6 by 6  matrix
matrix=['ABCDEF','GHIJKL','MNOPQR','STUVWX','YZ1234','56789_'];
datapath = 'data\';

subject = 'Subject_B';
disp(['loading data for ' subject]);
load([datapath subject]);

featureTrain = double(featureTrain);
labelTrain = double(labelTrain);
featureTest = double(featureTest);
labelTest = double(labelTest);
numChars = dims(1);
numRepeats = dims(2);
numSamples = dims(3);
numChannels = dims(4);
numTrain = size(featureTrain,1)/(numChars*numRepeats);
numTest = size(featureTest,1)/(numChars*numRepeats);

% e.g. group = [1 1 1 2 2 2 3 3 3 4 4 4]' indicates that there are 4 group with 3 members in each
groupChannel = reshape(repmat(1:numChannels,numSamples,1),1,numChannels*numSamples);

disp('training models');
X = featureTrain;
X = reshape(X,size(X,1),size(X,2)*size(X,3));
X = svmscale(X,[0 1],'range','s');
y = labelTrain;
y(y==0) = -1;
clear featureTrain;
clear labelTrain;

model = bardgroupreg(y, X, groupChannel);

disp('classifying');
X = featureTest;
X = reshape(X,size(X,1),size(X,2)*size(X,3));
X = svmscale(X,[0 1],'range','r');
y = labelTest;
y(y==0) = -1;
clear featureTest;
clear labelTest;

idxp = find(y==1);
idxn = find(y==-1);

yprob = X*model.b + model.b0;
ypred = sign(yprob);

TP = length(find(ypred(idxp)==1));
FP = length(find(ypred(idxn)==1));
TN = length(find(ypred(idxn)==-1));
FN = length(find(ypred(idxp)==-1));
confusion = [TP,TN,FP,FN];

targetTrue = targetTest;
targetPredicted = zeros(numRepeats,numTest);
for trial = 1:numTest
    yprob1 = yprob(:,1);
    ytrial = yprob1((trial-1)*numChars*numRepeats+(1:numChars*numRepeats));
    ytrial = reshape(ytrial,numChars,numRepeats);
    for repeat = 1:numRepeats
        yavg = mean(ytrial(:,1:repeat),2);
        [dummy,pRow] = max(yavg(7:12));
        [dummy,pCol] = max(yavg(1:6));
        targetPredicted(repeat,trial) = matrix((pRow-1)*6+pCol);
    end
end

for j = 1:numRepeats
    accuracyTest(j) = length(find(squeeze(targetPredicted(j,:)) == targetTrue'))/numTest;
end

disp('showing results');

f1 = figure;
w1 = model.b;
wgroup = reshape(w1, length(w1)/numChannels, numChannels);
wtopo = mean(abs(wgroup), 1);
chanlocs = readlocs('eloc64.txt',  'filetype', 'loc');
topoplot(wtopo, chanlocs, 'maplimits', 'absmax', 'electrodes', 'on');

f2 = figure;
hold on; grid on;
plot(accuracyTest*100,'r-','LineWidth',2);
axis([1 numRepeats 0 100]);
xlabel('Repeat (n)');
ylabel('Accuracy (%)');
title(subject);
