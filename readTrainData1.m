function [data target] = readTrainData1(datafile, hdfilter)

disp('Loading train dataset');
load(datafile);

% convert to double precision
% Signal = double(Signal);
% Flashing = double(Flashing);
% StimulusCode = double(StimulusCode);
% StimulusType = double(StimulusType);

numTrials = size(Signal, 1);
numChars = 12;
numRepeats = 15;
numSamples = 240;
numChannels = size(Signal, 3);

Signal_filtered = zeros(size(Signal));
for i = 1:numTrials
    Signal_trial = squeeze(Signal(i,:,:));
    Signal_filtered(i,:,:) = reshape(filter(hdfilter, Signal_trial), 1, size(Signal_trial,1), size(Signal_trial,2));
end

data = cell(numTrials, 1);
target = zeros(numTrials, 1);
for i = 1:numTrials
    repeat = zeros(1, numChars);
    signalTrial = zeros(numChars, numRepeats, numSamples, numChannels);
    for n = 2:size(Signal, 2)
        if Flashing(i, n-1)==0 && Flashing(i, n)==1
            event = StimulusCode(i, n);
            repeat(event) = repeat(event) + 1;
            signalTrial(event, repeat(event), :, :) = Signal_filtered(i, n:n+numSamples-1, :);
        end
    end
    
    data{i} = signalTrial;
    target(i) = TargetChar(i);
end