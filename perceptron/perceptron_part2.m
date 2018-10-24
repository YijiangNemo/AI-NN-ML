clear;
[FileName,PathName] = uigetfile('*.txt','Please select the data(txt) file');

Get_input={'Max iteration times: ', ...
    'Learning rate: ', ...
    'Number of columns in data file'};
Title='Perceptron part 2 parameters'; 
Input_number=inputdlg(Get_input,Title);
Iteration_Maxtimes = str2double(Input_number{1}); 
LearningRate = str2double(Input_number{2});
Number_columninfile = str2double(Input_number{3});

File = fileread(strcat(PathName,FileName));
Format = repmat('%f',1,Number_columninfile);
Data_in_file = textscan(File, Format, 'Delimiter', ',');
Number_original_datarow = size(Data_in_file{1},1);

% create Data_updated list with zeros 
Data_updated = zeros(Number_original_datarow, Number_columninfile);

for i = 1:Number_columninfile
    Data_updated(:,i)=Data_in_file{i};
end

% Target is the last column of data file
Target = Data_updated(:,Number_columninfile);


Data_updated = Data_updated(:,1:(Number_columninfile-1));

% Add x0(bias) at first column
Data_updated = [ones(Number_original_datarow,1) Data_updated];

Number_attributes = size(Data_updated,2);

% create weight list
Weight = zeros(1, Number_attributes);

Threshold = 0.5;

Number_instances = Number_original_datarow;
%MSE is default to 0
MSE = zeros(1,Iteration_Maxtimes);
i = 1;

e = zeros(1,Iteration_Maxtimes*Number_instances);
Weights = zeros(Iteration_Maxtimes*Number_instances, Number_attributes);
counter = 1;

% linear separability indicator is default to 0
Linear_Sep_indicator = 0;

while 1
    for j = 1:Number_instances
        Input_value_current = Data_updated(j,:);
        % For classification, use hard-limit transfer function(hardlim) as 
        % activation function
        WX = hardlim( Input_value_current * Weight' - Threshold );
        e(counter) = Target(j) - WX;
        Weights(counter, :) = Weight;
        if e(counter) ~= 0
            Weight = Weight + LearningRate .* e(counter) .* Data_updated(j,:);
            MSE(i) = MSE(i) + 1;
        end
        counter = counter + 1;
    end
    % If MSE is 0 then is linearly-separable,break the loop
    if MSE(i) == 0
        Linear_Sep_indicator = 1;
        break;
    end
    % If it is non-linearly-separable and it reach the maximum iteration
    % times, then break the loop
    if i > Iteration_Maxtimes-1
        break;
    end
    i = i + 1;
end


Size_weight = counter-1;


clf;
figure(1);
hold on;
title ('MSE in each time of Iteration');
xlabel('Iteration Times');
ylabel('Mean Squared Error');
plot(1:length(MSE),MSE);
hold off;

figure(2);
hold on;
title('Weights in each time update');
xlabel('Instance');
ylabel('Weights(i)');
for i = 1:length(Weight)
    plot(1:Size_weight, Weights(1:Size_weight,i));
end
hold off;

Output_Functions = 'Output Function: ';

for i=1:length(Weight)
    Weight_current = strcat(num2str(Weight(i)), strcat('X',num2str(i-1)));
    Output_Functions = [Output_Functions Weight_current];
    if i < length(Weight)
        Output_Functions = strcat(Output_Functions, ' +  ');
    end
end

Threshold_Output = [' > ' num2str(Threshold)];
Output_Functions = strcat(Output_Functions, Threshold_Output );
Output_Functions = strcat(Output_Functions, '\n');

fprintf(Output_Functions);

if Linear_Sep_indicator
    fprintf('The dataset is linearly-separable. \n');
else
    fprintf('The dataset is non-linearly-separable (or the times of iteration is too small for this learning rate). \n');
end


Message = input('Test the algorithm? ', 's');
Test_trigger = isequal(upper(Message),'YES');

Number_input_value = Number_columninfile-1;

while Test_trigger
    Data_input_test = str2num(input('Input the data: ', 's'));
    while length(Data_input_test) ~= Number_input_value
        warning = ['Please enter ' num2str(Number_input_value) ' boolean values: '];
        Data_input_test = str2num(input(warning, 's'));
    end
    % add the x0=1 for bias
    Data_input_test_bias = [1 Data_input_test];
    prediction = hardlim( Data_input_test_bias * Weight' - Threshold );
    fprintf('The prediction output is %f \n', prediction);
    Message = input('Test the algorithm? ', 's');
    Test_trigger = isequal(upper(Message),'YES');
end