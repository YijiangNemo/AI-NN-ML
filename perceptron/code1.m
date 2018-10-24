%Written by Yijiang wu,z5121109
clear;
[FileName,PathName] = uigetfile('*.txt','input Data');

Get_input={'times of iteration: ','Learning rate: ', ...
    'Number of attributes: ', ...
    'Number of discrete value column: '};
Title='Assignment2 Topic3 Perceptron / Linear Unit part1'; 
Input_number=inputdlg(Get_input,Title);
Iteration_times = str2double(Input_number{1}); 
LearningRate = str2double(Input_number{2});
Number_columninfile = str2double(Input_number{3});
Discrete_values_column = str2num(Input_number{4});
File = fileread(strcat(PathName,FileName));

File = strrep(File, '?', '');% Transfer '?' to ''

FormatString = repmat('%f',1,Number_columninfile);
% Transfer empty value '' to NaN
Data_in_file = textscan(File, FormatString, 'Delimiter', ',', 'EmptyValue', NaN);

Number_original_datarow = size(Data_in_file{1},1);% number of rows before deleting missing value rows

Data_updated = zeros(Number_original_datarow, Number_columninfile);% create Data_updated list with zeros 



for i = 1:Number_columninfile
    Data_updated(:,i)=Data_in_file{i};
end

% Skip the NaN value
Data_updated = Data_updated(~any(isnan(Data_updated),2),:);
Data_original = Data_updated;


Number_update_datarow = size(Data_updated,1);
Number_discrete_column = zeros(Number_update_datarow,length(Discrete_values_column));
Number_Discrete_features = zeros(1,length(Discrete_values_column));

% Get the Discrete values
for i = 1:length(Discrete_values_column)
    Discrete_column_current = sort(unique(Data_updated(:,Discrete_values_column(i))));
    Discrete_values_input = length(Discrete_column_current);
    % Discrete_values_input: The number of distinct values for i'th discrete input
    Number_Discrete_features(i) = Discrete_values_input;
    Number_discrete_column(1:Discrete_values_input,i) = Discrete_column_current;
end

% Transfer the discrete values to boolean value into Data_updated
for i = 1:length(Discrete_values_column)
    Discrete_current_features = zeros(Number_update_datarow,Number_Discrete_features(i));
    Number_discrete_values = Number_Discrete_features(i);
    Discrete_values_features = Number_discrete_column(1:Number_discrete_values,i);
    for j = 1:Number_discrete_values
        Discrete_current_features(Data_updated(:,Discrete_values_column(i))==Discrete_values_features(j), j) = 1;
    end
    Data_updated = [Data_updated Discrete_current_features];
end

% Target is the last column of data file
Target = Data_updated(:,Number_columninfile);

% delete discrete values and target value
Columns_delete = [Discrete_values_column Number_columninfile];
Data_updated(:,Columns_delete)=[];

% Add x0(bias)
Data_updated = [ones(Number_update_datarow,1) Data_updated];

Number_attribute = size(Data_updated,2);

% min max normalisation
for i = 1:Number_attribute
    if max(Data_updated(:,i)) ~= min(Data_updated(:,i))
        Data_updated(:,i) = (Data_updated(:,i) - min(Data_updated(:,i)))./(max(Data_updated(:,i))-min(Data_updated(:,i)));
    end
end

% create a Weight list
Weight_updated = zeros(1, Number_attribute);

Number_instance = Number_update_datarow;

% create a iteration MSE(Mean Squared Error) list 
Iteration_Error = zeros(1, Iteration_times);

Weight_list = zeros(Iteration_times*Number_instance, Number_attribute);
Counter = 1;

for x = 1:Iteration_times
    for y = 1:Number_instance
        Data_current = Data_updated(y,:);
        % use linear transfer(purelin) as activation function
        WX = purelin( Data_current * Weight_updated');
        e = Target(y) - WX;
        Weight_list(Counter, :) = Weight_updated;
        Counter = Counter + 1;
        Weight_updated = Weight_updated + LearningRate .* e .* Data_updated(y,:);
        Iteration_Error(x) = Iteration_Error(x) + abs(e);
    end
    Iteration_Error(x) = Iteration_Error(x) / Number_instance;
end

clf;
figure(1);
hold on;
title('Error in each time of Iteration');
xlabel('Iteration Times');
ylabel('Error');
plot(1:length(Iteration_Error),Iteration_Error);
hold off;

figure(2);
hold on;
title('Weights in each time update');
xlabel('Instance');
ylabel('Weights');
for i = 1:length(Weight_updated)
    plot(1:length(Weight_list(:,i)), Weight_list(:,i));
end
hold off;

Output_Function = 'Output Function: ';

for i=1:length(Weight_updated)
    Weight_output = strcat(num2str(Weight_updated(i)), strcat('X',num2str(i-1)));
    Output_Function = [Output_Function Weight_output];
    if i < length(Weight_updated)
        Output_Function = strcat(Output_Function, ' +  ');
    end
end
 
Output_Function = strcat(Output_Function, '\n');

fprintf(Output_Function);

Message = input('Test the algorithm? ', 's');
Test_trigger = isequal(upper(Message),'YES');

Input_size = size(Data_original,2)-1;

while Test_trigger
    Data_input_test = str2num(input('Input the data: ', 's'));
    while length(Data_input_test) ~= Input_size
        Warning = ['Please enter ' num2str(Input_size) ' input values: '];
        Data_input_test = str2num(input(Warning, 's'));
    end
    Continuous_value_columns=1:Input_size;
    Continuous_value_columns(Discrete_values_column)=[];
    for i2 = 1: length(Continuous_value_columns)
        %  min max normalisation
        Data_input_test(Continuous_value_columns(i2)) = ( Data_input_test(Continuous_value_columns(i2)) - min(Data_original(:,Continuous_value_columns(i2))) ) / ...
            (max(Data_original(:,i2)) - min(Data_original(:,i2)));
    end
    % Transfer the discrete value input to boolean value
    Discrete_input_test = [];
    for i = 1:length(Discrete_values_column)
        Position_discrete = find(Number_discrete_column(:,i)==Data_input_test(Discrete_values_column(i)));
        Discrete_value_current = zeros(1, Number_Discrete_features(i));
        Discrete_value_current(Position_discrete) = 1;
        Discrete_input_test = [Discrete_input_test Discrete_value_current];
    end
    % delete the discrete values
    Data_input_test(Discrete_values_column) = [];
    Data_input_test = [Data_input_test Discrete_input_test];
    % add the x0=1 for bias
    Data_input_test_bias = [1 Data_input_test];
    Value_Prediction = purelin( Data_input_test_bias * Weight_updated');
    fprintf('The Prediction value is %f \n', Value_Prediction);
    Message = input('Test the algorithm? ', 's');
    Test_trigger = isequal(upper(Message),'YES');
end