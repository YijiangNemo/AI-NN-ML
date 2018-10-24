clear;
[FileName,PathName] = uigetfile('*.txt','Please select the data(txt) file');

prompt={'Learning rate: ','Number of Epochs: ', ...
    'Number of columns in data file', ...
    'Columns where data have discrete values,in the form of: c1,c2,... ,cn'};
title_p='Perceptron part1 parameters'; 
answer=inputdlg(prompt,title_p);
L_RATE = str2double(answer{1}); 
% Iteration = Epoch
N_ITER = str2double(answer{2});
FILE_NCOLS = str2double(answer{3});
% Discrete Value Columns
DVCOLUMNS = str2num(answer{4});

Str = fileread(strcat(PathName,FileName));
% replace missing value '?' with ''. These instances will be ignored
Str = strrep(Str, '?', '');
FormatString = repmat('%f',1,FILE_NCOLS);
% EmptyValue '' is assigned NaN
Data = textscan(Str, FormatString, 'Delimiter', ',', 'EmptyValue', NaN);

% number of rows before deleting missing value rows
n_rows_old = size(Data{1},1);
% Initialise M with zeros 
M = zeros(n_rows_old, FILE_NCOLS);

% Data{1} Data{2} Data{3} Data{4} Data{5} Data{6} Data{7} Data{8} are
% cylinders, displacement, horsepower, weight, acceleration, model year,
% origin, mpg(miles per gallon) respectively
% M=[Data{1} Data{2} Data{3} Data{4} Data{5} Data{6} Data{7} Data{8}];

% Generalise the above code
for i = 1:FILE_NCOLS
    M(:,i)=Data{i};
end

% Ignore any row with missing values(NaN)
M = M(~any(isnan(M),2),:);
oldM = M;

% The number of rows, where missing value ones are ignored
n_rows = size(M,1);

% Initialise C, the columns of discrete values
C = zeros(n_rows,length(DVCOLUMNS));

% dfL discrete feature lengths
dfL = zeros(1,length(DVCOLUMNS));

% Get the discrete values for the columns specified
for i = 1:length(DVCOLUMNS)
    tempC = sort(unique(M(:,DVCOLUMNS(i))));
    tempL = length(tempC);
    % tempL: The number of distinct values for i'th discrete input
    dfL(i) = tempL;
    C(1:tempL,i) = tempC;
end

% Add the discrete values(as boolean functions) into M one by one
for i = 1:length(DVCOLUMNS)
    % discreteX = current discrete feature 
    discreteX = zeros(n_rows,dfL(i));
    % number of discrete values for current feature
    n_discrete_vals = dfL(i);
    % dvals = the distinct values the current feature can take
    dvals = C(1:n_discrete_vals,i);
    for j = 1:n_discrete_vals
        % Set the j'th column of discreteX to 1 if the COLUMNS(i)'th column 
        % of M is equal to dvals(j)
        discreteX(M(:,DVCOLUMNS(i))==dvals(j), j) = 1;
    end
    M = [M discreteX];
end

% T = target values(mpg) is in the last column of data file
T = M(:,FILE_NCOLS);

% remove discrete values and target value columns
cols2remove = [DVCOLUMNS FILE_NCOLS];
M(:,cols2remove)=[];


% Add x0(bias)
M = [ones(n_rows,1) M];

% The number of input variables
N_VAR = size(M,2);

% Normalise each column(each variable) by min max normalisation
for i = 1:N_VAR
    if max(M(:,i)) ~= min(M(:,i))
        M(:,i) = (M(:,i) - min(M(:,i)))./(max(M(:,i))-min(M(:,i)));
    end
end

% Initialise weights(w0,w1, ...,wn) to 0
W = zeros(1, N_VAR);

% Number of instances
N_INS = n_rows;

% Initialise the iteration error E
E = zeros(1, N_ITER);

% Weights after each instance, k being counter
Weights = zeros(N_ITER*N_INS, N_VAR);
k = 1;

for i = 1:N_ITER
    % Need to compute E, the root mean squared error for each iteration
    for j = 1:N_INS
        % p = the input variables for current j'th instance
        p = M(j,:);
        % For numerical prediction, use linear transfer(purelin) as 
        % activation function
        a = purelin( p * W');
        e = T(j) - a;
        Weights(k, :) = W;
        k = k + 1;
        W = W + L_RATE .* e .* M(j,:);
        E(i) = E(i) + abs(e);
    end
    E(i) = E(i) / N_INS;
end

clf;
figure(1);
hold on;
title('Error vs Number of Epochs');
xlabel('Number of Epochs');
ylabel('Error');
plot(1:length(E),E);
hold off;

figure(2);
hold on;
title('Weights after each instance update');
xlabel('Instance');
ylabel('Weights_i');
for i = 1:length(W)
    plot(1:length(Weights(:,i)), Weights(:,i));
end
hold off;

statement = 'The perceptron rule is ';

for i=1:length(W)
    tempwx = strcat(num2str(W(i)), strcat('X',num2str(i-1)));
    statement = [statement tempwx];
    if i < length(W)
        statement = strcat(statement, ' +  ');
    end
end
 
statement = strcat(statement, '\n');

fprintf(statement);

res = input('Would you like to test the perceptron? ', 's');
goAgain = isequal(upper(res),'YES');

% Number of input variables expected from user
X_n = size(oldM,2)-1;

while goAgain
    cur_x = str2num(input('Enter your input values: ', 's'));
    while length(cur_x) ~= X_n
        errormsg = ['Please enter ' num2str(X_n) ' input values: '];
        cur_x = str2num(input(errormsg, 's'));
    end
    % normalise the non-discrete inputs
    index=1:X_n;
    % index, the indices of columns of oldM that are not d.v. inputs
    index(DVCOLUMNS)=[];
    for ii = 1: length(index)
        % normalise by min max normalisation
        cur_x(index(ii)) = ( cur_x(index(ii)) - min(oldM(:,index(ii))) ) / ...
            (max(oldM(:,ii)) - min(oldM(:,ii)));
    end
    % tempDV, the discrete valued input as boolean functions
    tempDV = [];
    for i = 1:length(DVCOLUMNS)
        % The position k, of the d.v. input in all its sorted values
        % e.g. 72 in {70,71,...,82} is in the 3rd position
        rank_pos = find(C(:,i)==cur_x(DVCOLUMNS(i)));
        % initialise all positions to 0 and then assign the rank_pos to 1
        cur_dv = zeros(1, dfL(i));
        cur_dv(rank_pos) = 1;
        tempDV = [tempDV cur_dv];
    end
    % remove the discrete values
    cur_x(DVCOLUMNS) = [];
    % add the boolean representation of d.v. inputs
    cur_x = [cur_x tempDV];
    % add the x0=1 for bias
    cur_x_bias = [1 cur_x];
    prediction = purelin( cur_x_bias * W');
    fprintf('The predicted value of mpg is %f \n', prediction);
    res = input('Would you like to test again ? ', 's');
    goAgain = isequal(upper(res),'YES');
end