clear;
[FileName,PathName] = uigetfile('*.txt','Please select the data(txt) file');

prompt={'Learning rate: ', ...
    'Maximum number of Epochs(required for non-separable cases): ', ...
    'Number of columns in data file'};
title_p='Perceptron part 2 parameters'; 
answer=inputdlg(prompt,title_p);
L_RATE = str2double(answer{1}); 
% N_MAX, maximum number of Epochs/Iterations
N_MAX = str2double(answer{2});
FILE_NCOLS = str2double(answer{3});

Str = fileread(strcat(PathName,FileName));
FormatString = repmat('%f',1,FILE_NCOLS);
% Read input into Data
Data = textscan(Str, FormatString, 'Delimiter', ',');

% number of rows 
n_rows = size(Data{1},1);

% Initialise M with zeros 
M = zeros(n_rows, FILE_NCOLS);

% Add the input to M column by column
for i = 1:FILE_NCOLS
    M(:,i)=Data{i};
end

% T = target values, the last column
T = M(:,FILE_NCOLS);

% M now contains only inputs  x1 .. xn, all columns except last
M = M(:,1:(FILE_NCOLS-1));
% Add x0(bias) at first column
M = [ones(n_rows,1) M];

% The number of input variables
N_VAR = size(M,2);

% Initialise bias and weights
W = zeros(1, N_VAR);
% Initialise threhold
threshold = 0.5;

% Number of instances
N_INS = n_rows;

% Intialise E to be 0 
E = zeros(1,N_MAX);
i = 1;

e = zeros(1,N_MAX*N_INS);
Weights = zeros(N_MAX*N_INS, N_VAR);
k = 1;

% Intialise linear separability to 0
lin_sep = 0;

while 1
    for j = 1:N_INS
        % p = the input variables for current j'th instance
        p = M(j,:);
        % For classification, use hard-limit transfer function(hardlim) as 
        % activation function
        a = hardlim( p * W' - threshold );
        e(k) = T(j) - a;
        Weights(k, :) = W;
        if e(k) ~= 0
            W = W + L_RATE .* e(k) .* M(j,:);
            E(i) = E(i) + 1;
        end
        k = k + 1;
    end
    % If all instances have no error(linearly separable), then done
    if E(i) == 0
        % if iteration error is 0, it is linearly separable
        lin_sep = 1;
        break;
    end
    % If it's not linearly separably, set maximum to avoid infinite loop
    if i > N_MAX-1
        break;
    end
    i = i + 1;
end


% The length of Weights is k-1, it may be N_MAX*N_INS if it's not linearly
% separable, or smaller if it's converged(linearly separable)
Wlength = k-1;


clf;
figure(1);
hold on;
title ('Error vs Number of Epochs');
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
    plot(1:Wlength, Weights(1:Wlength,i));
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

strthreshold = [' > ' num2str(threshold)];
statement = strcat(statement, strthreshold );
statement = strcat(statement, '\n');

fprintf(statement);

if lin_sep
    fprintf('The dataset is linearly separable ! \n');
else
    fprintf('The dataset is not linearly separable or the number of epochs is too small for given learning rate. \n');
end


res = input('Would you like to test the perceptron? ', 's');
goAgain = isequal(upper(res),'YES');

% Number of input variables expected from user
X_n = FILE_NCOLS-1;

while goAgain
    cur_x = str2num(input('Enter your input values: ', 's'));
    while length(cur_x) ~= X_n
        errormsg = ['Please enter ' num2str(X_n) ' boolean input values: '];
        cur_x = str2num(input(errormsg, 's'));
    end
    % add the x0=1 for bias
    cur_x_bias = [1 cur_x];
    prediction = hardlim( cur_x_bias * W' - threshold );
    fprintf('The predicted outcome is %f \n', prediction);
    res = input('Would you like to test again ? ', 's');
    goAgain = isequal(upper(res),'YES');
end