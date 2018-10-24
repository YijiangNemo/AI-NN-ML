Program: perceptron_part1.m

Inputs: 
1. Prompt user to select data file(i.e. mpgdata.txt) each row containing 
n features and target value x1,...,xn,t, which are comma separated 
2. Ask uses input for Learning Rate, Number of Iterations, Number of columns in data file 
and the Columns in file that are discrete valued such as origin {1,2,3}

Outputs:
1. Plots of a. The Iteration Error vs Number of Iterations
and b. The Weights after each instance update
2. Output the rule W0*X0 + W1*X1 + ... Wn*Xn to command window
3. Ask for input values(X1 ... Xn) in command window and give the predicted output

This program essentially implement the gradient descent algorithm for a fixed
number of iterations.

Note:
1. Rows with missing data '?' are ignored as it's only a small proportion and also there is 
not a statistically optimised way of imputing these values.
2. Discrete valued variables are split into multiple boolean variables for its values
3. The input variables are normalised by min max rule to avoid too large X values dominating
weight update 

Part 1 Data files:
mpgdata.txt - all the given data 
We've randomly selected 20 instances from mpgdata into mgptest, and leave the rest in mpgtrain
mpgtrain.txt
mpgtest.txt

One can use mpgtrain as input for program, and then input 1 or more of the mgptest instances
into command window for validation purposes.

-----------------------------------------------------------------------------------------

Program: perceptron_part2.m

Inputs: 
1. Prompt user to select data file(i.e. 10101010.txt) each row containing 
n(n=8 in this case) features and 1 target value x1,...,xn,t, which are comma separated 
2. Ask uses input for Learning Rate, Number of Iterations, Number of columns in data file 

Outputs:
1. Plots of a. The Iteration Error vs Number of Iterations 
and b. The Weights after each instance update
2. Output the rule W0*X0 + W1*X1 + ... Wn*Xn to command window
3. Ask for input values(X1 ... Xn) in command window and give the predicted output

This program essentially implement the gradient descent algorithm until
either (a) the iteration error is 0 (successfully classify all given data)
or (b) reach the max number of iterations (possibly due to data being not linearly separable)

Note:
1. The threshold is set to 0.5 for a boolean output.

Part 2 Data files:
10101010.txt
x.txt
xIFFy.txt