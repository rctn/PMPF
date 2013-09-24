% 
%     Gaussian-Bernoulli Restricted Boltzmann Machine Using
%           Minimum Probability Flow Learning

%     Steven Munn
%     email: stevenjlm@berkeley.edu

#####
Contents:
1) Quick Start
2) Matrix naming convention
3) Naming Convention Abbreviations
4) Minfunc options
5) Minibatch system
#####

1) Quick Start -----------------------------------

	To run any script starting with "GBRBM_main" you will need to,
    * download the minFunc scripts at http://www.di.ens.fr/~mschmidt/Software/minFunc.html
    * adapt the addpath lines (14, 15, 16) with the correct paths
    * change line 54 of GBRBM_main.m to use the learning data you want.
    
    I've included a learning data set 8by8data_W1neg3.mat
    The soon to be created repository PMPF data preprocessing will show where this data comes from.
    
    The header comments on each of these scripts contains a more detailed explantion of what they do.

*Before coding:

	Read the naming conventions. They are very important for the more mathematically involved scripts like k_dk_grbm.m

2) Naming -----------------------------------

*General variables:

	Capitalize first letter of each word and ommit space. Eg.: BurnIn, ImageSize.

*Counter variables:
	
	Begin variable name with n if it stores a total number of iterations value. Eg. nSamples is the total number of samples.
	Begin variable name with i if it stores the current iteration value of a for loop. Eg. iSample is the current sample the for loop is iterating through.

*Matrix Variables:

	Follow this format:		DescriptiveName_XbY

	Where X is the number of rows in the matrix, and Y the number of columns using the abbreviations in section 3.

*Functions and file names:

	Generally, try to keep all lower case, except for abbreviations. No spaces. Just underscore between words.

3) Naming Convention Abbreviations ------------------------------

	H = number of hidden units
	V = number of visible units
	N = number of training samples
	NP = number of training samples times number of PT chains
	BS = Batch Size

*Example:

	Dall_VbN = A matrix with all the training Data. It has as many rows are there are visible units and as many columns as there are training samples.

4) Common Minfunc options
    Helpful for checking the derivative code,
minf_options.DerivativeCheck='on';

5) Minibatch system
    Data is truncated.. (GRBM_main line 52)