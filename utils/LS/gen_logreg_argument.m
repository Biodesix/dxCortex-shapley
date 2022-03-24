function [linear] = gen_logreg_argument(ftrs_in_coalition,ftrs_used_in_kNN,N_ftrs_used_in_kNN,NS,Nmcs,mc_labels_dict,coeffs)
% gen_logreg_argument: calculate the linear part, argument of the logreg,
%   for a coalition function
% Input:
%       ftrs_in_coalition:  Array of the features used in a coalition
%       ftrs_used_in_kNN:   Array of the features used in a mini-classifier
%       N_ftrs_used_in_kNN: Array containing the number of ftrs used in mc
%       NS:                 Number of samples
%       Nmcs:               Number of mini-classifiers in the doi
%       mc_labels_dict:     dictionary of mini-classifier labels 
%                           key: sorted string of features
%                           value: array (over samples) of mini-classifier
%                           labels
%   v1.0        HR                  3/14/2022
%   v1.1        HR                  3/19/2022
%                                   replace int2str and num2st
%   v1.2        HR                  3/19/2022
%                                   hand code set intersection
%   v1.3        HR                  3/20/2022
%                                   include N_ftrs_used_in_kNN 
%
    linear = zeros(NS,1);
    linear(1:NS) = coeffs(1);
    for imc = 1:Nmcs
        ftrs_in_mc = ftrs_used_in_kNN(imc,1:N_ftrs_used_in_kNN(imc));
        %overlap = intersect(ftrs_in_coalition,ftrs_in_mc);
        % handcode intersection for arrays of positive integers
        % NOTE !!!!! for large integers one may have to use another
        % indirection
        if ( ~isempty(ftrs_in_coalition) && ~isempty(ftrs_in_mc) )
            PP = zeros(1, max(max(ftrs_in_coalition),max(ftrs_in_mc)));
            PP(ftrs_in_coalition) = 1;
            overlap = ftrs_in_mc(logical(PP(ftrs_in_mc)));
        else
            overlap = [];
        end
        %key = num2str(sort(overlap));           % sort, maybe unnecessary
        if( ~isempty(overlap) )
            key = myfastint2str(sort(overlap)); 
            if (mc_labels_dict.isKey(key))              % all other kNNs are 0
                linear = linear + coeffs(imc+1)*mc_labels_dict(key);
            end
        end
    end
    
end

