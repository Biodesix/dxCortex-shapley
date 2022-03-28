function [NL_SVs,phi_all] = NL_SVS(level_structure,mc_labels_dict,coeffs,ftrs_used_in_kNN, N_ftrs_used_in_kNN, verbose)
%NL_SVS: Calculates nested level SVs for a doi using
%   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
%   algorithm 5.3 modified for dois and simplified to get all ftrs' SV
%   Input:
%       level_structure:    A level structure (see class)
%       mc_labels_dict:     a containers.Map object giving the values of a
%                           kNN either -1 or +1; the keys are the arrays of
%                           the features used in a kNN converted to a
%                           string via num2str(ftrs_used_in_kNN(*,:)
%       coeffs:             the logreg expansion coefficients rescaled to
%                           work with values in(-1,+1) rather than (0,1)
%       ftrs_used_in_kNN:   the features used in the kNNs of the doi
%       N_ftrs_used_in_kNN: the number of ftrs used in a kNN
%       verbose:            if 1 displays the SVs of the middle layer
%   Output:
%       NL_SVs:             the nested level Shapley values as elements of
%                           a containers.Map with keys being the features
%                           as strings
%       phi_all:            the value of the grand coalition of all
%                           features in the doi
%   Note: This is using SH.m which calculates the SVs of game using the
%   direct formula. It might be faster to use the least squares formulation
%   when calculating the SVs for the components of th e middlelayer

%   v1.0        HR                  3/13/2022
%   v1.1        HR                  3/16/2022
%                                           -- added verbose == 1 option
%   v1.2        HR                  3/19/2022  replaced num2str and int2str
%   V1.3        HR                  3/20/2022  fix P; add # ftrs as arg to
%                                              gen_logreg_linear; add
%                                              N_ftrs_used_in_kNN as
%                                              argument
%

    NL_SVs  = containers.Map;
    tmp     = mc_labels_dict.values;
    NS      = length(tmp{1});
    Nmcs    = size(ftrs_used_in_kNN,1);
    %P = @(x) exp(x)/(1+exp(x)) ;        % predefine the softmax function
    P = @(x) 2/(1+exp(-x)) -1;        % predefine the softmax function
    
    % step 1: calculate the SV for the elements of the middle layer with the
    % components of the middlelayer as players

    % how many components
    N_comp = length(level_structure.middlelayer);

    % form a list of all coalitions the middlelayer can form 
    % for later the first N_comp elements are the singletons
    % the value of this list is an index into the components of the middlelayer
    elementlist = 1:1:N_comp;
    middlelayer_coalitions = cell(1,2^N_comp-1);
    cnt = 0;
    for k = 1:N_comp
        tmp = nchoosek(elementlist,k);
        for j = 1:size(tmp,1)
            cnt = cnt + 1;
            middlelayer_coalitions{cnt} = tmp(j,:);
        end
    end

    % make a value function for all of these; could merge with previous loop
    coalition_values = containers.Map;
    for i = 1:length(middlelayer_coalitions) 
        % grab the features over all components in the coalition
        coalition = middlelayer_coalitions{i};
        ftrs_in_coalition = cell2mat(level_structure.middlelayer{coalition(1)}.features);
        for j = 2:length(coalition)
            ftrs_in_coalition = [ftrs_in_coalition cell2mat(level_structure.middlelayer{coalition(j)}.features)];
        end
        % calculate the value
        [linear] = gen_logreg_argument(ftrs_in_coalition,ftrs_used_in_kNN,N_ftrs_used_in_kNN,NS,Nmcs,mc_labels_dict,coeffs);
        coalition_values(myfastint2str(sort(coalition))) = arrayfun(P,linear); %Note: the players are the middlelayer components
    end

    % calculate the SV for the components of the middlelayer
    % make use that the first N_comp elements of middlelayer.components are the
    % singletons and are numbered by their index in the middlelayer components
    % NOTE: it might be faster to use the least square formulation for this
    % step
    middlelayer_SVs = zeros(NS,N_comp);
    tau_i = middlelayer_coalitions(1:N_comp);
    for i = 1:N_comp
        t_i = tau_i{i};
        middlelayer_SVs(:,i) = SH(t_i,tau_i,coalition_values);
    end
    
    if (verbose )
        disp([' middlelayer SVs: ' num2str(middlelayer_SVs)])
    end

    % bottomlayer for final SVs
    % loop over components
    for ic = 1:N_comp
        % get the features in this component
        ftrs_in_component = level_structure.middlelayer{ic}.features;
        % make a dictionary from numbers to features so we can map back to
        % original feature names
        number2features = containers.Map;
        for i = 1:length(ftrs_in_component)
            number2features(myfastint2str(i)) = ftrs_in_component{i};      
        end
        %make the coalition values
        c_vals = containers.Map;
        % take the grand coalition from above
        c_vals(myfastint2str(1:1:length(ftrs_in_component))) = middlelayer_SVs(ic);
        % now all the coalitions that can be formed without the grand coalition
        elementlist = 1:1:length(ftrs_in_component);   
        % these should all be singletons let's try and speed it up and
        % fix bug in line " ftrs_used_in_coalition(i) =
        % number2features(myfastint2str(i)); " at the same time
        % this fix does NOT work , I can have more than 2 features!!!!
%         for i = 1:length(ftrs_in_component)
%             coalition = i;
%             ftrs_used_in_coalition = number2features(myfastint2str(i));
%             [linear] = gen_logreg_argument(ftrs_used_in_coalition,ftrs_used_in_kNN,N_ftrs_used_in_kNN,NS,Nmcs,mc_labels_dict,coeffs);
%             c_vals(myfastint2str(sort(coalition))) = arrayfun(P,linear); %Note: the players are in numbers from 1:length(coalition)
%         end
        for k = 1:length(ftrs_in_component) - 1     % not the grand coalition
            tmp = nchoosek(elementlist,k);
            for j = 1:size(tmp,1)
                coalition = sort(tmp(j,:));   %values from 1:length(ftrs_in_component)
                ftrs_used_in_coalition = zeros(1,length(coalition));
                for i = 1:length(coalition)
                    ftrs_used_in_coalition(i) = number2features(myfastint2str(coalition(i)));   % looks like array access but in fact it is dictionary access
                end
                % now we can calculate the value
                [linear] = gen_logreg_argument(ftrs_used_in_coalition,ftrs_used_in_kNN,N_ftrs_used_in_kNN,NS,Nmcs,mc_labels_dict,coeffs);
                c_vals(myfastint2str((coalition))) = arrayfun(P,linear); %Note: the players are in numbers from 1:length(coalition)
            end
        end

        % calculate the SVs (map back to original feature names )
        tau_i = num2cell(1:1:length(ftrs_in_component));
        for i = 1:length(tau_i)
            t_i = tau_i{i};
            try
            NL_SVs(myfastint2str(number2features(myfastint2str(t_i)))) = SH(t_i,tau_i,c_vals);
            catch
                disp([])
            end
        end

    end

    % NOTE: populate phi_all
    gc = cell2mat(level_structure.bottomlayer);
    [linear] = gen_logreg_argument(gc,ftrs_used_in_kNN,N_ftrs_used_in_kNN,NS,Nmcs,mc_labels_dict,coeffs);
    phi_all = arrayfun(P,linear);

end

