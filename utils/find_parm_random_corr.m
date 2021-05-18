function [lambda] = find_parm_random_corr(std_goal)
    % find_parm_random_corr finds the value of the beta distribution parameters
    % that gives a distribution of off_diagonal elements with a std pf std_goal
    %   H Roder 2/21/2021 

    slope = -0.448113406441215;
    intercept = -0.815117444519679;
    fit_f = @(x) exp(intercept) * exp(slope*log(x)) - std_goal;
    lambda = fzero(fit_f,[0.5,1000000]);
end

