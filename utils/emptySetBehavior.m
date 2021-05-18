classdef emptySetBehavior
    % enumeration of behavior of working with missing feature in SV
    % calculations
    
    enumeration
        doi
        mc_shallow
        mc_deep
    end
    
    methods
        function name = Name(obj)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            if (obj == emptySetBehavior.doi)
                name = 'doi';
            elseif (obj == emptySetBehavior.mc_shallow)
                name = 'mc_shallow';
            elseif (obj == emptySetBehavior.mc_deep)
                name = 'mc_deep';    
            end
        end
        
    end
end

