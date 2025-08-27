classdef SOM
    %SOM is a class for performing self-organizing maps analysis 
    %   This class utilize the "selforgmap" function from Neural Network
    %   Toolbox.
    %   
    %   Properties of the class:
    %   dim:            Dimension of the node organization, should be a vector
    %                   containing two integers
    %   dist_func:      Distance function (dist (default), linkdist, boxdist, mandist)
    %   topo_func:      Topology function (gridtop (default), hextop, randtop)
    %   init_neighbor:  Initial neighbor number (default: 3)
    %   cover_steps:    Cover steps (default: 100)
    %
    %   N:              node number, equals to dim(1)*dim(2)
    %   net:            SOM network object
    %   node_weight:    SOM node weight, a N-by-n matrix, n is sample dimension
    %   use_pca:        flag indicates if the network is trained after PCA
    %   eof:            PC coefficient / EOF modes
    %   expl:           explained variance percentage of each PC
    %   All methods available are list below in "How to use".
    %
    %   How to use:
    %   1. Construct a SOM object
    %       obj = SOM([3,3]);    % or
    %       obj = SOM([3,3], 'dist_func', 'linkdist', ...);
    %   2. Train the SOM network with real data
    %       obj = obj.train_som(data);
    %       obj = obj.train_som_pca(data);  % perform PCA and use PC for
    %                                       % SOM training, which is a faster way
    %   3. Classify original data or new data to different nodes
    %       [result_class, mean_value] = obj.classify(data1);
    %       % result_class is a vector containing p elements, p is the sample
    %       % number in data1; value is a integer from 1 to N indicating
    %       % the node to which the samples are classified.
    %       % mean_value is the mean sample value in each node.
    %   4. Get frequency of occurrence of each node
    %       f = SOM.get_frequency(result_class);
    %   5. Get correlation coefficient of each sample and weight of its 
    %      corresponding node
    %       [r,p,rm,rs] = get_corr_coef(data, result_class, node_weight)
    %       % r is correlation coefficient for all samples, and p is the
    %       % p-value; rm is the mean correlation coefficient for each
    %       % node, and rs is the standard deviation
    %   
    %Author: Yizhou Zhuang
    %Last Updated: 05/07/2018  
    properties
        dim
        dist_func
        topo_func
        init_neighbor
        cover_steps
        use_pca
        eof
        pc
        expl
    end
    properties (Hidden)
        net
    end
    properties(Dependent)
        N
        node_weight
        num_eof
    end
    
    methods
        %%
        function v = get.N(obj)         % number of node
           v = obj.dim(1) * obj.dim(2);
        end
        function num_eof = get.num_eof(obj)
            num_eof = get_num_eof(obj);
        end
        function iw = get.node_weight(obj)  % retrieve node weight
            iw = obj.get_node_weight();
        end
        %% constructor
        function obj = SOM(dim,varargin)
            %UNTITLED5 Construct an instance of this class
            %   Detailed explanation goes here
            p = inputParser;
            default_tf = 'gridtop';
            expected_tf = {'hextop', 'gridtop', 'randtop'};
            default_df = 'linkdist';
            expected_df = {'linkdist', 'dist', 'boxdist', 'mandist'};
            default_in = 3;
            default_ct = 100;
            addRequired(p, 'dim', @ismatrix);
            addParameter(p, 'topo_func', default_tf, ...
                @(x) any(validatestring(x,expected_tf)));
            addParameter(p, 'dist_func', default_df, ...
                @(x) any(validatestring(x,expected_df)));
            addParameter(p, 'cover_steps', default_ct, ...
                @(x) isnumeric(x) && isscalar(x) && x>0);
            addParameter(p, 'init_neighbor', default_in, ...
                @(x) isnumeric(x) && isscalar(x) && x>0);
            parse(p, dim, varargin{:});
            obj.dim = dim;
            obj.topo_func = p.Results.topo_func;
            obj.dist_func = p.Results.dist_func;
            obj.cover_steps = floor(p.Results.cover_steps);
            obj.init_neighbor = floor(p.Results.init_neighbor);
        end
        
        %% train network
        function obj = train_som(obj, data)     % normal training
            net = selforgmap1(obj.dim, obj.cover_steps, obj.init_neighbor, ...
                obj.topo_func, obj.dist_func); %%%%%%%%%%%%
%             net = selforgmap(obj.dim, obj.cover_steps, obj.init_neighbor, ...
%                 obj.topo_func, obj.dist_func); %%%%%%%%%%%%
            obj.net = train(net, SOM.remove_nan(data));
            obj.use_pca = false;
        end
        
        function obj = train_som_pca(obj, data) % perform PCA before training
            [data, ind_nan] = SOM.remove_nan(data);
            [eof, pc, expl] = GeoData.get_eof(data, 'MinExpl', 90);  %%%
            disp(90);
            obj = obj.train_som(pc');
            obj.use_pca = true;
            obj.eof = eof;
            obj.pc = SOM.restore_nan(pc', ind_nan)';
            obj.expl = expl;
        end
        %% SOM classifier
        function [result_class, mw] = classify(obj, varargin)
            if nargin==1
                if obj.use_pca, data = obj.pc'; 
                else, data = obj.value; end
            elseif nargin == 2
                if obj.use_pca, data = obj.eof' * varargin{1};
                else, data = varargin{1}; end
            end
            ne_total = size(data,1);
            [data1,ind_nan] = SOM.remove_nan(data);
            result_class = vec2ind(obj.net(data1));
            mw = nan(obj.N, ne_total);
            for i = 1 : obj.N
                mw(i,:) = nanmean(data1(:,result_class==i), 2);
            end
            if obj.use_pca          % for PCA training
                mw = mw * obj.eof';
            end
            result_class = SOM.restore_nan(result_class, ind_nan);
        end
        %% calculate quantization error
        function [mqe, mte, cbe] = quality(obj, data)
            sMap.codebook = obj.get_node_weight();
            sMap.topol.msize = obj.dim;
            sMap.topol.type = 'som_topol';
            sMap.topol.lattice = 'rect';
            sMap.topol.shape = 'sheet';
            sMap.mask = ones(size(sMap.codebook, 2), 1);
            sMap.type = 'som_map';
            [mqe, mte, cbe] = som_quality(sMap, data');
        end
        function [mqe, mte, cbe] = quality1(obj, data, data_node)
            sMap.codebook = data_node;
            sMap.topol.msize = obj.dim;
            sMap.topol.type = 'som_topol';
            sMap.topol.lattice = 'rect';
            sMap.topol.shape = 'sheet';
            sMap.mask = ones(size(sMap.codebook, 2), 1);
            sMap.type = 'som_map';
            [mqe, mte, cbe] = som_quality(sMap, data');
        end
        function [mqe, mte, cbe] = quality_pc(obj, data)
            sMap.codebook = obj.net.IW{1,1};
            sMap.topol.msize = obj.dim;
            sMap.topol.type = 'som_topol';
            sMap.topol.lattice = 'rect';
            sMap.topol.shape = 'sheet';
            sMap.mask = ones(size(sMap.codebook, 2), 1);
            sMap.type = 'som_map';
            [mqe, mte, cbe] = som_quality(sMap, data'*obj.eof);
        end
    end
    %% Static methods
    methods (Static)
        
        
        % calculate frequency of each node
        function f = get_frequency(result_class)    
            N = max(result_class);
            f = nan(N, 1);
            for i = 1 : N
                f(i) = sum(result_class==i)/numel(result_class) * 100;
            end
        end
        function [r,p,rm,rs] = get_corr_coef(data, result_class, nw)
            N = size(nw, 1);
            np = size(data, 2);
            r = nan(np, N);
            p = nan(np, N);  % p-value
            for i = 1 : np
                if ~isnan(result_class(i))
                    [R, P] = corrcoef(data(:,i), nw(result_class(i),:)');
                    r(i,result_class(i)) = R(1,2);
                    p(i,result_class(i)) = P(1,2);
                end
            end
            rm = nanmean(r, 1);
            rs = nanstd(r, 0, 1);
        end
        
    end
    %%
    methods (Access=protected)
        function iw = get_node_weight(obj)
           if ~isempty(obj.net)
                iw = obj.net.IW{1};
            else
                error('network is not trained!');
            end
            if obj.use_pca
                iw = iw * obj.eof';      
            end 
        end
        function num_eof = get_num_eof(obj)
           num_eof = size(obj.eof, 2); 
        end
        
    end
    methods (Access=protected,Static=true)
        function [data1,ind] = remove_nan(data)
            ind = any(isnan(data), 1);
            data1 = data(:,~ind);
        end
        function data1 = restore_nan(data,ind)
            data1 = nan(size(data,1), length(ind));
            data1(:,~ind) = data;
        end 
    end
end

