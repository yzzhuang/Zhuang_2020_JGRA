classdef MVSOM < SOM
    %MVSOM is a class for performing multivariate self-organizing maps analysis 
    %   This class is a subclass of SOM
    %   
    %   Extra Properties in addition to those of SOM:
    %   num_var:    number of variables
    %   pc_scale:   scales
    %   
    %   All methods available are list below in "How to use".
    %
    %   How to use:
    %   1. Construct a MVSOM object is the same as SOM
    %       obj = MVSOM([3,3]);    % or
    %       obj = MVSOM([3,3], 'dist_func', 'linkdist', ...);
    %   2. Train the MVSOM network with real data
    %       obj = obj.train_som_pca(data);
    %   3. Classify original data or new data to different nodes
    %       [result_class, mean_value] = obj.classify(data);
    %       % data is a cell vector containing multiple variables
    %       % result_class is a vector containing p elements, p is the sample
    %       % number in data1; value is a integer from 1 to N indicating
    %       % the node to which the samples are classified.
    %       % mean_value is a cell vector containing the mean sample value in each node for each variables.
    %   4. Get node weight
    %       iw = obj.get_node_weight();
    %       iw = obj.get_node_weight(data1); % train the network with data1
    %                                        % before retrieve the node weight
    %   6. Get correlation coefficient of each sample and weight of its 
    %      corresponding node
    %       [r,p,rm,rs] = get_corr_coef(data, result_class, node_weight)
    %       % data and node_weight are cell vectors, result_class is a
    %       % N-by-n array
    %       % r is correlation coefficient for all samples, and p is the
    %       % p-value; rm is the mean correlation coefficient for each
    %       % node, and rs is the standard deviation
    %   
    %Author: Yizhou Zhuang
    %Last Updated: 05/07/2018  
    properties
        num_var
        pc_scale
    end
    %%
    methods
        function obj = MVSOM(dim, varargin)
            obj = obj@SOM(dim, varargin{:});
        end
        
        function obj = train_som_pca(obj, data)
            obj = obj.train_som(data);
        end
        function obj = train_som(obj, data)
            obj = obj.pca_all(data);
            data = MVSOM.combine_data(obj.pc)';
            obj = train_som@SOM(obj, data);
            % net = selforgmap(obj.dim, obj.cover_steps, obj.init_neighbor, ...
            % obj.topo_func, obj.dist_func);
            % obj.net = train(net, data);
            obj.use_pca = true;
        end
        function obj = train_som_nopca(obj, data)
            data = MVSOM.combine_data(data, 1);
            obj = SOM.train_som(obj, data);
            obj.use_pca = false;
        end
        
        function [result_class, mw] = classify(obj, varargin)
            nv = obj.num_var;
            if nargin == 1  % classify original data
                data = MVSOM.combine_data(obj.pc)';  % concatenate PCs of all variables
            else            % classify new data
                data = cell(1, nv);
                for i = 1 : nv
                    % PCs: new data projected to EOF modes and rescaled
                    data{i} = obj.eof{i}' * varargin{1}{i} / obj.pc_scale(i);
                end
                data = MVSOM.combine_data(data);  % concatenate PCs of new data
            end
            ne_total = size(data,1);  % total EOF mode number
            result_class = vec2ind(obj.net(data));  % classification
            mw = nan(obj.N, ne_total);
            for i = 1 : obj.N
                mw(i,:) = nanmean(data(:,result_class==i), 2);
            end
            mw = MVSOM.separate_data(mw, obj.num_eof);
            for i = 1 : nv
                mw{i} = mw{i} * obj.pc_scale(i);
            end
        end
        
%         function [r,p,rm,rs] = get_corr_coef_pc(obj, data)
%             nw = obj.net.IW{1};
%             pc = MVSOM.combine_data(obj.pc);
%             [r,p,rm,rs] = get_corr_coef@SOM(pc, result_class, nw);
%         end
%% calculate quantization error
        function [mqe, mte, cbe] = quality(obj, data)
            sMap.codebook = MVSOM.combine_data(obj.get_node_weight());
            sMap.topol.msize = obj.dim;
            sMap.topol.type = 'som_topol';
            sMap.topol.lattice = 'rect';
            sMap.topol.shape = 'sheet';
            sMap.mask = ones(size(sMap.codebook, 2), 1);
            sMap.type = 'som_map';
            tmp = MVSOM.combine_data(data,1)';
            ind = all(isnan(tmp), 1);
            tmp(:,ind) = []; 
            sMap.mask(ind) = [];
            sMap.codebook(:,ind) = [];
            [mqe, mte, cbe] = som_quality(sMap, tmp);
        end
        function [mqe, mte, cbe] = quality_pc(obj, data)
            sMap.codebook = MVSOM.combine_data(obj.get_node_weight());
            sMap.topol.msize = obj.dim;
            sMap.topol.type = 'som_topol';
            sMap.topol.lattice = 'rect';
            sMap.topol.shape = 'sheet';
            sMap.mask = ones(size(sMap.codebook, 2), 1);
            
            sMap.type = 'som_map';
            obj1 = obj.pca_all(data);
            for i = 1 : obj.num_var
                data1{i} = obj1.eof{i} * obj1.pc{i}' * obj1.pc_scale(i);
            end
            tmp = MVSOM.combine_data(data1,1)';
            ind = all(isnan(tmp), 1);
            tmp(:,ind) = []; 
            sMap.mask(ind) = [];
            sMap.codebook(:,ind) = [];
            [mqe, mte, cbe] = som_quality(sMap,tmp);
        end
    end
    %%
    methods (Static)
        % calculate correlation coefficient for each variable
        function [r,p,rm,rs] = get_corr_coef_var(data, result_class, nw)
            nv = obj.num_var;
            N = size(nw{1}, 1);
            np = size(data{1}, 2);
            for j = 1 : nv
                r{j} = nan(np, N);
                p{j} = nan(np, N);  % p-value
                for i = 1 : np
                    [R, P] = corrcoef(data(:,i), nw(result_class(i),:)');
                    r{j}(i,result_class(i)) = R(1,2);
                    p{j}(i,result_class(i)) = P(1,2);
                end
                rm{j} = nanmean(r{j}, 1);
                rs{j} = nanstd(r{j}, 0, 1);
            end
        end
        % calculate correlation coefficient for all variables combined
        function [r,p,rm,rs] = get_corr_coef(data, result_class, nw)
            data = MVSOM.combine_data(data,1);
            nw = MVSOM.combine_data(nw,2);
            [r,p,rm,rs] = get_corr_coef@SOM(data, result_class, nw);
        end
    end
    %%
    methods (Access=protected,Static=true)
        function data1 = combine_data(data, varargin)
            if length(data)>1
                if nargin == 1
                    if size(data{1},1)==size(data{2},1)
                        ax=2;
                    elseif size(data{1},2)==size(data{2},2)
                        ax=1;
                    else
                        error('size of input data are inconsistent');
                    end
                else
                    ax = varargin{1};
                end
                data1 = cat(ax, data{:});
            else
                data1 = data{1};
            end
        end
        function data = separate_data(data1, ne, varargin)
            nv = length(ne);
            cne = cumsum([0,ne]);
            n = size(data1);
            data = cell(1, nv);
            if nargin == 2
                if n(2) == cne(end)
                    ax=2;
                elseif n(1) == cne(end)
                    ax=1;
                else
                    error('wrong input data size');
                end
            else
                ax = varargin{1};
            end
            
            if ax==2
                for i = 1 : nv
                    data{i} = data1(:,cne(i)+1:cne(i+1));
                end
            elseif ax==1
                for i = 1 : nv
                    data{i} = data1(cne(i)+1:cne(i+1),:);
                end
            else
                error('input data in wrong size');
            end
        end
    end
    methods (Access=protected)
        function iw = get_node_weight(obj)
            if ~isempty(obj.net)
                nv = obj.num_var;
                iw = obj.net.IW{1};
                iw = MVSOM.separate_data(iw, obj.num_eof);
                for i = 1 : nv
                    iw{i} = iw{i} * obj.eof{i}' * obj.pc_scale(i);
                end
            else
                error('network is not trained!');
            end
        end
        function obj = pca_all(obj, data)
            nv = length(data); 
            obj.num_var = nv;
            obj.eof = cell(1, nv);
            obj.pc = cell(1, nv);
            obj.expl = cell(1, nv);
%             obj.num_eof = zeros(1, nv);
            for iv = 1 : nv
                [obj.eof{iv}, obj.pc{iv}, obj.expl{iv}] = ...
                    GeoData.get_eof(data{iv}, 'MinExpl', 95);%90);
%                 obj.num_eof(iv) = size(obj.eof{iv}, 2);
                ind_nan = isnan(obj.pc{iv});
                if sum(~ind_nan)>0
                    obj.pc_scale(iv) = sqrt(sum(obj.pc{iv}(~ind_nan).^2));
                    obj.pc{iv} = obj.pc{iv} / obj.pc_scale(iv);
                else 
                    error('error in rescaling PCs');
                end
            end
        end
        function num_eof = get_num_eof(obj)
            num_eof = zeros(1, length(obj.eof));
            for i = 1 : length(obj.eof)
                num_eof(i) = size(obj.eof{i}, 2);
            end
        end
    end
end

