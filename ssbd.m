function ssbd()
clear all ; close all ; clc ;
%% 
% This is an example program for the recognition of three stimming classes
% (armflapping, headbanging, spinning) in Self-Stimulatory Behaviour
% Dataset (SSBD). 

% Input : None. The dataset is automatically downloaded from the web
% Output : Classification Accuracy results
% Dependencies : This program uses the functionality from VLFeat library.

% Install VLFeat library from http://www.vlfeat.org/. Thanks to Andrea Vedaldi 
% for VLFeat library and also for the example application code phow_caltech101.m. 
% This program (ssbd.m) is created using phow_caltech101.m as the starting point.

% Please refer to our paper "Self-Stimulatory Behaviours in the Wild for
% Autism Diagnosis" (http://staff.estem-uc.edu.au/roland/files/2009/05/
% Rajagopalan_Dhall_Goecke_ICCV2013_DSCSI_Self-StimulatoryBehavioursInTheWildForAutismDiagnosis.pdf)
% for details about the dataset used by this program. 


% Author : Shyam Sundar Rajagopalan (Shyam.Rajagopalan@canberra.edu.au)
% Please contact the author for any assistance

    %%  Macros
    SETUP = 1 ;
    DATA_DIR = './data' ;
    DATA_URL = 'https://googledrive.com/host/0B4_UYih-LBmrVlg2U2xQYkpCTEU/ssbd-stip.zip';
    

    main();
    fprintf ('Completed Successfully \n') ;


    %% Nested functions
    function main()

        %%  Setup - Download stip files and copy it to local folders. Three sub-folders will be created within current folder.
        % This portion of the code downloads the stip files corresponding to all
        % the 75 videos in three classes (armflapping, headbanging, spinning) from
        % the web and copies it to local folders. The total size of the stip files
        % is approximately 4.58GB and hence the download might take time. Do not
        % proceed to next steps, until this step is successfully completed as this is
        % the dataset used by subsequent portion of the code.
        if (SETUP)
            %% Add VLFeat library paths and run the vl_setup program
            %addpath('<Path To VLFeat Install Folder>/vlfeat-0.9.17/') ;
            addpath('./vlfeat-0.9.17/toolbox/') ;
            vl_setup 
            
            % Download STIP files
            if  ~exist(DATA_DIR,'dir') 
                mkdir(DATA_DIR) ;
                fprintf('Downloading stip files of size around 4.5GB. It might take a while ... \n') ;
                unzip(DATA_URL, DATA_DIR) ;
                fprintf('Download complete and the contents are extracted into %s\n', DATA_DIR) ;
            end
            
            fprintf('Setup Complete ! \n') ;
            
            
            
        end  % if (SETUP)
        %%



        fprintf ('Starting stimming behaviour learn and recognition \n');
        %% Initialization of variables and test sets
        conf.calDir = DATA_DIR ;
        conf.dataDir = DATA_DIR ;
        conf.autoDownloadData = true ;
        conf.numTrain = 20 ;
        conf.numTest = 5 ;
        conf.numClasses = 3 ;
        conf.numWords = 500 ;
        conf.numSpatialX = [2 4] ;
        conf.numSpatialY = [2 4] ;
        conf.quantizer = 'kdtree' ;
        conf.svm.C = 10 ;

        conf.svm.solver = 'sdca' ;
        %conf.svm.solver = 'sgd' ;
        %conf.svm.solver = 'liblinear' ;

        conf.svm.biasMultiplier = 1 ;
        conf.phowOpts = {'Step', 3} ;
        conf.clobber = true ;
        conf.tinyProblem = true ;
        conf.prefix = 'ssbd' ;
        conf.randSeed = 1 ;

        if conf.tinyProblem
          conf.prefix = 'tiny' ;
          conf.numClasses = 3 ;
          conf.numSpatialX = 2 ;
          conf.numSpatialY = 2 ;
          conf.numWords = 500 ;
          conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5} ;
        end

       

        randn('state',conf.randSeed) ;
        rand('state',conf.randSeed) ;
        vl_twister('state',conf.randSeed) ;

        % Setup Train and Test data
        classes = dir(conf.calDir) ;
        classes = classes([classes.isdir]) ;
        classes = {classes(3:conf.numClasses+2).name} ;

        images = {} ;
        imageClass = {} ;
        for ci = 1:length(classes)
          ims = dir(fullfile(conf.calDir, classes{ci}, '*-stip.txt'))' ;
          ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
          ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
          images = {images{:}, ims{:}} ;
          imageClass{end+1} = ci * ones(1,length(ims)) ;
        end
        imageClass = cat(2, imageClass{:}) ;


        model.classes = classes ;
        model.phowOpts = conf.phowOpts ;
        model.numSpatialX = conf.numSpatialX ;
        model.numSpatialY = conf.numSpatialY ;
        model.quantizer = conf.quantizer ;
        model.vocab = [] ;
        model.w = [] ;
        model.b = [] ;
        model.classify = @classify ;

%       Enable for caching stip files - Might need more memory around 8GB RAM        
        global stipMap ;
        stipMap = containers.Map() ;


        nFolds = 5 ; % Number of validation folds. 
        N = 75; % Total number of video(stip) files in all three classes
        totalAccuracy  = 0.0 ;
        accuracy = zeros(nFolds) ;

        % Example test set. 5 videos (stip files) are used for test and the
        % remaining for training. 
        test(1,:) = [1,2,3,4,5,26,27,28,29,30,51,52,53,54,55] ;
        test(2,:) = test(1,:) + 5 ;
        test(3,:) = test(1,:) + 10 ;
        test(4,:) = test(1,:) + 15 ;
        test(5,:) = test(1,:) + 20 ;
        
        % Enable this for different set of 5 folds. 
%         test(1,:) = [2,3,4,5,6,27,28,29,30,31,52,53,54,55,56] ;
%         test(2,:) = test(1,:) + 5 ;
%         test(3,:) = test(1,:) + 10 ;
%         test(4,:) = test(1,:) + 15 ;
%         test(5,:) = [3,4,5,6,7,28,29,30,31,32,53,54,55,56,57] ;

        %%
        for v = 1 : nFolds
           % Obtain the test and training test
           selTest = test(v,:) ;
           selTrain = setdiff(1:length(images),selTest) ;
           strVocab = sprintf('-vocab%2d.mat',v) ;
           strHist = sprintf('-hists%2d.mat',v) ;
           strModel = sprintf('-model%2d.mat',v) ;
           strResult = sprintf('-result%2d',v) ;
           
           conf.vocabPath = fullfile(conf.dataDir, [conf.prefix strVocab]) ;
           conf.histPath = fullfile(conf.dataDir, [conf.prefix strHist]) ;
           conf.modelPath = fullfile(conf.dataDir, [conf.prefix strModel]) ;
           conf.resultPath = fullfile(conf.dataDir, [conf.prefix strResult]) ;
           

           %% Train the vocabulary
            if ~exist(conf.vocabPath) || conf.clobber
              
              % Get some STIP descriptors to train the dictionary
              selTrainFeats = vl_colsubset(selTrain, 30) ;
              descrs = {} ;
              for ii = 1:length(selTrainFeats)
              %parfor ii = 1:length(selTrainFeats)
                stipFile = fullfile(conf.calDir, images{selTrainFeats(ii)}) ;
                fprintf ('Validation Run %d - Train Vocabulary - Importing %s \n',v, stipFile) ;
                [~,descrs{ii}] = ReadStipFile(stipFile) ;
              end

              descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
              descrs = single(descrs) ;

              % Quantize the descriptors to get the visual words
              vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
              save(conf.vocabPath, 'vocab') ;
            else
              load(conf.vocabPath) ;
            end
            model.vocab = vocab ;
            clear descrs ;

            if strcmp(model.quantizer, 'kdtree')
              model.kdtree = vl_kdtreebuild(vocab) ;
            end


            %% Compute spatial histograms
            if ~exist(conf.histPath) || conf.clobber
                fprintf ('Validation Run %d - Compute Spatial Histograms',v) ;
                hists = {} ;
                %parfor ii = 1:length(images)
                for ii = 1:length(images)
                    fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;

                    videoStipFile = fullfile(conf.calDir, images{ii}) ;
                    h= getVideoDescriptor(model,videoStipFile) ;
                    hists{ii} = h ;

                    clear h ;
                end

                hists = cat(2, hists{:}) ;
                save(conf.histPath, 'hists') ;
            else
                load(conf.histPath) ;
            end
            % Compute feature map
            psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;


            %% Train SVM
            if ~exist(conf.modelPath) || conf.clobber
              fprintf ('Validation Run %d - Train SVM \n', v) ;
              switch conf.svm.solver
                case {'sgd', 'sdca'}
                  lambda = 1 / (conf.svm.C *  length(selTrain)) ;
                  w = [] ;
                  for ci = 1:length(classes)
                    perm = randperm(length(selTrain)) ;
                    fprintf('Training model for class %s\n', classes{ci}) ;
                    y = 2 * (imageClass(selTrain) == ci) - 1 ;
                    [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
                      'Solver', conf.svm.solver, ...
                      'MaxNumIterations', 50/lambda, ...
                      'BiasMultiplier', conf.svm.biasMultiplier, ...
                      'Epsilon', 1e-3);
                  end

                case 'liblinear'
                  svm = train(imageClass(selTrain)', ...
                              sparse(double(psix(:,selTrain))),  ...
                              sprintf(' -s 3 -B %f -c %f', ...
                                      conf.svm.biasMultiplier, conf.svm.C), ...
                              'col') ;
                  w = svm.w(:,1:end-1)' ;
                  b =  svm.w(:,end)' ;
              end

              model.b = conf.svm.biasMultiplier * b ;
              model.w = w ;

              save(conf.modelPath, 'model') ;
            else
              load(conf.modelPath) ;
            end

            %% Test SVM and evaluate
            fprintf('Validation Run %d - Test SVM \n',v) ;
            % Estimate the class of the test videos
            scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
            [drop, imageEstClass] = max(scores, [], 1) ;

            % Compute the confusion matrix
            idx = sub2ind([length(classes), length(classes)], ...
                          imageClass(selTest), imageEstClass(selTest)) ;
            confus = zeros(length(classes)) ;
            confus = vl_binsum(confus, ones(size(idx)), idx) ;

            % Plots
            figure(1) ; clf;
            subplot(1,2,1) ;
            imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
            set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
            subplot(1,2,2) ;
            imagesc(confus) ;
            title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
                          100 * mean(diag(confus)/conf.numTest) )) ;
            print('-depsc2', [conf.resultPath '.ps']) ;
            save([conf.resultPath '.mat'], 'confus', 'conf') ;
            accuracy(v) = 100 * (mean(diag(confus)/conf.numTest) );
            fprintf ('Validation Run = %d, Accuracy = %2.2f\n',v,accuracy(v)) ;
            totalAccuracy = totalAccuracy + accuracy(v) ;

        end % for v = 1 : nFolds


        %% Display overall statistics
        for v = 1 : nFolds
            fprintf ('Fold = %d, Accuracy = %2.2f\n', v,accuracy(v)) ;
        end
        meanAccuracy = totalAccuracy / nFolds ;
        fprintf ('\nMean Accuracy is %2.2f over %d folds\n',meanAccuracy, nFolds) ;
    end

    
    function hist = getVideoDescriptor(model, video)
        % get the xpath mechanism into the workspace
        import javax.xml.xpath.*
        factory = XPathFactory.newInstance;
        xpath = factory.newXPath;
        
        xmlFilePath = strrep(video,'-stip.txt','.xml') ;
        [idx] = strfind(xmlFilePath,'v_') ;
        xmlFile = xmlFilePath(idx:length(xmlFilePath)) ;
        AnnFilePath = ['./Annotations/' xmlFile] ;
        docNode = xmlread(AnnFilePath);
        % height
        expression = xpath.compile('video/height');
        node = expression.evaluate(docNode, XPathConstants.NODE);
        height = str2num(node.getTextContent);
        % width
        expression = xpath.compile('video/width');
        node = expression.evaluate(docNode, XPathConstants.NODE);
        width = str2num(node.getTextContent);
        % frames
        expression = xpath.compile('video/frames');
        node = expression.evaluate(docNode, XPathConstants.NODE);
        nFrames = str2num(node.getTextContent);

        numWords = size(model.vocab, 2) ;


        [frames, descrs] = ReadStipFile(video) ;
        % quantize local descriptors into visual words
        switch model.quantizer
          case 'vq'
            [drop, binsa] = min(vl_alldist(model.vocab, single(descrs)), [], 1) ;
          case 'kdtree'
            binsa = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                                          single(descrs), ...
                                          'MaxComparisons', 50)) ;
        end

        for i = 1:length(model.numSpatialX)
          binsx = vl_binsearch(linspace(1,width,model.numSpatialX(i)+1), frames(1,:)) ;
          binsy = vl_binsearch(linspace(1,height,model.numSpatialY(i)+1), frames(2,:)) ;

          % combined quantization
          bins = sub2ind([model.numSpatialY(i), model.numSpatialX(i), numWords], ...
                         binsy,binsx,binsa) ;
          hist = zeros(model.numSpatialY(i) * model.numSpatialX(i) * numWords, 1) ;
          hist = vl_binsum(hist, ones(size(bins)), bins) ;
          hists{i} = single(hist / sum(hist)) ;
        end
        hist = cat(1,hists{:}) ;
        hist = hist / sum(hist) ;
    end


    % Read the Stip Files and retrun the HOG/HOF descriptors
    function [frames, descrs] = ReadStipFile(video) 
       frames = [] ; descrs = [] ;
       % Use the global map to cache the files to enable faster reading for
       % various validation runs. However, this might need more memory,
       % around 8GB
       
       global stipMap ;
       keyExists = isKey(stipMap,video) ;
       if (keyExists)
           A = stipMap(video) ;
       else
            A = importdata(video) ;
           stipMap(video) = A ;
       end
        B = A(1).data ;
        descrs = B(:,10:171)' ;
        frames = B(:,5:6)' ;
        frames([1 2],:) = frames([2 1],:) ;
        clear A ;
        clear B ;
    end


    function [className, score] = classify(model, stipVideoFile)
        % hist = getImageDescriptor(model, im) ;
        hist = getVideoDescriptor(model, stipVideoFile) ;
        psix = vl_homkermap(hist, 1, 'kchi2', 'period', .7) ;
        scores = model.w' * psix + model.b' ;
        [score, best] = max(scores) ;
        className = model.classes{best} ;
    end

       
end