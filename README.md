To create the simulation study, the following scripts were used in the given order:
1. SimStudy_Samples.py: Creates gaussian samples with a given precision matrix, sample sizes and number of samples per sample size.
2. SimStudy_zeros.Rmd: Iteratively performs the stepwise selection model for each sample created in the previous script, with LRT, p-value 0.01, 1000 iterations in a random search. The script updates the models with the RCOX function of the 'gRc' package and returns the updated precision matrices and updated log-likelihoods. Important file created: full_model_logL_results.csv, logL_results.csv, dimension_results.csv, K_matrices
3. HC_SimStudy.py: Performs hierarchical clustering on the partial variances and correlations separately for different threshold levels for all samples, where the edges have already been deleted. Returns the clustered VCC's and ECC's. Reads K_matrices and Returns Clustering_Results.
4. SimStudy_Symm.Rmd: Reads the VCC's and ECC's and updates the models with the RCOX model. Saves the final models. Reads Clustering_results and Simstudy_samples, returns files in final_model_results.csv
5. SimStudy_final.Rmd: Finds the log-likelihood and dimension of the final models. Will calculate the LR-statistic and calculate the p-value of accepting the model with symmetries against the model without symmetries. Also makes a summary of percentage of accepted models per threshold/sample size. saves to merged_model_results.csv
6. Acceptance.py: Calculates how many VCC's are correctly found
7. Edgecompare.py: Creates the tables on information for the edges found in the paper, such as total edges found, correct edges found, etc..
8. FullSymmcompare.py: Calculates the number of accepted models.

IMPORTANT FILES IN SIMULATION STUDY:
- full_model_logL_results.csv, contains the logL of the full model, edge deletion model and both dimensions. also contains the p-values.
- merged_model_results.csv, contains the logL of the symmetry model and edge deletion model and both dimensions. also tests them and finds the p-values. extracts information from final_model_results.csv

For the EEG-data processing and transformations the following scripts were used:
1. EPOC_Data_Creation.py: This file reads the EP1.01.txt file which contains all of the EPOC data and processes the data to a dataframe we can work with.
2. EEG_Transformation.py: Transforms the marginal distributions in the data to uniform distributions and then to  standard normal distributions.
5. MVN_Mardia.Rmd: Used to perform the Marida test for multivariate normality.
3. (NOT ADDED YET) EEG_MBD.Rmd: Performs edge deletion and symmetry calculations on the dataframe.
4. HC_EEG.py: Performs Hierarchical Clustering on the EEG model with deleted edges.

Other files:
- bep3dmulti.Rmd: This file was used to create the examples throughout chapter 2 and 3.
