To create the simulation study, the following scripts were used in the given order:
1. SimStudy_Samples.py: creates gaussian samples with a given precision matrix, sample sizes and number of samples per sample size.
2. SimStudy_zeros.Rmd: iteratively performs the stepwise selection model for each sample created in the previous script, with LRT, p-value 0.05, 100 iterations in a random search. The script updates the models with the RCOX function of the 'gRc' package and returns the updated precision matrices and updated log-likelihoods.
3. HC_SimStudy.py: Performs hierarchical clustering on the partial variances and correlations separately for different threshold levels for all samples, where the edges have already been deleted. Returns the clustered VCC's and ECC's.
4. SimStudy_Symm.Rmd: Reads the VCC's and ECC's and updates the models with the RCOX model. Saves the final models.
5. SimStudy_final.Rmd: Finds the log-likelihood and dimension of the final models. Will calculate the LR-statistic and calculate the p-value of accepting the model with symmetries against the model without symmetries. Also makes a summary of percentage of accepted models per threshold/sample size.
6. Acceptance.py: Calculates how many VCC's are correctly found
7. Edgecompare.py: Creates the tables on information for the edges found in the paper, such as total edges found, correct edges found, etc..
