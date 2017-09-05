# Zaggle

## Plan:

0. Prepare data. Use one hot encoding for categorical variables (objects and numeric that are categorical). Use -1 to fill NAs?
1. Implement XGB, LinReg, and neural network models. Ensemble and submit.
2. Feature engineer for XGB?
3. Add other data sources? Will be allowed after first round, I think.


Some notes:
- All the ID variables are coded quantiative but shouldn't be interpretated as such. Even some variables in the data dictionary (e.g. 'airconditioningtypeid') should not be coded as numeric. 
- The object variables should be coded as categorical. Input -1 for missing?

To do:
- Determine how to easily run tests with varying parameters on my CV?

- Add these created variables to XGB:

https://www.kaggle.com/nikunjm88/creating-additional-features?scriptVersionId=1379783
