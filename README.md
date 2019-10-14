# An approach to identifying minimal and equivalent mutants from relevant mutant features and properties

Mutation Testing
-----------------

This project presents a tool that apply an approach that aims to reduce the cost of Mutation Testing.

Our work intends to collect data from mutants and use Machine Learning Algorithms to classify a new mutant (without classification) as minimal, or not, as equivalent, or not.

Step by Step
-------------
 - Execute Proteum to generate and execute mutants and generate informations about them (status, program graph node, operator, offset, etc)
 - Gather generated informations in CSV Files
 - Import CSV files into ML algorithms
 - Preprocess data before classify the mutants
 - Train ML algorithms
 - Evaluate ML algorithms

How to run 
-----------
> python3 experiment.py executionMode
 - executionMode =  1   Run Proteum and Analyze Data
                    2   Just run Proteum
                    3   Just analyze

> python3 ML/ML_Mutants.py {parameters} --column {targetColumn} --classifier {classifier}
 - The parameters for execution are:
    --all - Execute all programs, all target columns and all classifiers.
    --allPbp - Execute all target columns, all classifiers and all programs, but with one execution for each one.
    --column | The targetColumn to be classified. Could be 'MINIMAL' or 'EQUIVALENT'
    --classifier | The classifier used to classify. Could be 'KNN' for K Nearest Neighbors, 'DT' for Decision Tree, 'RF' for Random Forest, 'SVM' for Support Vector Machine, 'LDA' for Linear Discriminant Analysis, 'LR' for Logistic Regression and 'GNB' for Gaussian Naive Bayes
    --program | The specified program to classify the target column
    --pbp | Execute program by program
    --best | Indicating that will be execute the classifiers with the best parameters

 - Possible executions
    --all
    --all --best
    --allPbp
    --allPbp --best
    --column {column} --classifier {classifier}
    --column {column} --classifier {classifier} --best
    --column {column} --classifier {classifier} --program {programName}
    --column {column} --classifier {classifier} --program {programName} --best
    --column {column} --classifier {classifier} --pbp
    --column {column} --classifier {classifier} --pbp --best

 - For each ML algorithm and classification (minimal or equivalent) will be created a result CSV File