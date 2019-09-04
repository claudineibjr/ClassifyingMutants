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

 - The CSV files will be on the Programs/0Results folder named as 'mutants_equivalents.csv' and 'mutants_minimals.csv'
 - The CSV files for each program will be on the Programs/0Results/Programs folder named as 'mutants_equivalents_{SessionName}.csv' and 'mutants_minimals_{SessionName}.csv'
 - Copy CSV files to folder ML/Mutants/
    - Replace ';' for ',' on CSV Files
    - On 'ML/Mutants/{Equivalent or Minimal}' copy the CSV File
    - Remove {'_equivalents' or '_minimals'} from File Names

> python3 ML/ML_Mutants.py --column {targetColumn} --classifier {classifier}
 - For each ML algorithm and classification (minimal or equivalent) will be created a CSV File
 - The {targetColumn} value could be '_IM_MINIMAL' or '_IM_EQUIVALENT'
 - The {classifier} value could be 'KNN', 'DT', 'RF' or 'SVM'