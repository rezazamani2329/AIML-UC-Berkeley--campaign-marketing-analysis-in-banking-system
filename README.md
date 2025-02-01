# Big Picture of the Project and its process:  

### 1- Using AIML to evaluate marketing strategy

Suppose you take part in a meeting to evaluate campaign marketing strategy in banking industry

Speaker represents the strategy and now is your turn to give suggestion 

You can suggest this: Oh, let see what AI says. 

### 2- Structure designer of the project and Python code writer 
Here is the answer of AIML for your suggestion, please follow it. You will find some good ideas.

This code has user friendly structure. It is classification issue in machine learning and designed by `UC Berkeley`, from `Berkeley Engineering` and `Berkeley HAAS` and written by `Reza Zamani`.

### 3- Process of the project
We will go step by steps (in `11 problems`). Generally, these steps can be seen in following sections:

1- **Data understanding, freature engineering and visualization** : From `problem 1 to problem 3, and in porblem 11`

2- **Business understanding**: `problem 4`

3- **Baseline and simple Models** (to get general clue): from `problem 5 to problem 9`

4- **Other models and model comparison** (evaluation, choosing the best model): `problem 10`

5- **best parameters** for each classifier with gridsearch and choosing the best model among them: `problem 11`

6- **feature importance by best model**: `problem 11`

7- **Suggestions**: last part (`after problem 11`) 


### 4- **Order of actions**

1- **Data udnerstanding**

2- **Business udnerstanding**

3- **Methodology: classification**

4- **Visualizatio and EDA**

5- **Baseline and simple models**

6- **Performance evaluation of clasiffiers with default parameters**

7- **Finding best model with default parameters**

8- **Accurate Feature engineering and EDA: multiple correlation analysis**

9- **Using Grid Search to find best parameters for each classifer**

10- **Performance evaluation of clasiffiers with best parameters**

11- **Finding best model** among different calissifers with best parameters

12- **Featuer Importance with best model**

13- **Cross check** of feature improtance with EDA and feature engineering

14- **Clear suggestions** with attention to business undestanding




# Project
What derives the success rate of campaign marketing strategy in banking system. 

# OVERVIEW
In `Portugal`, as our case study, for the period of `2008-2010`, a marketing approach defiend and applied by `17 groups` as member of campaign. each member who taked part in this marketing called people and tried to to convince them to depostie their money in the bank.


In This project, I apply machine learning algorithms to understandthe how factors (personal, conatact related issues, economic and social condition of society and other factors) affect the succes rate of campaign marketing in banking system. 

# Business Understanding

### 1-Problem Definition:
During 2008-2010, the world experienced financial crieses, and there was a huge pressure on European banks to increase financial asset. To solve this issue, campaign marketing in Portugal as a case study of this project started. They created different campaigns.

### 2- Strategy:
strategy was to offer attractive long-term deposit applications with good interest rates, particularly by using directed marketing campaigns. Thus, there is a need for improvement in efficiency: lesser customer contacts must be done, but same success rate (clients subscribing to the deposit) must be maintained.

### 3- Efficinecy issue:
In this perspective bank can increase the efficiency of its marketing campaigns, reduce costs, and improve conversion rates, ultimately leading to better customer targeting and higher profitability

### 4- Business objective:
Business objective of this poroject is to predict whether a client will subscribe to a term deposit based on various client, campaign, economic and social indicators.

# Data Understanding (general)
1- We have 41k data with 21 factors: one target and 20 features ( 10 numerical, 10 categorical). 

2- `Target variable`; It represents the final achievement of calling each person. Data type of target variable is object, symbole of yes' represents that campaign was successful in marketing, and 'no' shows failure.  

3- `Features`: they can be devided as following: 
 a- Bank client data: 7 features including age, job, employed, marital status, education, default (credit), housing loan, personal loan
 b- facfors related with the last contact of the current campaign: 4 features inclusing contact, month, day, and duration.
 c-other attributes: 4 features including campaign, pdays, previous and poutcome. 
 d- Social and economic context attributes: 5 features including employment variation rate, number of imployment, confidence, cpi and interest rate. 
 here is total list wit detial: 
 
### - Client Data:
  - `age`: Age of the client
  - `job`: Job type (categorical)
  - `marital`: Marital status (categorical)
  - `education`: Education level (categorical)
  - `default`: Has credit in default? (categorical)
  - `housing`: Has housing loan? (categorical)
  - `loan`: Has personal loan? (categorical)
  
### - Contact Data:
  - `contact`: Communication type (categorical)
  - `month`: Last contact month (categorical)
  - `day_of_week`: Last contact day of the week (categorical)
  - `duration`: Last contact duration in seconds (numeric)
  
### - Campaign Data:
  - `campaign`: Number of contacts performed during this campaign (numeric)
  - `pdays`: Number of days since the client was last contacted (numeric)
  - `previous`: Number of contacts before this campaign (numeric)
  - `poutcome`: Outcome of the previous campaign (categorical)
  
- Economic Indicators:
  - `emp.var.rate`: Employment variation rate (numeric)
  - `cons.price.idx`: Consumer price index (numeric)
  - `cons.conf.idx`: Consumer confidence index (numeric)
  - `euribor3m`: Euribor 3 month rate (numeric)
  - `nr.employed`: Number of employees (numeric)

- Target Variable:
  - `y`: Subscription to a term deposit (binary: 'yes' or 'no')


- Points:
- `unknown` values in categorical variables such as `job`, `education`, `default`, `housing`, and `loan` represent missing data.
- `999` in `pdays` indicates that the client was not previously contacted.


# Data visualization, EDA and feature engineering  

-Missing data, outlier and buplicated data are checke. 
- Handling Missing Values:
   - `No missing values`. 

- Data Cleaning:
   - `Removed 12 duplicate rows from the datase`.

- Numerical features:
Nnumerical features are: age, duration, campaign, pdays, previous, employment variation ('emp_var'), cpi, confidence ('conf'), interest rate ('euribor3m') and number of employees ('nr_emp').
-For numerical features, we check disribution of each feature (with boxplot, violinplot, hitogram and so on), its relationship with target variable and finally cehck correlation between them (with correlation function and heatmap). see problem 3 for details. 

- Correlation Analysis for numerical features
    - A correlation matrix was calculated for the numerical features to understand the relationships between variables. Some key insights include:
      - `pdays` and `previous` show a negative correlation. 
      - `emp.var.rate`, `euribor3m`, and `nr.employed` have strong positive correlated with each other. 
      - `duration` has a minimal correlation with other features but is crucial in predicting the target variable.

- Categorical features:
-Categorical features are: 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', and 'poutcome'
-For categorical data we check distribution of each feature (with counplot), their effect on target variable with heatmap, and their relationship with success rate(  For more detail see problem 3.) 
  - `day_of_week`: there are 5 days and all of the are around 20%, then thid feature can not help the model.
  - `poutcome`: 84% of this feature is nonexistence, then we can remover it too 



# First Round of Modeling: 
1- what kind of maching learing model? as our target variable is object ('yes' or 'no'), we have classification, and as target variable is `imbalance ('no'= 89%, 'yes'= 11%)`, then we have `imbalance` dataset.

2-Feature enginering in first round: I remove one numerical featue with maximum correlation. now we have 9 numerical feature and 10 categorical. 

3- Preprocessor and encoding:
-use `LabelEncoder()` for target variable 
-use `StandardScaler()` for numerical variables.
-use `OneHotEncoder()` for categorical variables. 

4- `Baseline model is DummyClassifier`. why we use it? if you want to predict, without more attemp, what is your prediction? with attention to struture of target variable, if you predict 'no' you get 89% accuracy, but as your prediction for 'yes' is wrong, your recall and precision is 0% and then your AUC is 50%. Dummy classification does exacly that. For more detial see problem 7: 

5- `Simple model`: we use logistic regression (`LogisticRegression()`) where all hyperparameters are define. score of the model reaches to 91% and very sharp improvement achieved in precision and recall for target = 1(or 'yes' before labelencoding). This shows that simple model has changed the quality of prediction sharply. For detail see problems 8 and 9:
      
6- Adding `other models`: we use `decision tree`, `KNN` and `SVM` models too. We check calssification report for all of them, and comapre them. 

7- `Evaluation`: Moreover, we compare train score, test score and average fit time, here is the result: 

 	                 Train Score 	Test Score 	Average Fit Time 	Precision 	Recal 	f1
Model 						
Decision Tree 	        1.00 	     0.89 	         1.32             0.52 	    0.51 	0.51
KNN 	                0.93 	     0.90 	         0.21 	          0.59 	    0.44 	0.50
SVC 	                0.92 	     0.91 	        33.48 	          0.67 	    0.43 	0.53
Logistic Regression 	0.91 	     0.91 	         0.42 	          0.66 	    0.42 	0.51


8- `Best model in first round of modeling (SVM)`: with attention to all criteria (train score, test score, time, recall, precision and f1), `SVM` is best model and second best model is logistic regression. see evaluation is 10-2-model comparison and 10-3-best model. 

# Second round of modeling: 
1- `More Feature enginering`: I remove two numerical featue with maximum correlation and remove two categorical features. Now we have 7 numerical and 8 categoricalfeatures.  

2- `Preprocessor and encoding:`
-use `LabelEncoder()` for target variable 
-use `StandardScaler()` for numerical variables.
-use `OneHotEncoder()` for categorical variables.

3- using `GridSearchCV` for tuning the hyperparameters: we define differtent hyperparameters for each classifier and define pipeline with preprocessor, gridsearch for each classifier. 

4- `Best parameters` for each classifier are found: 

`Best Params for  Decision Tree classification`              : {'model__criterion': 'gini', 'model__max_depth': 3}
`Best Params for  KNN classification`                        : {'model__n_neighbors': 2, 'model__p': 1, 'model__weights': 'uniform'}
`Best Params for SVM classification`                         : {'model__C': 0.1, 'model__kernel': 'rbf'}
`Best Params for Logistic Regression classification`        : {'model__C': 1, 'model__solver': 'lbfgs'}

5- `Pipeline` with best parameters are defined to compare classifiers `user friendly`.

6- `Evaluation`: evaluatin is based on accuracy score, time, calssification report (recall, precision, f1), confusion matrix and ROC curve. here are the resutls, for more information see 11-5-2 and 11-5-3 sections.
 

 	                 Accuracy 	    Precision 	Recall 	    F1-Score 	ROC-AUC
KNN 	                0.895484 	0.597101 	0.221983 	0.323645 	0.789870
Logistic Regression 	0.913693 	0.696203 	0.414871 	0.519919 	0.939016
Decision Tree 	        0.912843 	0.614379 	0.607759 	0.611051 	0.878884
SVM 	                0.908109 	0.693878 	0.329741 	0.447042 	NaN

7- `best model: logistic regression` is best model. see our evaluation in 11-5 section of pythone code

### Feature importance in best model: 

1- `Duration`:
it is the most improtant factor affecting positively the result of direct marketing. Times goes up, success probability goes up too.  

2- `Interest rate`
Interest rate is opportunity cost of money. Higher interest rate if bank pay for people they will deposit more. 

3- `Month`
End of each season has the highest level of success. March and June have the highest rate of success among month. on the other hand, the last maonth of year campaign are off. 

4- `Job`
Highest rate of success comes back to retired, followd by sutdnents and unemployed. retired can get more money from deposite as they are risk averse in comparison to people how are working. Illiterate people alos accept better the suggestion of campagin, maybe their information is low and they do not check the alternatives. Blue-collar, entreprenuer, self-employed, servcies, management and technician have negative effect on the rate of success, respectively.  

5- `Marital satatus`
It is not important.  

6- `loan satatus`
having loan is not important.  

7- `previous contant satatus`
It has negative effect on success, maybe they are more familiar with the strategy of contact. 

8- `Pdays contant satatus`
It has negative effect on success. it shows that if you as a memebr of campaingn starts a calling one person, try to reach the result and do not postpone it for long days.  

9- `Housing loan`
It has negative effect, but its coefficient is low. It shows that who has hosing loan maybe does not have extra money to deposite in the bank.   </span>

 10- `age`
It has positive effect, but its coefficient is low.

 11- `contact`
telephone has negative effect, cellular is better.

 12- `campaign`
if the number of calling increase, the level of success will decrease. we should complete this information with our data analysis in problem thee. There we represted that after three call, it is completly true.</span>

 13- `confidence`
it has positive effect, which shows when people are hopeful to the future they are more eager to accept the suggestion of the campaign .</span>

 14- `Education`
university, basic6y, illiterate, and professional has positive effect, and high school and basic 9y have negative effect.   </span>


# Suggestions and Recommendations to improve success rate 
## A- Bank client factors: 
 1-`Age`:
It is not important factor. Do not pay attention to that. 

 2-`job`:
 1. Focus on these groups: Retired student, and unemployed,  
2. Risk lovers have low success rate, try do not waste your time with them. I mean these groups: blue-collar, entrepreneur, self-employed, services, management, and technician, respectively

3- `Maritual`:
It is not important factor. Do not pay attention to that. 

4- `Education`:
 1. Devid people in three groups: low, middle and high education.
 2. Focus on people with low education or high education, they have high rate of success 
 3. People with middle education have low rage of success, do not focus on them. 

5- `Housing`:
 It has little negative effect, but generally it is not important.  

6- `Loan`: 
 It has little negative effect, but generally it is not important.  

##  B- Factors related with the last contact:
7- `Contact`:
Call people with cellular, try do not call them with telephone. 

8- `Month`:
 1. Last month of each season is vital, focus on them. 
 2. Never loose March. 
 3. First and last months of the year you can go vacations. 

 9- `Duration`:
  1. It is the most important numerical factor. Focus on it.  
 2. Golden duration of a call is 3-5 minutes. try talk to people at least 3 minues. 
 3. If your call time reaches to 15 minutes, try to finish it. 


#  C- Social and economic context attributes:

10- `interest rate`:
 Try to increase marketing in lower interest rate. It is also indices for macroeconomic expectations.  

11- `Confidence level`:
Try to increase marketing in higher confidential levels of society. 
 
## D- Other attributes: 

12- `Campaign`:
 1. The average number of contact is around 2.5, then 2 or 3 times calling is good. 
 2. Keep in mind that as the number of calling goes up (more than 3 time), the level of success goes down. 

13- `pdays`:
 If you, as a member of marketing campaign, start to call a person, try to reach the result and do not postpone it for long days. 

14- `previous`:
 It decrease the chance of success, if you found that they have previous call for several time, it means your task is hard and should try harder or finish it soon. 


