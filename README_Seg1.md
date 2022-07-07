Team members: Elizabeth Ayisi, Lauren Triple, and Dursun Caliskan
## Segment 1 Project Overview

### Square Role - Liz

The team member in the square role will be responsible for setting up the repository. This includes naming the repository and adding team members.
Once team members are all on board, it will be your responsibility to ensure everyone has his or her own branch to work from. 
You can create branches for them or they can create their own. Either way, 
it's important to separate your work and to keep the main branch free from code in progress.

### Triangle Role - Dursun:

He is responsible for creating different machine learning models and comparing their accuracies.
Since the problem is prediction problem models will be supervised learning, linear regression models. They are
"Dummy Regressor", "Ridge", "Lasso", "DecisionTree", "AdaBoost", "Bagging", "GradientBoosting", "Random Forest" and "XGBoost Regressor"

### Circle Role - Lauren:

you're using a SQL-based database, including an ERD of the database and a document pointing out how it is integrated into your database and 
how it works with the code. You'll need to use either sample data or even fabricated data to test it. When you submit this database for your
weekly grade, make sure you're submitting the data used for testing as well. Make sure to upload it to the repository along with the rest of the 
database-related work.

### Selected Topic

We will be creating different models that predict the final price of homes in Ames, Iowa, and comparing their accuracies.


### Data
The data files provide categorical features about 1400 homes, most datatypes are string and integer. Number of rooms, square feet of rooms, dwelling, zoning, 
lot size, condition, and sale price are a few of the features.

### Database Storage

PostgreSQL will the the database. We used Google Colab, pandas and pySpark to explore the data, extract, transform, and load into the database.
The database was created using AWS, RDS.

** Systems:**

AWS, RDS, S3, PostgreSQL, spark, pyspark, google colab, pandas,numpy, seaborn,sklearn.impute, scipy, scipy.stats

**Files:**

Two csv files downloaded from Kaggle. The prices.csv was created in order to perform a join requirement for the project.

* [Resources/train.csv](Resources/train.csv)

* [Resources/test.csv](Resources/test.csv)

* [Resources/prices.csv](Resources/prices.csv)


### Summary

### Triangle Role completed the following tasks:

1. Got the acces to data base that Lauren created to ge the data neded for th eproject.


2. Determined the column which have more than 30 % of missing data, and droped these columns


3. Decresed the number of numerical features by finding pairwise correlation coefficients, drop one feature from ecah pair 
   which has corelation coefficients more than 0.8, that means these features are strongly dependent. Four futures are   droped in this step. 
   
   
4. Filled the missing data by using KNNInputer with proving that it is the most relevant method, by using distribution graphs before and after as follows:


5. Decresed the number of categorical features by combination of box and whisker plots and Chi-Squre test, as seen in the following figure:


5. Converted the categorical dataset to binary dataset.


6. Merged the numerical and categorical dataset

  You can find the script in the following link: 

 * Created colab notebook, (refer to file House_Prices.ipynb), for machine learning model and used boto and psycopg2 to pull in the data from the PostgreSQL database.
	![connect to data](https://user-images.githubusercontent.com/99093289/177671263-3ebdd12e-5dea-413f-90f4-05b4b2becf95.PNG)
* Turned the table data into dataframe
	![turn into dataframe](https://user-images.githubusercontent.com/99093289/177671317-1c3e3b91-0dcd-484a-bea1-795e8eb874cb.PNG)
	
* Began exploratory analysis of data inorder to prep for a successful machine learning model.
	
	
Square Role completed the following tasks:
* Named and created a GitHub repository with a clear README_Seg1.md
* Created individual branches for group collaboration.

![branches](https://user-images.githubusercontent.com/99093289/177673070-a42a6b6d-14f9-4259-8489-7ae32894438e.PNG)

Circle Role completed the following tasks:
  * Created PostgreSQL database through AWS RDS  
![RDS_database](https://user-images.githubusercontent.com/99093289/177668969-ea0c6b99-7e28-40af-87a8-635b75b1edf4.png)
  
  * Created IAM roles for group collaboration.
![IAMrole](https://user-images.githubusercontent.com/99093289/177668985-d616ed9f-0f50-42f4-9ad1-c52da15341c0.png)

  * Used pgAdmin to create the table schema in RDS.
  
	* House Data; 78 columns with Id as the primary key
	* Price Data; 2 columns with Id as foreign key
	* joined_data table: 80 columns, joining both data source files

  * Uploaded the CSV files to S3. 
  
![S3_data](https://user-images.githubusercontent.com/99093289/177669012-559034e8-e0b2-43e0-8885-0361bd12ada2.png)

  * Used Spark on Colab to clean and transform the data (refer to ipynb file housepricesdatabasedataload.ipynb).

![ColabCode_Join](https://user-images.githubusercontent.com/99093289/177669029-6a3f9418-a963-4efd-94b9-26f0c75198ac.png)


  * Load the data from Pandas DataFrames into RDS.
  
![pgadmin_joined_data](https://user-images.githubusercontent.com/99093289/177669088-1c133c8a-2e2f-4b99-93c1-5183dca1ca01.png)

  * Created ERD using pgadmin, however the clean joined_data table was created using Colab and Spark. 
  
![ERD pgerd](https://user-images.githubusercontent.com/99093289/177669126-d287936f-e01f-44a4-98a9-84f7dfec0e84.png)


### Attribution

The dataset was sourced from [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data].