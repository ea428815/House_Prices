Team members: Elizabeth Ayisi, Lauren Triple, and Durban Caliskan
### Segment 1 Project Overview

Square Role - Liz
The team member in the square role will be responsible for setting up the repository. This includes naming the repository and adding team members.
Once team members are all on board, it will be your responsibility to ensure everyone has his or her own branch to work from. 
You can create branches for them or they can create their own. Either way, 
it's important to separate your work and to keep the main branch free from code in progress.

Triangle Role - Dursun: is responsible for creating a simple machine learning model. 
Creating a simple model this early in the design process helps a team better understand where and how a machine learning model will fit into the project. 
This also grants more time to work out the specifics related to machine learning.The first segment is all about preparation, so a simple model will cover the first question—the type of machine learning model chosen and why. 
To get started, create a simple model that isn't concerned with accuracy.

Circle Role - Lauren: you're using a SQL-based database, including an ERD of the database and a document pointing out how it is integrated into your database and 
how it works with the code. You'll need to use either sample data or even fabricated data to test it. When you submit this database for your
weekly grade, make sure you're submitting the data used for testing as well. Make sure to upload it to the repository along with the rest of the 
database-related work.

### Selected Topic
We will be creating a model that predicts the final price of homes in Ames, Iowa. 

### Data
The data files provide categorical features about 1400 homes, most datatypes are string and integer. Number of rooms, square feet of rooms, dwelling, zoning, 
lot size, condition, and sale price are a few of the features.

### Database Storage

PostgreSQL will the the database. We used Google Colab, pandas and pySpark to explore the data, extract, transform, and load into the database.
The database was created using AWS, RDS.

** Systems:**
AWS, RDS, S3, PostgreSQL, spark, pyspark, google colab, pandas


**Files:**

Two csv files downloaded from Kaggle. The prices.csv was created in order to perform a join requirement for the project.

* [Resources/train.csv](Resources/train.csv)

* [Resources/test.csv](Resources/test.csv)

* [Resources/prices.csv](Resources/prices.csv)


### Summary
Triangle Role completed the following tasks:

 * Created colab notebook and used boto and psycopg2 to pull in the data from the PostgreSQL database.
	![connect to data](https://user-images.githubusercontent.com/99093289/177671263-3ebdd12e-5dea-413f-90f4-05b4b2becf95.PNG)
* Turned the table data into dataframe
	![turn into dataframe](https://user-images.githubusercontent.com/99093289/177671317-1c3e3b91-0dcd-484a-bea1-795e8eb874cb.PNG)
	
* Began exploratory analysis of data inorder to prep for a successful machine learning model.
	
	
Square Role completed the following tasks:
* Named and created a GitHub repository with a clear README.md
* Created individual branches for group collaboration.

![ alt text for screen readers](base) Elizabeths-MacBook-Pro:houseprices elizabethayisi$ git branch

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

  * Used Spark on Colab to clean and transform the data.

![ColabCode_Join](https://user-images.githubusercontent.com/99093289/177669029-6a3f9418-a963-4efd-94b9-26f0c75198ac.png)


  * Load the data from Pandas DataFrames into RDS.
  
![pgadmin_joined_data](https://user-images.githubusercontent.com/99093289/177669088-1c133c8a-2e2f-4b99-93c1-5183dca1ca01.png)

  * Created ERD using pgadmin, however the clean joined_data table was created using Colab and Spark. 
  
![ERD pgerd](https://user-images.githubusercontent.com/99093289/177669126-d287936f-e01f-44a4-98a9-84f7dfec0e84.png)


### Attribution

The dataset was sourced from [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data].
