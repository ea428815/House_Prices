Team members: Elizabeth Ayisi, Lauren Triple, and Durban Caliskan
### Segment 1 Project Overview

Square Role - Liz
The team member in the square role will be responsible for setting up the repository. This includes naming the repository and adding team members.
Once team members are all on board, it will be your responsibility to ensure everyone has his or her own branch to work from. 
You can create branches for them or they can create their own. Either way, 
it's important to separate your work and to keep the main branch free from code in progress.

Triangle Role - Dursun
The team member is responsible for creating a simple machine learning model. 
Creating a simple model this early in the design process helps a team better understand where and how a machine learning model will fit into the project. 
This also grants more time to work out the specifics related to machine learning. The first segment is all about preparation, so a simple model will cover the first question—the type of machine learning model chosen and why. 
To get started, create a simple model that isn't concerned with accuracy.

Circle Role - Lauren
The team member will be using a SQL-based database, including an ERD of the database and a document pointing out how it is integrated into your database and 
how it works with the code. You'll need to use either sample data or even fabricated data to test it. When you submit this database for your
weekly grade, make sure you're submitting the data used for testing as well. Make sure to upload it to the repository along with the rest of the 
database-related work.

### Selected Topic
We will be creating a model that predicts the final price of homes in Ames, Iowa. 

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


Square Role completed the following tasks:
	* Named and created a GitHub repository with a clear README.md
	* Created individual branches for group collaboration.
![ alt text for screen readers](base) Elizabeths-MacBook-Pro:houseprices elizabethayisi$ git branch
* main
(base) Elizabeths-MacBook-Pro:houseprices elizabethayisi$ git checkout -b Lauren-Tripple
Switched to a new branch 'Lauren-Tripple'
(base) Elizabeths-MacBook-Pro:houseprices elizabethayisi$ git branch
* Lauren-Tripple
  main
(base) Elizabeths-MacBook-Pro:houseprices elizabethayisi$ git checkout -b Dursun-Caliskan
Switched to a new branch 'Dursun-Caliskan'
(base) Elizabeths-MacBook-Pro:houseprices elizabethayisi$ git branch
* Dursun-Caliskan
  Lauren-Tripple
  main
(base) Elizabeths-MacBook-Pro:houseprices elizabethayisi$ 
	*Created the Google Slides for future use of the final presentation


Circle Role completed the following tasks:
  * Created PostgreSQL database through AWS RDS  
![ alt text for screen readers](C:\Users\ltipp\Documents\Lauren school\Data analytics bootcamp\Module20\Final\Images\RDS_database.png)
RDS_database.png
  * Created IAM roles for group collaboration.
![ alt text for screen readers](C:\Users\ltipp\Documents\Lauren school\Data analytics bootcamp\Module20\Final\Images\IAMrole.png)
IAMrole.png
  * Used pgAdmin to create the table schema in RDS.
	* House Data; 78 columns with Id as the primary key
	* Price Data; 2 columns with Id as foreign key
	* joined_data table: 80 columns, joining both data source files

  * Uploaded the CSV files to S3. 

  * Used Spark on Colab to clean and transform the data.
![ alt text for screen readers](C:\Users\ltipp\Documents\Lauren school\Data analytics bootcamp\Module20\Final\Images\ColabCode_Join.png)
ColabCode_Join.png
  * Load the data from Pandas DataFrames into RDS.
pgadmin_joined_data.png
  * Created ERD using pgadmin, however the clean joined_data table was created using Colab and Spark. 
![ alt text for screen readers](C:\Users\ltipp\Documents\Lauren school\Data analytics bootcamp\Module20\Final\Images\ERD.pgerd)
ERD.pgerd.png


## Google Slide
	*The Presentation will be in Google Slides and can be found at https://docs.google.com/presentation/d/1fbQLs9TsDw1QjUfXXw7MaDj-cBf5Nmi3FydxrKNgQAo/edit?usp=sharing

### Reference

The dataset was sourced from [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data].
