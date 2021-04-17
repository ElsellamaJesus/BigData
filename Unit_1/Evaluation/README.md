# Evaluation
--- 
## Exam Unit 1 - Big Data
- Import libraries
```scala
import org.apache.spark.sql.SparkSession
import spark.implicits._
```
**1. Start a simple Spark session.**
 ```scala 
val spark = SparkSession.builder().getOrCreate()
```
**Explanation:** In point 1, we were asked to start a session in Spark, we started by importing the sql library to start the session, then we imported another library that will serve us later with the operations with implicit data and finally we assign to a variable the session created.\

**Result**
<html><div align="center"><img src="https://i.ibb.co/Xt582DY/Cap1.png" alt="Cap1" border="0"></div></html>



**2. Load Netflix Stock CSV file, have Spark infer data types.**
```scala
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
```
**Explanation:** In point 2, we append the dataframe to be used, assign it to a variable that we call "df (dataframe)" and infer the data. It should be noted that the CSV file was in the same folder as the scala file, so the path was not specified.

**Result**
<html><div align="center"><img src="https://ibb.co/GJmVm44"></div></html>



**3. What are the column names?**
```scala
df.columns
```
**Explanation:** In point 3, with a property called ".columns", we only send the names of the columns of our dataframe.

**Result**
<html><div align="center"><img src="https://ibb.co/WyGwbXd"></div></html>



**4. What is the schematic like?**
```scala 
df.printSchema()
```
**Explanation:** In point 4, with the function ".printSchema ()" we show a table with the complete panorama of the dataframe.

**Result**
<html><div align="center"><img src="https://ibb.co/MD8bn35"></div></html>



**5. Print the first 5 columns.**
```scala
df.head(5)
```
**Explanation:** In point 5, with the “.head ()” function, we show the first values ​​of the dataframe, which in this one are the first 5.

**Result**
<html><div align="center"><img src="https://ibb.co/98qJSds"></div></html>



**6. Use describe () to learn about the DataFrame.** 
```scala
df.describe().show()
```
**Explanation:** In point 6, with the function “.describe ()” the information with the most relevant statistical data (total, mean, standard deviation, minimum and maximum value) of the dataframe is displayed.

**Result**
<html><div align="center"><img src="https://ibb.co/2cdbBNC"></div></html>



**7. Create a new dataframe with a new column called "HV Ratio" which is the relationship between the price of the column "High" versus the column "Volume" of shares traded for one day. (Hint: It is a column operation).**
```scala
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))
df2.show() 
```
**Explanation:** In point 7, we create another column with the function “.withColumn ()” that consists of the division of 2 fields of the dataframe and is assigned to another variable so as not to affect the original dataframe.

**Result**
<html><div align="center"><img src="https://ibb.co/wQ5tKYZ"></div></html>



**8. Which day had the highest peak in the “Close” column?**
```scala
df.select("Date","Close").groupBy(dayofweek(df("Date")).alias("Day")).max("Close").sort(col("Day").asc).show(1)
```
**Explanation:** In point 8, we create a query with the commands offered by the import SQL functions. With the function ".select ()" we will carry out the query and as is usually done in the SQL language, we group the data according to a field, assign an alias, specify the column that filters the data and finally the order in which it is will display the query.

**Result**
<html><div align="center"><img src="https://ibb.co/b31n6Vv"></div></html>



**9. Write in your own words in a comment of your code. Which is the meaning of the Close column “Close”?**

**Answer:** It means the price with which the stock ended at the closing of the stock market that day.


**10. What is the max and min of the Volume column?**
```scala
df.select(max("Volume")).show()
df.select(min("Volume")).show()
```
**Explanation:** In point 10, we use the functions “.max ()” and “.min ()” to show the value of the maximum and minimum volume of the Netflix dataframe.

**Result**
<html><div align="center"><img src="https://ibb.co/5rSyGyw"></div></html>



**11. With the Syntax Scala/Spark $ answer the following:**
**a.** How many days was the Close lower than $ 600?
```scala
df.filter($"Close" < 600).count()
```
 **Explanation:** In part a), we use the “.filter ()” function to filter thequery according to a condition that in this one was data less than 600 fromthe “Close” column.

**Result**
<html><div align="center"><img src="https://ibb.co/2htwgMp"></div></html>



**b. What percentage of the time was the High greater than $500?**
```scala
(df.filter($"High">500).count()*1.0/df.count())*100
```
**Explanation:** In part b), we also filter the content of the dataframe, butin this case we perform an operation to calculate the percentage of time.

**Result**
<html><div align="center"><img src="https://ibb.co/wJd36Zy"></div></html>



**c. What is the Pearson correlation between High and Volume?**
```scala
df.select(corr("High","Volume")).show()
```
**Explanation:** In part c), by means of a “.select ()”, we use the “.corr”function to calculate the correlation (Pearson's correlation coefficient)between the “High” and “Volume” columns.

**Result**
<html><div align="center"><img src="https://ibb.co/8PgsNxM"></div></html>



**d. What is the max High per year?**
```scala
df.groupBy(year(df("Date")).alias("Year")).max("High").sort(col("Year").desc)show()
```
**Explanation:** In part d), in a similar way to SQL we create a query throughthe function ".groupBy ()" that groups the data by the year of the column"Date" and the highest value of the column "High ”.

**Result**
<html><div align="center"><img src="https://ibb.co/8KHL1nZ"></div></html>



**e. What is the average Close for each Calender Month?**
```scala
df.groupBy(month(df("Date")).alias("Month")).avg("Close").sort(col("Month")asc).show() 
```
**Explanation:** In part d), also similarly to SQL, we create a query throughthe function ".groupBy ()" that groups the data by the month of the column "Date" and the average of the column "Close".

**Result**
<html><div align="center"><img src="https://ibb.co/HGRDvmQ"></div></html>