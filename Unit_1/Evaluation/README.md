# Evaluation
--- 
## // Exam Unit 1 - Big Data
Import libraries
```scala
import org.apache.spark.sql.SparkSession
import spark.implicits._
```
**1.** Start a simple Spark session.
 ```scala 
val spark = SparkSession.builder().getOrCreate()
```
**2.**Load Netflix Stock CSV file, have Spark infer data types.
```scala
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
```
**3.** What are the column names?
```scala
df.columns
```
**4.** What is the schematic like?
```scala 
df.printSchema()
```
**5.** Print the first 5 columns.
```scala
df.head(5)
```
**6.** Use describe () to learn about the DataFrame. 
```scala
df.describe().show()
```
**7.** Create a new dataframe with a new column called "HV Ratio" which is the relationship between the price of the column "High" versus the column "Volume" of shares traded for one day. (Hint: It is a column operation).
```scala
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))
df2.show() 
```
**8.** Which day had the highest peak in the “Close” column?
```scala
df.select("Date","Close").groupBy(dayofweek(df("Date")).alias("Day")).max("Close").sort(col("Day").asc).show(1)
```
**9.** Write in your own words in a comment of your code. Which is the meaning of the Close column “Close”?
**Answer:** It means the price with which the stock ended at the closing of the stock market that day.

**10.** What is the max and min of the Volume column?
```scala
df.select(max("Volume")).show()
df.select(min("Volume")).show()
```
**11.** With the Syntax Scala/Spark $ answer the following:
    **a.** How many days was the Close lower than $ 600?
   ```scala
    df.filter($"Close" < 600).count()
    ```
    **b.** What percentage of the time was the High greater than $500?
    ```scala
    (df.filter($"High">500).count()*1.0/df.count())*100
    ```
    **c.** What is the Pearson correlation between High and Volume?
    ```scala
    df.select(corr("High","Volume")).show()
    ```
    **d.**What is the max High per year?
    ```scala
    df.groupBy(year(df("Date")).alias("Year")).max("High").sort(col("Year").desc).show()
    ```
    **e.** What is the average Close for each Calender Month?
    ```scala
    df.groupBy(month(df("Date")).alias("Month")).avg("Close").sort(col("Month").asc).show() 
    ```