
# Evaluation Unit 2 - Big Data

**1. We import the necessary libraries for the cleaning, analysis and interpretation of the data.**
```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
```
**2. Start a simple Spark session.**
 ```scala 
val  spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()
```
**Explanation:** In point 2, you have to start a session in Spark, for which we already import the sql library to start the session, finally we assign the created session to a variable.


**3. We load the file iris.csv Load in a dataframe Iris.csv, ([https://github.com/jcromerohdz/BigData/blob/master/Spark_DataFrame/ContainsNull.scalatti ])**
```scala
val  df = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("/home/js/Escritorio/examen/iris.csv")
```
**Explanation:** At point 3, we append the dataframe to use, assign it to a variable that we call "df (dataframe)" and infer the data. It should be noted that the csv file was not in the same folder as the scala file, so the path was specified.



**4. We clean the data**
```scala
val  data = df.na.drop()
```
**Explanation:** In point 4, we use the "na.drop" function to eliminate the rows that contain null values, since if they have no value they will only cause the use of more resources and errors in the whole analysis.




**5. What are the column names?**
```scala 
data.columns
```
**Explanation:** In point 5, with a property called ".columns", we only called the names of the columns of our dataframe.

**Result**
```scala 
res4: Array[String] = Array(sepal_length, sepal_width, petal_length, petal_width, species)
```



**6. What is the scheme like?**
```scala
data.printSchema()
```
**Explanation:** In point 6, with the function ".printSchema ()" we show a table with the complete panorama of the dataframe.

**Result**
```scala 
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
```



**7. Print the first 5 columns**
```scala
data.show(5)
```
 **Explanation:** In point 7, with the “.show ()” function, we show the first values ​​of the dataframe, which in this one are the first 5.

**Result**
```scala
+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 5 rows
```



**8. Describe the nature of the data**
```scala
data.describe.show()
```
**Explanation:** In point 8, with the function “.describe ()”, we show the characteristics of the data.

**Result**
```scala
+-------+------------------+-------------------+------------------+------------------+---------+
|summary|      sepal_length|        sepal_width|      petal_length|       petal_width|  species|
+-------+------------------+-------------------+------------------+------------------+---------+
|  count|               150|                150|               150|               150|      150|
|   mean| 5.843333333333335| 3.0540000000000007|3.7586666666666693|1.1986666666666672|     null|
| stddev|0.8280661279778637|0.43359431136217375| 1.764420419952262|0.7631607417008414|     null|
|    min|               4.3|                2.0|               1.0|               0.1|   setosa|
|    max|               7.9|                4.4|               6.9|               2.5|virginica|
+-------+------------------+-------------------+------------------+------------------+---------+
```



**9. We make the pertinent transformation for the categorical data which will be our labels to be classified.**
```scala
val  assembler = new  VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
```
**Explanation:** At point 9, we have to convert the selected columns into a vector, which will be our characteristics.


**10. Define output data**
```scala
val  output = assembler.transform(data)
```
**Explanation:** At point 10, we have to transform the features using the dataframe.


**11. Index data**
```scala
val  indexer = new  StringIndexer().setInputCol("species").setOutputCol("label")

val  indexed = indexer.fit(output).transform(output)
```
**Explanation:** In point 11, we create indexes of the species column that is transformed into numerical data and we adjust with the output vector “output”.


**12. Build the classification model and explain the architecture.**

```scala
val  splits = indexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

val  train = splits(0)

val  test = splits(1)
```
**Explanation:** At point 12, we have to randomly divide the data into training "train 60% (0.6)" and test "test 40% (0.4)", then we assign variables to the selected data.


**13. We create the neural network**
```scala
val  layers = Array[Int](4, 5, 4, 3)

val  trainer = new  MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
```
**Explanation:** In point 13, we define the stages of the neural network, four input layer elements, two cultured layers, one of five and one of four elements respectively, and three output layer elements. Next we assign the training data, the model with which we will work in this chapter.


**14. Train the model**
```scala
val  model = trainer.fit(train)

val  result = model.transform(test)

val  predictionAndLabels = result.select("prediction", "label")
```
**Explanation:** In point 14, Train the model with the data selected for training. Evaluate the training result using the same model but with the test data, and print the prediction results vs. those we already had in the dataframe.



**15. Evaluate the model**
```scala
val  evaluator = new  MulticlassClassificationEvaluator().setMetricName("accuracy")

result.show(50)

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
** Explanation: ** In point 15, Determine a metric to evaluate the model, that is, the precision. Print the first 50 rows of the resulting data frame. And finally, print the data classification precision percentage.

**Result**
```scala
+------------+-----------+------------+-----------+-------+-----------------+-----+--------------------+--------------------+----------+
|sepal_length|sepal_width|petal_length|petal_width|species|         features|label|       rawPrediction|         probability|prediction|
+------------+-----------+------------+-----------+-------+-----------------+-----+--------------------+--------------------+----------+
|         4.3|        3.0|         1.1|        0.1| setosa|[4.3,3.0,1.1,0.1]|  2.0|[15.6939102472457...|[4.49726715588115...|       2.0|
|         4.4|        2.9|         1.4|        0.2| setosa|[4.4,2.9,1.4,0.2]|  2.0|[15.7205776664764...|[5.32055408135416...|       2.0|
|         4.4|        3.0|         1.3|        0.2| setosa|[4.4,3.0,1.3,0.2]|  2.0|[15.7038134381871...|[4.78697261626649...|       2.0|
|         4.6|        3.4|         1.4|        0.3| setosa|[4.6,3.4,1.4,0.3]|  2.0|[15.6877056144553...|[4.32476070392186...|       2.0|
|         4.6|        3.6|         1.0|        0.2| setosa|[4.6,3.6,1.0,0.2]|  2.0|[15.6739436876185...|[3.96538767140727...|       2.0|
|         4.7|        3.2|         1.6|        0.2| setosa|[4.7,3.2,1.6,0.2]|  2.0|[15.7412734317472...|[6.06200464499190...|       2.0|
|         4.8|        3.1|         1.6|        0.2| setosa|[4.8,3.1,1.6,0.2]|  2.0|[15.7753958949252...|[7.51682313582786...|       2.0|
|         5.0|        3.2|         1.2|        0.2| setosa|[5.0,3.2,1.2,0.2]|  2.0|[15.7415843992936...|[6.07389961264308...|       2.0|
|         5.0|        3.3|         1.4|        0.2| setosa|[5.0,3.3,1.4,0.2]|  2.0|[15.7498146125576...|[6.39734401981583...|       2.0|
|         5.0|        3.4|         1.5|        0.2| setosa|[5.0,3.4,1.5,0.2]|  2.0|[15.7461427849689...|[6.25096742565720...|       2.0|
+------------+-----------+------------+-----------+-------+-----------------+-----+--------------------+--------------------+----------+
only showing top 10 rows
 
 scala>   
     |    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
Test set accuracy = 0.9607843137254902
```
#
**Conclusion and observations**

We have learned the basics of classification, the main algorithms and how to adjust them according to the problem we face. There is always the risk of overtraining or biases in which the data have a large difference between one and the other.
