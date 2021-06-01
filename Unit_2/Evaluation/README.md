
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

**Result**
<html><div><img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fhelp.turitop.com%2Fhc%2Farticle_attachments%2F360012747880%2Ferror.png&f=1&nofb=1" alt="Cap6"></div></html>



**3. We load the file iris.csv Load in a dataframe Iris.csv, ([https://github.com/jcromerohdz/BigData/blob/master/Spark_DataFrame/ContainsNull.scalatti ])**
```scala
val  df = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("/home/js/Escritorio/examen/iris.csv")
```
**Explanation:** At point 3, we append the dataframe to use, assign it to a variable that we call "df (dataframe)" and infer the data. It should be noted that the csv file was not in the same folder as the scala file, so the path was specified.

**Result**
<html><div><img src="" alt="Cap6"></div></html>


**4. We clean the data**
```scala
val  data = df.na.drop()
```
**Explanation:** In point 4, we use the "na.drop" function to eliminate the rows that contain null values, since if they have no value they will only cause the use of more resources and errors in the whole analysis.

**Result**
<html><div><img <html><div><img src="" alt="Cap6"></div></html>



**5. What are the column names?**
```scala 
data.columns
```
**Explanation:** In point 5, with a property called ".columns", we only called the names of the columns of our dataframe.

**Result**
<html><div><img src="" alt="Cap6"></div></html>



**6. What is the scheme like?**
```scala
data.printSchema()
```
**Explanation:** In point 6, with the function ".printSchema ()" we show a table with the complete panorama of the dataframe.

**Result**
<html><div><img src="" alt="Cap5"></div></html>



**7. Print the first 5 columns**
```scala
data.show(5)
```
 **Explanation:** In point 7, with the “.show ()” function, we show the first values ​​of the dataframe, which in this one are the first 5.

**Result**
<html><div><img src="" alt="Cap6"></div></html>



**8. Describe the nature of the data**
```scala
data.describe.show()
```
**Explanation:** In point 8, with the function “.describe ()”, we show the characteristics of the data.

**Result**
<html><div><img src="" alt="Cap6"></div></html>



**9. We make the pertinent transformation for the categorical data which will be our labels to be classified.**
```scala
val  assembler = new  VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
```
**Explanation:** At point 9, we have to convert the selected columns into a vector, which will be our characteristics.

**Result**
<html><div><img src="" alt="Cap6"></div></html>



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

**Result**
<html><div><img src="" alt="Cap6"></div></html>



**12. Build the classification model and explain the architecture.**

```scala
val  splits = indexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

val  train = splits(0)

val  test = splits(1)
```
**Explanation:** At point 12, we have to randomly divide the data into training "train 60% (0.6)" and test "test 40% (0.4)", then we assign variables to the selected data.

**Result**
<html><div><img src="" alt="Cap6"></div></html>



**13. We create the neural network**
```scala
val  layers = Array[Int](4, 5, 4, 3)

val  trainer = new  MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
```
**Explanation:** In point 13, we define the stages of the neural network, four input layer elements, two cultured layers, one of five and one of four elements respectively, and three output layer elements. Next we assign the training data, the model with which we will work in this chapter.

**Result**
<html><div><img src="" alt="Cap6"></div></html>



**14. Train the model**
```scala
val  model = trainer.fit(train)

val  result = model.transform(test)

val  predictionAndLabels = result.select("prediction", "label")
```
**Explanation:** In point 14, Train the model with the data selected for training. Evaluate the training result using the same model but with the test data, and print the prediction results vs. those we already had in the dataframe.

**Result**
<html><div><img src="" alt="Cap6"></div></html>



**15. Evaluate the model**
```scala
val  evaluator = new  MulticlassClassificationEvaluator().setMetricName("accuracy")

result.show(50)

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
** Explanation: ** In point 15, Determine a metric to evaluate the model, that is, the precision. Print the first 50 rows of the resulting data frame. And finally, print the data classification precision percentage.

**Result**
<html><div><img src="" alt="Cap6"></div></html>

#
**Conclusion and observations**
