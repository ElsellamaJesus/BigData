<center><h1>Exam Unit 3</h1></center>

## Code 
#### Import Libraries
```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

#### Create a Spark Session Instance
```scala
val  spark = SparkSession.builder().getOrCreate()
```

#### Load the Wholesale Customers Data, File direction
```scala
val  dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale_customers_data.csv")
```

#### Select the following columns for the training set:	
***
1. Fresh 
2. Milk
3. Grocery
4. Frozen
5. Detergents_Paper
6. Delicassen
#### Cal this new subset feature_data
```scala
val  feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

dataset.printSchema
```

#### Use the assembler object to transform the feature_data.  Call this new data training_data
```scala
val  training_data = assembler.transform(feature_data).select("features")
```

#### Create a Kmeans Model with K=3
```scala
	val  kmeans = new  KMeans().setK(3).setSeed(1L)
```

#### Fit that model to the training_data
```scala
val  model = kmeans.fit(training_data)
```

#### Evaluate clustering by computing Within Set Sum of Squared Errors
```scala
val  WSSSE = model.computeCost(training_data)

println(s"a = $WSSSE")
```

#### Shows the result
```scala
println("Cluster Centers: ")

model.clusterCenters.foreach(println)
```
<br>

## Output
```scala
scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession

scala> import org.apache.log4j._
import org.apache.log4j._

scala> Logger.getLogger("org").setLevel(Level.ERROR)

scala> val spark = SparkSession.builder().getOrCreate()
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@77262e71

scala> import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.KMeans

scala> val dataset = spark.read.option("header","true").option("inferSchema","true").csv("/home/js/Escritorio/Exam_U3/Wholesale_customers_data.csv")
dataset: org.apache.spark.sql.DataFrame = [Channel: int, Region: int ... 6 more fields]

scala> val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
feature_data: org.apache.spark.sql.DataFrame = [Fresh: int, Milk: int ... 4 more fields]

scala> dataset.printSchema
root
 |-- Channel: integer (nullable = true)
 |-- Region: integer (nullable = true)
 |-- Fresh: integer (nullable = true)
 |-- Milk: integer (nullable = true)
 |-- Grocery: integer (nullable = true)
 |-- Frozen: integer (nullable = true)
 |-- Detergents_Paper: integer (nullable = true)
 |-- Delicassen: integer (nullable = true)


scala> import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}

scala> import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vectors

scala> val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_528b25806063

scala> val training_data = assembler.transform(feature_data).select("features")
training_data: org.apache.spark.sql.DataFrame = [features: vector]

scala> val kmeans = new KMeans().setK(3).setSeed(1L)
kmeans: org.apache.spark.ml.clustering.KMeans = kmeans_8021bb184bc0

scala> val model = kmeans.fit(training_data)
21/06/11 20:40:00 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/06/11 20:40:00 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
model: org.apache.spark.ml.clustering.KMeansModel = kmeans_8021bb184bc0         

scala> val  = model.computeCost(training_data)
warning: there was one deprecation warning; re-run with -deprecation for details
WSSSE: Double = 8.095172370767671E10

scala> println(s"Within Set Sum of Squared Errors = $WSSSE")
Within Set Sum of Squared Errors = 8.095172370767671E10

scala> println("Cluster Centers: ")
Cluster Centers: 

scala> model.clusterCenters.foreach(println)
[7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]
[9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]
[35273.854838709674,5213.919354838709,5826.096774193548,6027.6612903225805,1006.9193548387096,2237.6290322580644]
```