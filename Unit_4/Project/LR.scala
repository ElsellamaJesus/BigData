// Import Libraries
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer,VectorIndexer}

//Error level code.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Spark session
val spark = SparkSession.builder().getOrCreate()

//The data from the bank-full.csv dataset is imported
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

//The makes the process of categorizing string variables to numeric
val yes = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val cd = no.withColumn("y",'y.cast("Int"))

// An array of the selected fields is created in assembler
val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")

//Transforms into a new dataframe
val features = assembler.transform(cd)

//The column and label are given a new name
val featuresLabel = features.withColumnRenamed("y", "label")

//Select the indexes
val dataIndexed = featuresLabel.select("label","features")

//Splitting the data in train 70% and test 30%.
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))

// We can also use the multinomial family for binary classification
val logisticReg = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")

// Fit the model
val model = logisticReg.fit(training)

// The test prediction is created
val predictions = model.transform(test)

// An adjustment is made in prediction and label for the final calculation
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd

// Results
val metrics = new MulticlassMetrics(predictionAndLabels)
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error: ${(1.0 - metrics.accuracy)}")
