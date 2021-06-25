// Import Libraries
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression}
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, VectorAssembler}

// Error level code.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Spark Session
val spark = SparkSession.builder.getOrCreate()

// The data from the bank-full.csv dataset is imported
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/home/js/Documentos/GitHub/BigData/Unit_4/Proyecto/bank-full.csv")

//The makes the process of categorizing string variables to numeric
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(df)
val indexed = labelIndexer.transform(df).drop("y").withColumnRenamed("indexedLabel", "label")

// An array of the selected fields is created in assembler
val vectorFeatures = (new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features"))

//Transforms into a new dataframe
val features = vectorFeatures.transform(indexed)

// The column and label are given a new name
val featuresLabel = features.withColumnRenamed("y", "label")

// Select the indexes
val dataIndexed = featuresLabel.select("label","features")
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)

// Splitting the data in train 70% and test 30%.
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))

// Linear Support Vector Machine object.
val supportVM = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Fit the model
val model = supportVM.fit(training)

// The test prediction is created
val predictions = model.transform(test)

// An adjustment is made in prediction and label for the final calculation
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd

// Results
val metrics = new MulticlassMetrics(predictionAndLabels)
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error = ${(1.0 - metrics.accuracy)}")

