import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
val spark = SparkSession.builder.getOrCreate()
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank.csv")
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(df)
val indexed = labelIndexer.transform(df).drop("y").withColumnRenamed("indexedLabel", "label")
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
val features = vectorFeatures.transform(indexed)
val featuresLabel = features.withColumnRenamed("y", "label")
val dataIndexed = featuresLabel.select("label","features")
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))
val logisticReg = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val model = logisticReg.fit(trainingData)
val predictions = model.transform(testData)
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error: ${(1.0 - metrics.accuracy)}")
