import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
//Spark session.
val spark = SparkSession.builder.appName("DecisionTree").getOrCreate()
//Reading the csv file.
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("C:/Users/Carlos Bojorquez/Desktop/Noveno semestre/bank.csv")
//Indexing.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(df)
val indexed = labelIndexer.transform(df).drop("y").withColumnRenamed("indexedLabel", "label")
 
//Vector of the numeric category columns.
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
//Transforming the indexed value.
val features = vectorFeatures.transform(indexed)
//Renaming the column y as label.
val featuresLabel = features.withColumnRenamed("y", "label")
//Union of label and features as dataIndexed.
val dataIndexed = featuresLabel.select("label","features")
//Creation of labelIndexer and featureIndexer for the pipeline, Where features with distinct values > 4, are treated as continuous.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)
//Training data as 70% and test data as 30%.
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))
//Creating the Decision Tree object.
val decisionTree = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//Creating the Index to String object.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//Creating the pipeline with the objects created before.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, decisionTree, labelConverter))
//Fitting the model with training data.
val model = pipeline.fit(training)
//Making the predictions transforming the testData.
val predictions = model.transform(test)
 
//Showing the predictions.
predictions.select("predictedLabel", "label", "features").show(5)
 
//Creating the evaluator.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
 
//Accuracy.
val accuracy = evaluator.evaluate(predictions)
 
//Accuracy and Test Error.
println(s"Accuracy: ${(accuracy)}")
println(s"Test Error: ${(1.0 - accuracy)}")
 
