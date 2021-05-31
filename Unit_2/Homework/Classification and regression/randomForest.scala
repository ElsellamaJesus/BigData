// Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

// Create a SparkSession
val spark = SparkSession.builder().getOrCreate()

// Load data and convert to DataFrame
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

data.printSchema()
data.show()

// Creation of the label index
// Adjust index to the dataframe
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Identify and index the features
// Features values > 4 are continuous.
val featureIndexer = new VectorIndexer()
.setInputCol("features")
.setOutputCol("indexedFeatures")
.setMaxCategories(4)
.fit(data)

// Split the data into training and test sets (70% training and 30% for tests)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Create a RandomForest model.
val rf = new RandomForestClassifier()
.setLabelCol("indexedLabel")
.setFeaturesCol("indexedFeatures")
.setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.setLabels)

// Pipeline with chain indexers and forest.
val pipeline = new Pipeline()
.setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Fit Model.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Show test rows.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

// Print RandomForestModel
val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
