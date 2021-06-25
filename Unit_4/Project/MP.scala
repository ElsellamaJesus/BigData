// Import libraries
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler} 
 
//Error level code.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
//Spark session.
val spark = SparkSession.builder.appName("MultilayerPerseptron").getOrCreate()
//Reading the csv file.
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")
 
//Indexing.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(df)
val indexed = labelIndexer.transform(df).drop("y").withColumnRenamed("indexedLabel", "label")
 
//Vector of the numeric category columns.
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
 
//Transforming the indexed value.
val features = vectorFeatures.transform(indexed)
 
//Fitting indexed and finding labels 0 and 1.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(indexed)
//Splitting the data in 70% and 30%.
val splits = features.randomSplit(Array(0.7, 0.3))
val trainingData = splits(0)
val testData = splits(1)
//Creating the layers array.
val layers = Array[Int](5, 4, 1, 2)
 
//Creating the Multilayer Perceptron object of the Multilayer Perceptron Classifier.
val multilayerP = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)  
 
//Fitting trainingData into the model.
val model = multilayerP.fit(trainingData)
 
//Transforming the testData for the predictions.
val prediction = model.transform(testData)
 
//Selecting the prediction and label columns.
val predictionAndLabels = prediction.select("prediction", "label")
 
//Creating a Multiclass Classification Evaluator object.
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

//Accuracy and Test Error.
println(s"Accuracy: ${evaluator.evaluate(predictionAndLabels)}")
println(s"Test Error: ${1.0 - evaluator.evaluate(predictionAndLabels)}")

