// EXAMEN UNIT_2

// Libraries Import
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline

// Spark session is imported
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

// 1. Load into a dataframe Iris.csv and later clean the data(https://github.com/jcromerohdz/BigData/blob/master/Spark_DataFrame/ContainsNull.scala)
val df = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("/home/js/Documentos/GitHub/BigData/Unit_2/Evaluation/Exam_U2/iris.csv")

val data = df.na.drop()

// 2. What are the names of the columns?
data.columns

// 3. How is the scheme?
data.printSchema()

// 4. Print the first 5 columns.
data.show(5)

// 5. Use the describe () method to learn more about the data in the DataFrame.
data.describe().show()

// 6. Make the pertinent transformation for the categorical data which will be our labels to be classified.
   // Convert selected columns to vector
    val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")

    // Features are transformed using the dataframe data
    val output = assembler.transform(data)

    // Index the column species that is transformed into numerical data and adjust with the vector output
    val indexer = new StringIndexer().setInputCol("species").setOutputCol("label")
    
    val indexed = indexer.fit(output).transform(output)
    
// 7. Build the classification model and explain its architecture.
    // Divide the data randomly into train 60% and test 40%
    val splits = indexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
    
    val train = splits(0)
    
    val test = splits(1)

    // Specify layers for the neural network
    val layers = Array[Int](4, 5, 4, 3)

    // Create training data and adjust parameters
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    // The model is trained with the training data
    val model = trainer.fit(train)

     // Predictions
    val result = model.transform(test)
    
    val predictionAndLabels = result.select("prediction", "label")
    
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// 8. Imprima los resultados del modelo
    // Results
    result.show(50)
    
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")