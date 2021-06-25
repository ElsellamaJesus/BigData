# Unit 4 - Final Project

## Table of Contents
- ### [Introduction](#introduction)
- ### [Theoretical framework](#iheoretical-framework)
	- #### [Logistic Regression](#logistic-regression)
	- #### [Decision Tree](#decision-tree)
	- #### [Support Vector Machine](#support-vector-machine)
	- #### [Multilayer Perceptron](#multilayer-perceptron)
- ### [Implementation](#implementation)
- ### [Results](#results)
- ### [Conclusions](#conclusions)
- ### [References](#references)


## Introduction
enter code hereIn this evaluative practice we conclude the course by comparing the performance of the following Logistic Regresion machine learning algorithms. Which are Support Vector Machine, Decision Tree and Multilayer perceptron.

## Theoretical framework

### Logistic Regression
The logistic regression algorithm is one of the most used today in machine learning. Its main application being binary classification problems. It is a simple algorithm in which you can easily interpret the results obtained and identify why one result or another is obtained. Despite its simplicity, it works really well in many applications and is used as a performance benchmark.

![LR](https://miro.medium.com/max/640/1*blOad1e0c5V8EsTx03chWg.gif)
- **Features**
	- It is a method for classification problems.
	- You get a binary value between 0 and 1.
	- The relationship between the dependent variable is measured, with one or more independent variables.
- **Pros**
	-  It is not necessary to have large computational resources, both in training and in execution.
	- The results are highly interpretable.
- **Cons**
	- Impossibility of solving non-linear problems directly.
	- The dependence it shows on the characteristics.

### Decision Tree
A decision tree is a predictive model that divides the predictor space by grouping observations with similar values ​​for the response or dependent variable.

![DT](https://1.bp.blogspot.com/-Ax59WK4DE8w/YK6o9bt_9jI/AAAAAAAAEQA/9KbBf9cdL6kOFkJnU39aUn4m8ydThPenwCLcBGAsYHQ/s0/Random%2BForest%2B03.gif)

- **Features**
	- Poses the problem from different perspectives of action.
	- Its structure allows to analyze the alternatives, the events, the probabilities and the results.
	- Allows you to view all possible solutions to a problem.
- **Pros**
	-  It does not require you to prepare excessively complex data.
	- They are easily combinable with other decision-making tools.
- **Cons**
	- They are unstable, any small change in the input data can lead to a completely different decision tree.
	- A decision tree can easily become too complex, losing its usefulness.

### Support Vector Machine
Support Vector Machines constitute a learning-based method for solving classification and regression problems. In both cases, this resolution is based on a first training phase (where they are informed with multiple examples already solved, in the form of pairs {problem, solution}) and a second phase of use for problem solving. In it, SVMs become a "black box" that provides an answer (output) to a given problem (input).

![SVM](https://vatsalparsaniya.github.io/ML_Knowledge/_images/gif.gif)

- **Features**
	- Kernels make SVMs more flexible and capable of handling non-linear problems.
	- They perform well on many classification and regression tasks.
	- Although SVM algorithms are formulated for binary classification, multiclass SVM algorithms are built by combining several binary classifiers.
- **Pros**
	-  Effective in large spaces.
	- It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
	- It is versatile, different kernel functions can be specified for the decision function.
- **Cons**
	- The algorithm is prone to overfitting, if the number of features is much greater than the number of samples.

### Multilayer Perceptron
A multilayer perceptron (MLP) is an artificial feedback neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses reverse propagation to train the network. MLP is a deep learning method.	

![ML](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/ANN-Graph.gif)

- **Features**
	- The algorithm generates a predictive model for one or more dependent (target) variables based on the values ​​of the predictor variables.
- **Pros**
	-  It is a powerful algorithm.
- **Cons**
	- It is necessary to adjust hyperparameters such as the number of hidden layers, activation functions, number of iterations, solver (optimizer) and number of neurons per layer.
	- They are very sensitive to the scale of the data, so it is convenient to transform them in such a way that all the characteristics have a similar scale, for example, by scaling them to the range [0, 1] or to [-1, 1], or by standardizing them in a that they all have the same mean value and the same standard deviation.

## Implementation
### Code Logistic Regression
```scala
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
val  spark = SparkSession.builder().getOrCreate()
  
//The data from the bank-full.csv dataset is imported
val  df = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

//The makes the process of categorizing string variables to numeric
val  yes = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val  no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val  cd = no.withColumn("y",'y.cast("Int")) 

// An array of the selected fields is created in assembler
val  assembler = new  VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
  
//Transforms into a new dataframe
val  features = assembler.transform(cd)
  
//The column and label are given a new name
val  featuresLabel = features.withColumnRenamed("y", "label")

//Select the indexes
val  dataIndexed = featuresLabel.select("label","features")

//Splitting the data in train 70% and test 30%.
val  Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))

// We can also use the multinomial family for binary classification
val  logisticReg = new  LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")

// Fit the model
val  model = logisticReg.fit(training)

// The test prediction is created
val  predictions = model.transform(test)

// An adjustment is made in prediction and label for the final calculation
val  predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd

// Results
val  metrics = new  MulticlassMetrics(predictionAndLabels)
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy: " + metrics.accuracy)
println(s"Test Error: ${(1.0 - metrics.accuracy)}")
```
### Code Decision Tree
```scala
//Import Libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

//Error level code.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Spark session
val spark = SparkSession.builder().getOrCreate()

//The data from the bank-full.csv dataset is imported
val df  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

//The makes the process of categorizing string variables to numeric
val yes = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val cl = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))

// An array of the selected fields is created in assembler
val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")

//Transforms into a new dataframe
val df2 = assembler.transform(df)

//The column and label are given a new name
val featuresLabel = df2.withColumnRenamed("y", "label")

//Select the indexes
val dataIndexed = featuresLabel.select("label","features")


//Creation of labelIndexer and featureIndexer for the pipeline, Where features with distinct values > 4, are treated as continuous.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)

//Training data as 70% and test data as 30%.
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))

//Creating the Decision Tree object.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

//Creating the Index to String object.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//Creating the pipeline with the objects created before.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

//Fitting the model with training data.
val model = pipeline.fit(training)

//Making the predictions transforming the testData.
val predictions = model.transform(test)

//Showing the predictions
predictions.select("predictedLabel", "label", "features").show(5)

////Creating the evaluator of prediction
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

//Accuracy and Test Error
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

//Show Decision Tree Model
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```
### Code Support Vector Machine
```scala
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
```
### Code Multilayer Perceptron
```scala
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
```
## Results
![Comparativa](https://i.ibb.co/60hQ2Cw/Proyecto-Final-Hoja-1-page-0001.jpg)
### Observations
Regarding the results of our table, after a sample of 45,212 records with 30 iterations of the logistic regression algorithms, decision trees, SVM and multilayer perceptron respectively, we have a table that shows us a clear comparison between the four in terms of to your position. In general it could be concluded that the logistic regression algorithm was the best on this occasion.

| Position | Acuraccy | Time |
|--|--|--|
| 1 | Logistic Regression | Decision Tree |
| 2 | Decision Tree | SVM |
| 3 | Multilayer Perceptron | Logistic Regression |
| 4 | SVM | Multilayer Perceptron |

## Conclusions
In general the algorithms are very fast, there are even some specially optimized ones, the reality is that the algorithms vary in a few seconds in processing, certainly in human terms it is not much, but in machine time it could be relevant. Each algorithm runs under a different mathematical model and yet all algorithms approximate around 88% accuracy at least for this particular dataset.

## References
-   IBM. (2014, 19 noviembre). Regresión Logística. ibm.com. [https://www.ibm.com/docs/es/spss-statistics/SaaS?topic=regression-logistic](https://www.ibm.com/docs/es/spss-statistics/SaaS?topic=regression-logistic)
    
-   Merayo, P. (2021, 22 febrero). Qué son los árboles de decisión y para qué sirven. Máxima Formación. [https://www.maximaformacion.es/blog-dat/que-son-los-arboles-de-decision-y-para-que-sirven/](https://www.maximaformacion.es/blog-dat/que-son-los-arboles-de-decision-y-para-que-sirven/)
    
-   MathWorks. (s. f.). Support Vector Machine (SVM). MATLAB & Simulink. Recuperado 17 de junio de 2021, de [https://la.mathworks.com/discovery/support-vector-machine.html](https://la.mathworks.com/discovery/support-vector-machine.html)
    
-   DeepAI. (2020, 25 junio). Multilayer Perceptron. [https://deepai.org/machine-learning-glossary-and-terms/multilayer-perceptron](https://deepai.org/machine-learning-glossary-and-terms/multilayer-perceptron)
    
-   Machine Learning y Support Vector Machines: porque el tiempo es dinero | Blog. (2020, 1 septiembre). Merkle. [https://www.merkleinc.com/es/es/blog/machine-learning-support-vector-machines](https://www.merkleinc.com/es/es/blog/machine-learning-support-vector-machines)
    
-   Multilayer Perceptron - an overview | ScienceDirect Topics. (2020). ScienceDirect. [https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron](https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron)
    
-   Techopedia. (2017, 30 marzo). Multilayer Perceptron (MLP). Techopedia.Com. [https://www.techopedia.com/definition/20879/multilayer-perceptron-mlp](https://www.techopedia.com/definition/20879/multilayer-perceptron-mlp)
    
-   Rodríguez, D. (2018, 1 julio). La regresión logística. Analytics Lane. [https://www.analyticslane.com/2018/07/23/la-regresion-logistica/](https://www.analyticslane.com/2018/07/23/la-regresion-logistica/)
