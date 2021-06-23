//Librerias 
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler


//Atrapar los errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark
val spark = SparkSession.builder().getOrCreate()

//Leemos el dataset e inferimos su esquema
val dataframe  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("bank.csv")




//Creamos un indice 
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(dataframe)
val indexed = labelIndexer.transform(dataframe).drop("y").withColumnRenamed("indexedLabel", "label")

//Creamos un vector con las columnas que tienen datos categoricos que convertiremos a datos numericos 
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))

//Transformamos los valores del indice quie creamos en forma de vector 
val features = vectorFeatures.transform(indexed)

//Cambiamos el nombre de la columna que nos interesa predecir 
val featuresLabel = features.withColumnRenamed("y", "label")

//Agregamos la columana al indice para formar una nueva tabla o estructura de datos
val dataIndexed = featuresLabel.select("label","features")

//Creacion del indice donde la caracteristicas de tipo string son como maximo 4 categorias
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)

//Creando vector para dividir datos en entrenamiento y prueba
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))


//SVM
////////////////////////////////////////////////////////////////
//Creando el modelo de support vector machine con un maximo de  11 iteraciones
val SVM = new LinearSVC().setMaxIter(11).setRegParam(0.1)

//Entrenando el modelo con los datos correspondientes
val model = SVM.fit(trainingData)

//Elavuando las predicciones del modelo con los datos de prueba
val predictions = model.transform(testData)

//Creando la matriz o trabla con los datos 
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

//Creando la matriz de confusion 
println("Confusion matrix:")
println(metrics.confusionMatrix)

//Opteniendo la presicion y el porcentaje de error
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error = ${(1.0 - metrics.accuracy)}")



//DECISION THREE
////////////////////////////////////////////////////////////////
//Crando el modelo
val DTR = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

//Crendo un indice de datos de tipo string 
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//Usamos pipeline para asignar donde quereos el indice de datos y con que datos 
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, DTR, labelConverter))

//Entrenamos el modelos con el indice de datos creado  asignado con pipeline
val model = pipeline.fit(trainingData)

//Elavuando las predicciones del modelo con los datos de prueba
val predictions = model.transform(testData)

//Imprimimos 10 filas de al prediccion para observarlos
predictions.select("predictedLabel", "label", "features").show(10)

//Creamo el evaluador del modelo, en donde se analiza la prediccion y nos arroja una presicion 
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

//Entonces determinamos la presicion deacuerdo al evaluador 
val accuracy = evaluator.evaluate(predictions)
 
//Opteniendo la presicion y el porcentaje de error
println(s"Accuracy: ${(accuracy)}")
println(s"Test Error: ${(1.0 - accuracy)}")
 


//LOGISTIC REGRESION 
////////////////////////////////////////////////////////////////
//Crando el modelo Logistic regressio
val LGR = new LogisticRegression().setMaxIter(11).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")

//Entrenando el modelo con los datos correspondientes
val model = LGR.fit(trainingData)

//Elavuando las predicciones del modelo con los datos de prueba
val predictions = model.transform(testData)

//Creando la matriz o trabla con los datos 
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

//Creando la matriz de confusion 
println("Confusion matrix:")
println(metrics.confusionMatrix)

//Opteniendo la presicion y el porcentaje de error
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error: ${(1.0 - metrics.accuracy)}")



//MULTILAYER PERSEPTRON
////////////////////////////////////////////////////////////////
//Asignando las capas de entrada, ocultas y de salida 
val layers = Array[Int](5, 3, 3, 2)

//Creando el modelos, con las capas, el bloque, la semilla de aleatoreidad y el numero maximo de iteraciones
val MP = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)  

//Entrenando el modelo con los datos correspondientes
val model = MP.fit(trainingData)

//Elavuando las predicciones del modelo con los datos de prueba
val prediction = model.transform(testData)

//Creando la matriz o trabla con los datos
val predictionAndLabels = prediction.select("prediction", "label")

//Creamo el evaluador del modelo, en donde se analiza la prediccion y nos arroja una presicion 
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

//Opteniendo la presicion y el porcentaje de error
println(s"Accuracy: ${evaluator.evaluate(predictionAndLabels)}")
println(s"Test Error: ${1.0 - evaluator.evaluate(predictionAndLabels)}")


