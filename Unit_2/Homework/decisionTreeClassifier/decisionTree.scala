// El siguiente ejemplo carga un conjunto de datos y lo divide en conjuntos de prueba y entrenamiento.

// Importar las librerias necesarias 
import  org.apache.spark.ml.Pipeline 
import  org.apache.spark.ml.classification.DecisionTreeClassificationModel 
import  org.apache.spark.ml.classification.DecisionTreeClassifier 
import  org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
import  org.apache .spark.ml.feature. { IndexToString ,  StringIndexer ,  VectorIndexer }
import  org.apache.spark.sql.SparkSession

// Cargue los datos almacenados en formato LIBSVM como un DataFrame. 
// Es implementado para la optimización, admite clasificación y regresión. 
val data = spark.read.format("libsvm").load("/home/eduardo/Escritorio/expo/sample_libsvm_data.txt")

// Etiquetas de índice, agregando metadatos a la columna de etiquetas. 
// Encajar en el conjunto de datos completo para incluir todas las etiquetas en el índice. 
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Identifica automáticamente características categóricas e indexalas. 
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Divida los datos en conjuntos de prueba y entrenamiento (30% reservado para pruebas). 
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Entrene un modelo DecisionTree. 
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// Convertir etiquetas indexadas de nuevo a etiquetas originales. 
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Cadena de indexadores y árbol en un Pipeline. 
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Modelo de train. Esto también ejecuta los indexadores. 
val model = pipeline.fit(trainingData)

// Hacer predicciones. 
val predictions = model.transform(testData)

// Seleccione filas de ejemplo para mostrar. En este caso solo seran 5
predictions.select("predictedLabel", "label", "features").show(10)

// Seleccione (predicción, etiqueta verdadera).
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// calcule el error de prueba. 
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

// Mostrar por etapas la clasificación del modelo de árbol
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")