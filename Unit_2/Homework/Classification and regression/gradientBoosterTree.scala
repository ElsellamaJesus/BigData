// Importamos las librerias que ocupamos
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Cargamos el archivo txt de la ruta establecida
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")


//Crearmos una columna usando stringIndexer para que los datos tengan su categorización
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
  
// Creamos un vector que tendra un maximo de 4 categorias
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Separamos los datos en dos partes, un llamado training con 70% y el otro test seta con 30%
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

//Se entren al modelo GPT
val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)
  .setFeatureSubsetStrategy("auto")

//Comvertimos las etiqutas indezadas a etiquetas originales
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

//La cadena de los indezadores y GPT EN Pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

//Se entreana al modelo. Esto también ejecuta los indexadores
val model = pipeline.fit(trainingData)

//Creamos las predicciones.
val predictions = model.transform(testData)

//Selecionamos las primeras 5 filas para desplegarlas
predictions.select("predictedLabel", "label", "features").show(5)

//Seleccionamos prediccion y calculo del error de prueba.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")