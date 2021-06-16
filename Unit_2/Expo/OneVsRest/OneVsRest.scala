// Importar Librerias
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Cargar el archivo
val inputData = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")

// Generar la division de conjunto train y test.
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// Instanciar el clasificador base
val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

// Se crea una instancia del clasificador One Vs Rest.
val ovr = new OneVsRest().setClassifier(classifier)

// Se entrena (train) el modelo multiclase.
val ovrModel = ovr.fit(train)

// Se puntua el modelo en los datos de prueba (test).
val predictions = ovrModel.transform(test)

// Se obtiene el evaluador
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// Se calcula el error de clasificación en los datos de prueba.
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")