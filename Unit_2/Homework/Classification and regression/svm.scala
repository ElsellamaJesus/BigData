
//Importamos la librería "LinearSVC", este clasificador binario optimiza la pérdida de bisagra utilizando el optimizador OWLQN.  
import org.apache.spark.ml.classification.LinearSVC

// Importamos y creamos la sesión en spark.
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

// Cargamos los datos de entrenamiento.
val training = spark.read.format("libsvm").load("/Archivos/sample_libsvm_data.txt")

// Establecemos el número máximo de iteraciones y el parámetro de regularización.
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Realizamos un fit para ajustar el modelo.
val lsvcModel = lsvc.fit(training)

// Imprime los coeficientes e intercepta para el Linear SVC.
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")