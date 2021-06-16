//correlation

//Empezamos con esta libreria para tener acceso a matrices locales y Métodos de fábrica para Vector.
import org.apache.spark.ml.linalg.{Matrix, Vectors}
//libreria para usar el metodo de correlacion
import org.apache.spark.ml.stat.Correlation
//permite acceder a un valor de una fila a través del acceso genérico por ordinal,  así como el acceso primitivo
import org.apache.spark.sql.Row

//Crea vectores densos y dispersos  a partir de sus valores, dentro de la matriz 
val data = Seq(
   (4, Seq((0, 1.0), (3, -2.0))),
  Vectors.dense(4.0, 5.0, 0.0, 3.0),
  Vectors.dense(6.0, 7.0, 0.0, 8.0),
  Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)

//Se extraen los datos de nuestra matriz y se crea un dataframe respecto a las caracteristicas 
val df = data.map(Tuple1.apply).toDF("features")
//Se crea la matriz de correlacion Pearson usando el dataframe que acabamos de crear y le pedimos los primeros valores con head
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
//imprimimos el resultado
println(s"Pearson correlation matrix:\n $coeff1")
//Se crea la matriz de correlacion Spearman usando el dataframe que acabamos de crear y le pedimos los primeros valores con head
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
//imprimimos el resultado
println(s"Spearman correlation matrix:\n $coeff2")



//Hypothesis testing


//se hace uso de la siguiente libreria para aplicar metodos a vectores 
import org.apache.spark.ml.linalg.{Vector, Vectors}
//Tambien se utiliza la libreria de chiSquare para realizar los calculos necesarios
import org.apache.spark.ml.stat.ChiSquareTest

// se crea la siguiente secuencia de vectores densos 
val data = Seq(
  (0.0, Vectors.dense(0.5, 10.0)),
  (0.0, Vectors.dense(1.5, 20.0)),
  (1.0, Vectors.dense(1.5, 30.0)),
  (0.0, Vectors.dense(3.5, 30.0)),
  (0.0, Vectors.dense(3.5, 40.0)),
  (1.0, Vectors.dense(3.5, 40.0))
)

//Creacion del dataframe a partir del conjunto de vectores anterior 
val df = data.toDF("label", "features")
//se toman los primeros valores del dataframe previamente creado
val chi = ChiSquareTest.test(df, "features", "label").head
//De inicio con las partes de la prueba, se buscaran los valores de p 
println(s"pValues = ${chi.getAs[Vector](0)}")
//Despues se buscaran los grados de libertad del modelo
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
//por ultimo se extraeran ciertos valores de un vector dererminado todo en base a la funcion chi cuadrado
println(s"statistics ${chi.getAs[Vector](2)}")


//Summarizer

//importacion de librerias necesarias, en este uso de vectores y el propio summarizer
import spark.implicits._    
import Summarizer._

//se crea un conjunto de vectores o secuencia
val data = Seq(
  (Vectors.dense(2.0, 3.0, 5.0), 1.0),
  (Vectors.dense(4.0, 6.0, 7.0), 2.0)
)

//Creacion del dataframe a partir de los vectores
val df = data.toDF("features", "weight")


//se hace uso de la libreria summarizer para obtener la media y la varianza de algunos datos en el dataframe solicitado
val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()

//se imprimen las variables trabajadas anteriormente
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

//se repite el procesos con 2 nuevas variables 
val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
  .as[(Vector, Vector)].first()
// impresión de variables
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")