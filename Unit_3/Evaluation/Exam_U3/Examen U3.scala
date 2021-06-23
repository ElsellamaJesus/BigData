// Import SparkSession
import org.apache.spark.sql.SparkSession

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Create a Spark Session Instance
val spark = SparkSession.builder().getOrCreate()

// Import Kmeans clustering Algorithm
import org.apache.spark.ml.clustering.KMeans

// Load the Wholesale Customers Data, File direction 
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("/home/js/Escritorio/Exam_U3/Wholesale_customers_data.csv")

// Select the following columns for the training set:
// Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
// Cal this new subset feature_data
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

dataset.printSchema

// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors


// Create a new VectorAssembler object called assembler for the feature
// columns as the input Set the output column to be called features
// Remember there is no Label column
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")


// Use the assembler object to transform the feature_data
// Call this new data training_data
val training_data = assembler.transform(feature_data).select("features")

// Create a Kmeans Model with K=3
val kmeans = new KMeans().setK(2).setSeed(1L)

// Fit that model to the training_data
val model = kmeans.fit(training_data)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")


// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

//**********************
// Here is the info on the data:
// 1)   FRESH: annual spending (m.u.) on fresh products (Continuous);
// 2)   MILK: annual spending (m.u.) on milk products (Continuous);
// 3)   GROCERY: annual spending (m.u.)on grocery products (Continuous);
// 4)   FROZEN: annual spending (m.u.)on frozen products (Continuous)
// 5)   DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
// 6)   DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
// 7)   CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
// 8)   REGION: customers Region- Lisnon, Oporto or Other (Nominal)
//**********************