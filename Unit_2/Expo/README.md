
# One-vs-Rest classifier (OvR)
## Transformación a binario 
-   **Clasificación binaria** : Tareas de clasificación con dos clases.
-   **Clasificación multiclase** : Tareas de clasificación con más de dos clases.

Un enfoque para usar algoritmos de clasificación binaria para problemas de clasificación múltiple es dividir el conjunto de datos de clasificación de clases múltiples, en múltiples conjuntos de datos de clasificación binaria y ajustar un modelo de clasificación binaria en cada uno. Dos ejemplos diferentes de este enfoque son las estrategias One-vs-Rest (OvR) y One-vs-One (OvO).
<html><div align="center"><img src="https://utkuufuk.com/2018/06/03/one-vs-all-classification/one-vs-all.png"></div></html>

## OneVsRest
One-vs-rest (OvR para abreviar, también conocido como One-vs-All u OvA) es un método heurístico para usar algoritmos de clasificación binaria para la clasificación de clases múltiples.

Implica dividir el conjunto de datos de clases múltiples en varios problemas de clasificación binaria. Luego, se entrena un clasificador binario en cada problema de clasificación binaria y se hacen predicciones utilizando el modelo que es más confiable. 

Ésta es la estrategia más utilizada para la clasificación multiclase y es una opción justa por defecto.

<html><div align="center"><img src="https://miro.medium.com/max/700/1*4Ii3aorSLU50RV6V5xalzg.png"></div></html>

-   La estrategia One-vs-Rest divide una clasificación de clases múltiples en un problema de clasificación binaria por clase.

## OneVsOne
Se encarga de dividir el problema de N clases en N(N-1)/2 subproblemas binarios. Se realizan clasificaciones entre dos clases {Ci, Cj} donde obtendremos un grado de confianza en favor de Ci en el rango de [0,1]. Cada subproblema binario se almacena en una matriz de votos. Para extraer cual es la clase a la que pertenece se usa la estrategia del voto ponderado la que alcanza la mayor confianza total es la que se predice.
-   La estrategia Uno contra Uno divide una clasificación de clases múltiples en un problema de clasificación binaria por cada par de clases.

## ¿Cómo funciona OvR?

En la clasificación uno contra todos, para el conjunto de datos de instancias de clase N, tenemos que generar los modelos de clasificador N-binario. El número de etiquetas de clase presentes en el conjunto de datos y el número de clasificadores binarios generados deben ser iguales.

`OneVsRest`se implementa como un `Estimator`. Para el clasificador base, toma instancias `Classifier`y crea un problema de clasificación binaria para cada una de las **k** clases. El clasificador de la clase **i** está entrenado para predecir si la etiqueta es **i** o no, distinguiendo la clase **i** de todas las demás clases.

-   El método **One-vs-Rest** para la clasificación multiclase: distinguir entre una clase y todas las demás, donde gana la predicción de clase con mayor probabilidad.
-   El método **One-vs-One** : se entrena un clasificador para cada par de clases, lo que nos permite hacer comparaciones continuas. Gana la predicción de clase con mayor cantidad de predicciones.

<html><div align="center"><img src="https://www.researchgate.net/profile/Shervan-Fekri-Ershad/publication/332370066/figure/fig22/AS:746765703720962@1555054226099/Sample-example-of-a-multiclass-support-vector-machine-The-SVM-algorithms-applied-to-a.ppm"></div></html>

### Tablas One-vs-Rest
<html><div align="center"><img src="https://miro.medium.com/max/1400/1*r8RGcYgqPl8EPQgzaKir8w.png"></div></html>
<html><div align="center"><img src="https://miro.medium.com/max/1400/1*KKXrh7mm-VdRWK_Dq7HDSg.png"></div></html>

## Ventajas
- Una ventaja de este enfoque es su interpretabilidad.

## Desventajas
- Cuando tienes una diferencia muy grande entre clases. (ej. A = 10, B =1000).
- Problema de Desequilibrio.
- Entre mas clases es menos la efectividad.

## Problemática
Para un problema de varias clases con 'n' número de clases, se debe crear 'n' número de modelos, lo que puede ralentizar todo el proceso. Sin embargo, es muy útil con conjuntos de datos que tienen una pequeña cantidad de clases, donde queremos usar un modelo como SVM o Regresión logística.

## Aplicaciones
Este algoritmo es muy usado en muchos campos, incluyendo entre ellos el aprendizaje automático, la mayoría de ellos usados a nivel médico, como por ejemplo para averiguar la gravedad de un paciente se ha llegado a usar este tipo de algoritmo o por ejemplo para predicción de que se produzca una tormenta geomagnética usando modelos de este tipo. Su uso también se puede ver reflejado en el campo de la ingeniería, especialmente para saber la probabilidad de fallo de un proceso, un producto o un sistema.

1.  OneVsOne:
-   Retorno de Clientes vs Clientes nuevos
-   Solicitantes exitosos vs Solicitantes no exitosos
-   Cáncer agresivo vs Cáncer pasivo
2.  OneVsRest:
-   Tipos de cancer
-   Tipos de clientes
-   Diferentes dificultades en videojuegos

## Ejemplo
 Este ejemplo One-vs-Rest esta suave
https://www.machinecurve.com/index.php/2020/11/11/creating-one-vs-rest-and-one-vs-one-svm-classifiers-with-scikit-learn/

#### Uno contra uno
Digamos que tiene tres categorías: perros, gatos y conejos. En la estrategia uno contra uno, entrenaría un modelo para cada una de las permutaciones binarias:
-   M1: conejos vs. perros.
-   M2: gatos contra conejos.
-   M3: perros contra gatos.

En el momento de la predicción, la probabilidad de una muestra es la probabilidad promedio sobre los dos modelos para esa muestra.
![Comparacion](https://user.oc-static.com/upload/2019/05/28/1559071092695_ch9_009_one_vs_one.png)
-------------Estrategia de clasificación multiclase uno contra uno---------------

Considere el conejo encerrado en un círculo azul. Visualmente, puede adivinar que esa muestra tendrá una gran probabilidad de ser clasificada como un conejo con M1 y M2. Su probabilidad general de ser clasificado como conejo es el promedio de las probabilidades dadas por M1 y M2.

Por el contrario, el conejo encerrado en un círculo rojo está más cerca de perros y gatos, y su probabilidad por debajo de M1 y M2 estaría más cerca de 0.5. La probabilidad general de que esa muestra sea un conejo sería menor que la del conejo encerrado en un círculo azul.

#### Uno contra el resto
Este enfoque es más secuencial.
**1.**  Elige arbitrariamente una clase, digamos conejos, y entrena a su modelo para distinguir entre conejos y todas las demás categorías (gatos y perros).
**2.**  Dejas a un lado todas las muestras que se identifican como conejos y construyes un segundo modelo binario en la muestra restante que identifica perros versus gatos.

Terminas construyendo dos modelos:
-   **M1:** conejos vs. no conejos.
-   **M2:** eliminar los conejos identificados, construir un modelo de perro contra gato.

![Comparacion](https://user.oc-static.com/upload/2019/05/28/15590716454362_ch09_010_one_vs_rest.png)
----------Estrategia de clasificación multiclase uno contra todos-----------
## Código
```scala
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
```

## Videos

 1. https://www.youtube.com/watch?v=u9kchxQAelM&ab_channel=AprendeIAconLigdiGonzalez
 2. https://www.youtube.com/watch?v=T8aCfSBlrqU&ab_channel=AprendeIAconLigdiGonzalez
 3. https://www.youtube.com/watch?v=_s3z8dQX3pM&ab_channel=DataScienceDojo

## Reference

 - https://spark.apache.org/docs/2.4.7/ml-classification-regression.html#one-vs-rest-classifier-aka-one-vs-all 
 - https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/OneVsRestExample.scala
 - [https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/one-vs-all-multiclass](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/one-vs-all-multiclass)
- [https://programmerclick.com/article/18631372894/](https://programmerclick.com/article/18631372894/)
- [https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)
- [https://www.geeksforgeeks.org/one-vs-rest-strategy-for-multi-class-classification/](https://www.geeksforgeeks.org/one-vs-rest-strategy-for-multi-class-classification/)
- Enlace a procedimiento de funcionamiento: [https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b](https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b)
- [https://www.machinecurve.com/index.php/2020/11/11/creating-one-vs-rest-and-one-vs-one-svm-classifiers-with-scikit-learn/](https://www.machinecurve.com/index.php/2020/11/11/creating-one-vs-rest-and-one-vs-one-svm-classifiers-with-scikit-learn/)
- [https://openclassrooms.com/en/courses/5873596-design-effective-statistical-models-to-understand-your-data/6233016-build-and-interpret-a-logistic-regression-model](https://openclassrooms.com/en/courses/5873596-design-effective-statistical-models-to-understand-your-data/6233016-build-and-interpret-a-logistic-regression-model)

