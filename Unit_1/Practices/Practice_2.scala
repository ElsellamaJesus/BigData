
// 1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"
var lista = List("rojo","blanco","negro")
lista

// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
lista = lista :+ "verde" :+ "amarillo" :+ "azul" :+ "naranja" :+ "perla"

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
lista slice (3,6)

// 4. Crea un arreglo de número en rango del 1-1000 en pasos de 5 en 5
val array = Array.range(1,1000,5)

// 5. Cuales son los elementos únicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversión a conjuntos
var Lista = List(1,3,3,4,6,7,3,7)
num.toSet

// 6. Crea una mapa mutable llamado nombres que contenga los siguiente
//     "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
val names = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))

// 6 a . Imprime todas la llaves del mapa
names.keys

// 7 b . Agrega el siguiente valor al mapa("Miguel", 23)
name += ("Miguel" -> 23)

