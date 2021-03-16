# Practices - Unit 1
---
## Practice 1

1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
```scala´
val pi = 3.1416
var c = 10;
println($"El radio es de ${c/pi}")
```

2. Desarrollar un algoritmo en scala que me diga si un numero es primo
```scala
var i:Int = 1
var cont:Int = 0
def numPrimo(n: Int): String= {
    for(i <- Array.range(1,n)){
        if(n % i == 0){
            cont = cont + 1
        }
    }
    if(cont > 2){
        return s"El numero ${n} es compuesto"
    }else{
        return s"El numero ${n} es primo"
    }
} 
```

3. Dada la variable bird = "tweet", utiliza interpolacion de string para imprimir "Estoy escribiendo un tweet"
```scala
val bird = "tweet";
val tweet = $"Estoy escribiendo un ${tweet}";
println(tweet);
```
4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke"
```scala
val message = "Hola Luke yo soy tu padre!";
message slice (5,9);
```

5. ¿Cual es la diferencia entre value (val) y una variable (var) en scala?

**var** es una variable. Es una referencia mutable a un valor.Dado que es mutable, su valor puede cambiar a lo largo de la vida útil del programa. Por otro lado, la palabra clave **val** representa un valor. Es una referencia inmutable, lo que significa que su valor nunca cambia. Una vez asignado, siempre mantendrá el mismo valor.

6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416 
```scala
val tup = (2,4,5,1,2,3,3.1416,23);
tup._7
```

## Practice 2

1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"
```scala
var lista = List("rojo","blanco","negro")
lista
```

2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
```scala
lista = lista :+ "verde" :+ "amarillo" :+ "azul" :+ "naranja" :+ "perla"  
```
  
3. Traer los elementos de "lista" "verde", "amarillo", "azul"  
```scala
lista slice (3,6)  
 ```
    
4. Crea un arreglo de número en rango del 1-1000 en pasos de 5 en 5  
```scala
val array = Array.range(1,1000,5)  
```

5. Cuales son los elementos únicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversión a conjuntos  
```scala
var  Lista = List(1,3,3,4,6,7,3,7)  
num.toSet  
```

6. Crea una mapa mutable llamado nombres que contenga los siguiente  
"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"  
```scala
val names = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))  
 ```

7 a . Imprime todas la llaves del mapa  
```scala
names.keys  
```
  
7b . Agrega el siguiente valor al mapa("Miguel", 23)  
```scala
name += ("Miguel" -> 23)
```
