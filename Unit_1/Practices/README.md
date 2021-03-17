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

7. a. Imprime todas la llaves del mapa  
```scala
names.keys  
```
  
7. b. Agrega el siguiente valor al mapa("Miguel", 23)  
```scala
name += ("Miguel" -> 23)
```

## Practice 3

### Fibonacci number

##### Algorithm 1. Descending recursive version
```scala
def fib1(n: Int):Int = {
    if(n < 2){
        return n
    } else {
        return fib1(n-1) + fib1(n-2)
    }
} 
```

##### Algorithm 2. Version with explicit formula
```scala
def fib2(n: Int):Double = {
    if(n < 2){
        return n
    }else{
        var i = ((1 + math.sqrt(5))/2)
        var j = ( (math.pow(i,n) - (1-i)) / math.sqrt(5) )
        return j
    }
}
```

##### Algorithm 3. Iterative version
```scala
def fib3(n: Int):Int = {
    var a = 0
    var b = 1
    var c = 0
    var i = 0
    for(i <- 1 to n){
        c = b + a
        a = b
        b = c
    }
    return a
}

```

##### Algorithm 4. Iterative version 2 variables
```scala
def fib4(n: Int):Int = {
    var a = 0
    var b = 1
    var i = 0
    for(i <- 1 to n){
        b = b + a
        a = b - a
    } 
    return a
}
```

##### Algorithm 5. Iterative version vector
```scala
def fib5(n: Int):Int = {
    if(n < 2){
        return n
    } else {
        var vector = Array.range(0,n + 1)
        vector(0) = 0
        vector(1) = 1
        var i = 0
        // n en vez n + 1
        for(i <- 2 to n){
            vector(i) = vector(i-1) + vector(i-2)
        }
        return vector(n)
    }
}
```

##### Algorithm 6. Divide and Conquer Version
```scala
def fib6(n: Double): Double = {
    if(n <= 0){
        return n; 
    }else{
        var i: Double = n - 1;
        while(i > 0){
        var a: Double = auxTwo;
        var b: Double = auxOne;
        var c: Double = auxOne;
        var d: Double = auxTwo;
        var auxOne: Double = 0;
        var auxTwo: Double = 1
                if(i % 2 == 0){
                    auxOne = ((d*b)+(c*a));
                    auxTwo = ((d*(b+a))+ (c*b));
                    a = auxOne;
                    b = auxTwo;
                    auxOne =  math.pow(c,2) + math.pow(d,2);
                    auxTwo = (d*((2*c)+d));
                    c = auxOne;
                    d = auxTwo;
                    i = i / 2;
                }
                return (a + b);
        }
    }
}
```