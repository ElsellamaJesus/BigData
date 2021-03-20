# Practices - Unit 1
---
## Practice 1

**1.** Develop a scala algorithm that calculates the radius of a circle
```scala´
val pi = 3.1416
var c = 10;
println($"El radio es de ${c/pi}")
```

2. Develop a scala algorithm that tells me if a number is prime
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

3. Given the variable bird = "tweet", use string interpolation to print "IEstoy escribiendo un tweet"
```scala
val bird = "tweet";
val tweet = $"Estoy escribiendo un ${tweet}";
println(tweet);
```
4. Given the variable message= "Hola Luke yo soy tu padre!" use slilce to extract the sequence "Luke"
```scala
val message = "Hola Luke yo soy tu padre!";
message slice (5,9);
```

5. What is the difference between value (val) and a variable (var) in scala?

**var** es una variable. Es una referencia mutable a un valor.Dado que es mutable, su valor puede cambiar a lo largo de la vida útil del programa. Por otro lado, la palabra clave **val** representa un valor. Es una referencia inmutable, lo que significa que su valor nunca cambia. Una vez asignado, siempre mantendrá el mismo valor.

6. Given the tuple (2,4,5,1,2,3,3.1416,23) returns the number 3.1416
```scala
val tup = (2,4,5,1,2,3,3.1416,23);
tup._7
```
***
## Practice 2

1. Create a list called  "lista" with the elements "rojo", "blanco", "negro"
```scala
var lista = List("rojo","blanco","negro")
lista
```

2. Add 5 more elements to "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
```scala
lista = lista :+ "verde" :+ "amarillo" :+ "azul" :+ "naranja" :+ "perla"  
```
  
3. Bring the elements of "lista" "verde", "amarillo", "azul"  
```scala
lista slice (3,6)  
 ```
    
4. Creates a number array in the range 1-1000 in steps of 5 by 5  
```scala
val array = Array.range(1,1000,5)  
```

5. What are the unique elements of the list Lista(1,3,3,4,6,7,3,7) use conversion to sets 
```scala
var  Lista = List(1,3,3,4,6,7,3,7)  
num.toSet  
```

6. Crea una mapa mutable llamado nombres que contenga los siguiente  
"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"  
```scala
val names = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))  
 ```

7. a. Print all keys on the map
```scala
names.keys  
```
  
7. b. Add the following value to the map("Miguel", 23)  
```scala
name += ("Miguel" -> 23)
```
***
## Practice 3

### Fibonacci number

##### - Algorithm 1. Descending recursive version
```scala
def fib1(n: Int):Int = {
    if(n < 2){
        return n
    } else {
        return fib1(n-1) + fib1(n-2)
    }
} 
```

##### - Algorithm 2. Version with explicit formula
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

##### - Algorithm 3. Iterative version
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

##### - Algorithm 4. Iterative version 2 variables
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

##### - Algorithm 5. Iterative version vector
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

##### - Algorithm 6. Divide and Conquer Version
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