// Practica 1

// 1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
val pi = 3.1416
var c = 10;
println($"El radio es de ${c/pi}")

// 2. Desarrollar un algoritmo en scala que me diga si un numero es primo
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

// 3. Dada la variable bird = "tweet", utiliza interpolacion de string para imprimir "Estoy escribiendo un tweet"
val bird = "tweet";
val tweet = $"Estoy escribiendo un ${tweet}";
println(tweet);

// 4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke"
val message = "Hola Luke yo soy tu padre!";
message slice (5,9);

// 5. Cual es la diferencia entre value y una variable en scala?
/*
Un var es una variable. Es una referencia mutable a un valor.Dado que es mutable, su valor puede cambiar
a lo largo de la vida útil del programa. Por otro lado, la palabra clave val representa un valor. 
Es una referencia inmutable, lo que significa que su valor nunca cambia. Una vez asignado, siempre mantendrá el mismo valor.
*/

// 6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416 
val tup = (2,4,5,1,2,3,3.1416,23);
tup._7
