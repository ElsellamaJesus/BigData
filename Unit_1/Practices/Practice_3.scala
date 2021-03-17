// Practica 3 - Serie Fibonacci

// Algoritmo #1. Versión Recursiva Descendente
def fib1(n: Int):Int = {
    if(n < 2){
        return n
    } else {
        return fib1(n-1) + fib1(n-2)
    }
} 


// Algoritmo #2. Versión con Fórmula Explícita 
def fib2(n: Int):Double = {
    if(n < 2){
        return n
    }else{
        var i = ((1 + math.sqrt(5))/2)
        var j = ( (math.pow(i,n) - (1-i)) / math.sqrt(5) )
        return j
    }
}


// Algoritmo #3. Version Iterativa
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


// Algoritmo #4. Versión Iterativa 2 Variables
// *** El algortimo pide retornar 'b' pero si colocamos 'b' el algoritmo omite un lugar 
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


// Algoritmo #5. Versión Iterativa Vector
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


// Algoritmo #6. Versión Divide y Vencerás 
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

