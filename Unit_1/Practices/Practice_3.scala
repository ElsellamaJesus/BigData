import util.control.Breaks._
import scala.collection.JavaConversions._

///ALGORITMO 1 DE LA SERIE FIBONACCI
def fib(n:Int): Int={
    if (n < 2){
        return n
    }
    else{
        return fib(n-1) + fib(n-2)
    }
}
  fib(8)
//********************************************************************************
//ALGORITMO 2 DE LA SERIE FIBONACCI
var u=0.0
var j=0.0

def fib(n:Int): Int={
    if (n < 2){
        return n
    }
    else{
        u = (1+(Math.sqrt(5))/2
        j = (((Math.pow(u,n))-(Math.pow((1-u),n)))/(Math.sqrt(5)))
        return Math.round(j)
    }
}

//********************************************************************************
//ALGORITMO 3 DE LA SERIE FIBONACCI
var a=0
var b=1
var c

def fib(n:Int): Int={
    a = 0
    b = 1
    for(k <- Range(0,n)){
        c = b+a
        a = b
        b = c         
    }
    return a
}

//********************************************************************************
//ALGORITMO 4 DE LA SERIE FIBONACCI
def fib(n:Int): Int={
    a = 0
    b = 1
    for(k <- Range(0,n)){
        b = b+a
        a = b-a
          
    }
    return b
}

//********************************************************************************
//ALGORITMO 5 DE LA SERIE FIBONACCI
var vector = Lista(0,1,2,3,4,5,6,7,8,9,10)
def fib(n:Int): Int={
    if (n < 2){
        return n
    }
    else{
        ffor(k <- Range(2,((Lista.length)+1)){
        
          Lista=Lsita-1+Lista-2
        }
    return vector(k)
    }
}

//********************************************************************************
// Algoritmo #6. Versión Divide y Vencerás