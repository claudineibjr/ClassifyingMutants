/* Introduction to Software Testing
   Authors: Paul Ammann & Jeff Offutt
   Chapter 2, section 2.3, page 63*/

/** *****************************************************
* Finds and prints n prime integers
* Jeff Offutt, Spring 2003
********************************************************* */

#include <stdio.h>

int isDivisible (int i, int j)
{
   if (j%i == 0)
      return 1;
   else
      return 0;
}

void printPrimes (int n)
{
   int curPrime =0;           /* Value currently considered for primeness*/
   int numPrimes=0;          /* Number of primes found so far.*/
   int isPrime = 0;        /* Is curPrime prime?*/
   int primes[100];
   int i, j;
 
   /* Initialize 2 into the list of primes.*/
   primes[0] = 2;
   numPrimes = 1;
   curPrime  = 2;
   while (numPrimes < n)
   {
      curPrime++;  /* next number to consider ...*/
      isPrime = 1;

        for ( i = 0; i <= numPrimes-1; i++)
        {
            if (isDivisible(primes[i], curPrime)) { /* Found a divisor, curPrime is not prime.*/
                isPrime = 0;
                break; /* out of loop through primes.*/
            }
        }
      if (isPrime==1)
      {   /* save it!*/
         primes[numPrimes] = curPrime;
         numPrimes++;
      }
   }  /* End while*/

   for (j = 0; j <= numPrimes-1; j++)
   {
           printf("Prime:    %d \n",primes[j]);
   }
   /*return primes;*/
}  /* end printPrimes*/

void main (int argc, char *argv[])
{  /* Driver method for printPrimes
      Read an integer from standard input, call printPrimes()*/
   int integer = 0;

  
   integer = atoi(argv[1]);

   printPrimes(integer);

}

