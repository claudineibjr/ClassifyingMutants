/* Introduction to Software Testing
// Authors: Paul Ammann & Jeff Offutt
// Chapter 1, section 1.2, page 12

// Translated to C by Marcio Delamaro */

#include    <stdio.h>

int numZero (int arr[], int size)
{
   /* return the number of occurrences of 0 in arr */
   int i, count = 0;

   /* As example in the book points out, this loop should start at 0. */
   for (i = 0; i < size; i++)
   {
      if (arr[i] == 0)
      {
         count++;
      }
   }
   return count;
}


int inArr[20];

void main (int argc, char* argv[])
{  /* Driver method for numZero
   // Read an array from standard input, call numZero() */

   int i, size;

   if (argc <= 1)
   {
      printf ("Usage: java numZero v1 [v2] [v3] ... \n");
      return;
   }

   size = argc - 1;
   for (i = 0; i < size; i++)
   {
         inArr[i] = atoi(argv[i+1]);
   }
   printf("Number of zeros is: %d\n", numZero (inArr,size));
}

