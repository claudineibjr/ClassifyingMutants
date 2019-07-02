/* Introduction to Software Testing
// Authors: Paul Ammann & Jeff Offutt
// Chapter 1, section 1.2, page 16 */

#include <stdio.h>

int findLast (int x[], int y, int size)
{  /*Effects:  return the index of the last element
   //   in x that equals y.
   //   If no such element exists, return -1 */
   int i;

   /* As the example in the book points out, this loop should end at 0. */
   for (i = size -1; i >= 0; i--)
   {
      if (x[i] == y)
      {
         return i;
      }
   }
   return -1;
}


int inArr[20];

void main (int argc, char* argv[])
{  /* Driver method for findLastZero
   // Read an array from standard input, call numZero() */

   int i, size, integer;

   if (argc <= 2)
   {
      printf ("Usage: findLast integer v1 [v2] [v3] ... \n");
      return;
   }

   size = argc - 2;
   integer = atoi(argv[1]);
   for (i = 0; i < size; i++)
   {
         inArr[i] = atoi(argv[i+2]);
   }
   printf("The index of the last element equals %d is %d\n", integer,  findLast (inArr, integer, size));
}
