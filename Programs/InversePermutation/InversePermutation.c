
/*************************************************************************
 *  Compilation:  javac InversePermutation.java
 *  Execution:    java InversePermutation 5 0 2 3 1 4
 *  
 *  Read in a permutation from the command line and print out the inverse
 *  permutation.
 *
 *    % java InversePermutation 5 0 2 3 1 4
 *    2 3 4 5 1 0 
 *
 *************************************************************************/

#include <stdio.h>

int v[100], exists[100], ainv[100];

void invert(int a[], int N)
{      // check if valid
int i;

      for (i = 0; i < N; i++) {
         if (a[i] < 0 || a[i] >= N || exists[a[i]])
         {
             printf("Input is not a permutation.\n");
             return;
         }
         exists[a[i]] = 1;
      }

      // invert
      for (i = 0; i < N; i++)
         ainv[a[i]] = i;


      // print out
      for (i = 0; i < N; i++)
         printf("%d ", ainv[i]);
      printf("\n");
}


void main(int argc, char *argv[]) { 

      int N = argc - 1, i;

      // read in permutation
      for (i = 0; i < N; i++)
         v[i] = atoi(argv[i+1]);
      invert(v, N);
}
