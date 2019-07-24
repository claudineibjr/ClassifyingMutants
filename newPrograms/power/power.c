/* Introduction to Software Testing
*  Authors: Paul Ammann & Jeff Offutt
*  Chapter 5, section 5.2, page 191 */
#include <stdio.h>

int power (int left, int right)
{
  /**************************************
   Raises Left to the power of Right
   precondition : Right >= 0
   postcondition: Returns Left**Right
  **************************************/
   int rslt, i;
   rslt = left;
   if (right == 0) {
      rslt = 1;
   }
   else {
	for (i = 2; i <= right; i++)
		rslt = rslt * left;
   }
   return rslt;
}

/*Driver function for power*/
int main (int argc, char* argv[])
{
int k,l;
   k = atoi(argv[1]);
   l = atoi(argv[2]);
   printf("The result is: %d\n", power(k,l));
   return 0;
}