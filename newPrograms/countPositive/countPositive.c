/* Introduction to Software Testing
 * Authors: Paul Ammann & Jeff Offutt
 * Chapter 1, section 1.2, page 16 */

#include <stdio.h>

int countPositive (int x[], int length)
{  /*Effects: return the number of
	positive elements in x.*/
   	int count = 0;
	int i;

   	for (i = 0; i < length; i++) {
      	if (x[i] >= 0) {
         	count++;
      	}
   	}
   	return count;
}
/* test:  x=[-4, 2, -1, 2]             
   Expected = 2 */


int inArr[100];

/*Driver function for countPositive*/
int main(int argc, char* argv[])
{
   	int i;


	for (i = 1; i < argc; i++) {
		inArr[i - 1] = atoi(argv[i]);
	}	
	printf("Number of positive numbers is: %d\n", countPositive(inArr, argc - 1));
	return 0;
}
