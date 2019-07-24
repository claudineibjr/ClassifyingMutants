/* Introduction to Software Testing
 * Authors: Paul Ammann & Jeff Offutt
 * Chapter 1, section 1.2, page 16 */
#include <stdio.h>

int oddOrPos (int x[], int length)
{  
   /* Effects: return the number of elements in x that
	are either odd or positive (or both) */
   	int count = 0;
	int i;

   	for (i = 0; i < length; i++) {
		if (x[i] % 2 == 1 || x[i] % 2 == -1 || x[i] > 0) {
			count++;
		}
	}
   	return count;
}

/* test:  x=[-3, -2, 0, 1, 4]           
   Expected = 3 */


int inArr[100];
/*Driver function for oddOrPos*/
int main(int argc, char* argv[])
{
   	int i;


	for (i = 1; i < argc; i++) {
		inArr[i - 1] = atoi(argv[i]);
	}
	
	printf("Number of elements that are either odd or positive is: %d\n", oddOrPos (inArr, argc - 1));
	return 0;
} 
