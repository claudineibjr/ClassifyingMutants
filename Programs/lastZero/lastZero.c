/* Introduction to Software Testing
 * Authors: Paul Ammann & Jeff Offutt
 * Chapter 1, section 1.2, page 16 */
#include <stdio.h>

int lastZero(int x[], int length)
{
	/*Effects: return the index of the LAST 0 in x.
	  Return -1 if 0 does not occur in x */
	int i;
	int index = -1;
		
	for (i = 0; i < length; i++) {
		if (x[i] == 0) {
			index = i;
		}
	}
	return index;
}

/* test:  x=[0, 1, 0]              
   Expected = 2 */

int inArr[100];

/*Driver function for lastZero*/
int main(int argc, char* argv[])
{
   	int i;


	for (i = 1; i < argc; i++) {
		inArr[i - 1] = atoi(argv[i]);
	}
	
	printf("The last index of zero is: %d\n", lastZero(inArr, argc - 1));
	return 0;
}
