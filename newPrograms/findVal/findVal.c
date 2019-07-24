/* Introduction to Software Testing
 * Authors: Paul Ammann & Jeff Offutt
 * Chapter 5, section 5.2, page 189 */

#include <stdio.h>

int findVal(int numbers[], int length, int val) 
{
	int findVal = -1;
	int i;
	
	for (i = 0; i < length; i++)
		if (numbers[i] == val)
			findVal = i;

	return findVal;                   
}

int inArr[100];

/*Driver function for findVal*/
int main(int argc, char* argv[])
{
   	int i, integer;

	integer = atoi(argv[1]);
	for (i = 2; i < argc; i++) {
		inArr[i - 2] = atoi(argv[i]);
	}
	printf("The LAST occurrence of %d is: %d\n", integer, findVal(inArr, argc - 2, integer));
	return 0;
}
