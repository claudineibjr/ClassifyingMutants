/* Introduction to Software Testing
*  Authors: Paul Ammann & Jeff Offutt
*  Chapter 5, section 5.2, page 189 */
#include <stdio.h>

int sum(int x[], int size)
{
	int s = 0;
	int i;
	
	for (i = 0; i < size; i++) {
		s = s + x[i];
	} 
	return s;
}

int inArr[100];

/*Driver function for sum*/
int main (int argc, char* argv[])
{
int i;
    
   
	for (i = 1; i < argc; i++) {
		inArr[i - 1] = atoi(argv[i]);
	}
	                  
	printf("The result is: %d\n", sum(inArr, argc - 1));
	return 0;
}