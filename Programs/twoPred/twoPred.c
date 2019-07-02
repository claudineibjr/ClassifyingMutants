/* Translated from Introduction to Software Testing
* Authors: Paul Ammann & Jeff Offutt
* Chapter 3, section 3.3, page 130 */
#include <stdio.h>

char twoPred(int x, int y) 
{
	int z = 0;
	if (x < y)
		z = 1;
	else
		z = 0;
	
	if (z && x + y == 10)
		return 'A';
	else
		return 'B';
}

/*Driver function for twoPred*/
int main (int argc, char* argv[])
{
	printf("The result is: %c\n", twoPred(atoi(argv[1]), atoi(argv[2])));
	return 0;
}