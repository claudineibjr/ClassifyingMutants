/* Introduction to Software Testing
 * Authors: Paul Ammann & Jeff Offutt
 * Chapter 3, section 3.3, page 130 */
#include <stdio.h>

void checkIt (int a, int b, int c)
{  
   if (a && (b || c))
   {
      printf("P is true\n");
   }
   else
   {
      printf("P isn't true\n");
   }
}

/*Driver function for checkIt*/
int main(int argc, char* argv[])
{
	checkIt(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
	return 0;
}