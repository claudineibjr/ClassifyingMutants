/* Introduction to Software Testing
 * Authors: Paul Ammann & Jeff Offutt
 * Chapter 2, section 2.3, page 56 */

#include <stdio.h>
#include <string.h>

#define MAX 100
#define NOTFOUND -1

char subject[MAX];
char pattern[MAX];


int pat (char subject[], int subjectLen, char pattern[], int patternLen) 
{ 
/* Post: if pattern is not a substring of subject, return -1 
 *       else return (zero-based) index where the pattern (first) 
 *       starts in subject  */
  
int iSub = 0;
int rtnIndex = NOTFOUND; 
int isPat  = 0; //false 
int iPat = 0;
   
	while (isPat == 0 && (iSub + patternLen - 1 < subjectLen)) 
	{ 
		if (subject[iSub] == pattern[0]) 
		{	 
			rtnIndex = iSub; // Starting at zero 
			isPat = 1; //found it! 
			for (iPat = 1; iPat < patternLen; iPat++) 
			{ 
				if (subject[iSub + iPat] != pattern[iPat]) 
				{ 
					rtnIndex = NOTFOUND; 
					isPat = 0; // false 
					break;  // out of for loop 
				} 
			} 
		} 
		iSub++; 
	} 
   return rtnIndex; 
}

/*Driver function for testPad*/
int main(int argc, char* argv[])
{	
	int n = 0;

	/*Read params from standard input, call testPad*/
	if (argc == 2)
	{
		pattern[0] = '\0';
	}
	else
	{
		strcpy(pattern, argv[2]);
	}
	strcpy(subject, argv[1]);

	if ((n = pat(subject, strlen(subject), pattern, strlen(pattern))) == -1) {
		printf("Pattern string is not a substring of the subject string\n");
	} else {
		printf("Pattern string begins at the character %i\n", n);
	}
	return 0;
}
