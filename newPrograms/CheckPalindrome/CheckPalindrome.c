#include <stdio.h>
#include <string.h>

  /** Check if a string is a palindrome */
int isPalindrome(char s[]) {
	/* The index of the first character in the string */
    int low = 0;

    /* The index of the last character in the string*/
    int high = strlen(s)-1;

    while (low < high) {
      if (s[low] != s[high])
        return 0; /* Not a palindrome*/

      low++;
      high--;
    }

    return 1; /* The string is a palindrome*/
  }

  /** Main method */
void main(int argc, char *argv[]) {

char *p = argv[1];

if (isPalindrome(p))
  printf("%s is a palindrome",p);
else
  printf("%s is not a palindrome",p);

}
