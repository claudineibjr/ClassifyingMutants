#include    <stdio.h>



void digit_reverser(int N)
{
char v[20], *s;
int M = 0, digit;
      // reverse using arithmetic

      while (N != 0) {
         M = (10 * M) + (N % 10);
         N = N / 10;
      }
      printf("%d\n", M);


      // reverse again using string concatenation
      
      
      s = v;
      while (M != 0) {
         digit = M % 10;
         s[0] =  '0' + digit;
         s++;
         M = M / 10;
      }
      s[0] = '\0';
      printf("%s\n", v);

}

void main(int argc, char *argv[]) { 
      int N = atoi(argv[1]);        // original integer
      digit_reverser(N);
}
