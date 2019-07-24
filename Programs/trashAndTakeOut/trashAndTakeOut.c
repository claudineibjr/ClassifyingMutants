/* Introduction to Software Testing
 * Authors: Paul Ammann & Jeff Offutt
 * Chapter 2, section 2.4, page 74 */
#include <stdio.h>

void trash (int x)    
{                         
   int m, n, o;                 

   m = 0;                    
   if (x > 0)               
      m = 4;                
   if (x > 5)              
      n = 3 * m;              
   else                     
      n = 4 * m;             
   o = takeOut(m, n);
   printf("o is: %d\n", o);  
}

int takeOut (int a, int b) 
{
   int d, e; 

   d = 42 * a; 
   if (a > 0) 
      e = 2 * b + d; 
   else 
      e = b + d; 
   return e; 
} 

main(argc, argv)
int	argc;
char	*argv[];
{
    if ( strcmp("-", argv[1]) == 0 )
    {

        driver(atoi(argv[2]), argc, argv);
    }
    else
    {
        driver(0, argc, argv);
    }
    return 0;
}

