#include <stdio.h>


/*Driver function for trashAndTakeOut*/
int driver(int tc_number, int argc, char* argv[])
{

   switch (tc_number)
   {
        case 0:
            trash(atoi(argv[1]));
            break;
        
        case 1:
            printf("%d\n", takeOut(atoi(argv[3]), atoi(argv[4])) );
            break;
   }
}