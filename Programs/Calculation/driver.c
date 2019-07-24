#include <stdio.h>


void driver(int argc, char  *argv[])
{
   int tc_number = atoi(argv[1]);
   switch (tc_number)
   {
        case 1:
            printf("%d\n", FnMinus(0,0));
            printf("%d\n", FnMinus(5,0));
            printf("%d\n", FnMinus(-5,0));
            printf("%d\n", FnMinus(0,-5));
            printf("%d\n", FnMinus(0,5));
            printf("%d\n", FnMinus(-5,5));
            printf("%d\n", FnMinus(-5,-5));
        break;
        case 2:
            printf("%d\n", FnTimes(0,0));
            printf("%d\n", FnTimes(5,0));
            printf("%d\n", FnTimes(-5,0));
            printf("%d\n", FnTimes(0,-5));
            printf("%d\n", FnTimes(0,5));
            printf("%d\n", FnTimes(-5,5));
            printf("%d\n", FnTimes(-5,-5));
        break;
        case 3:
            printf("%d\n", FnDivide(0,0));
            printf("%d\n", FnDivide(5,0));
            printf("%d\n", FnDivide(-5,0));
            printf("%d\n", FnDivide(0,-5));
            printf("%d\n", FnDivide(0,5));
            printf("%d\n", FnDivide(-5,5));
            printf("%d\n", FnDivide(-5,-5));
        break;
        case 4:
            printf("%d\n", DefineAndRoundFraction(0,0));
            printf("%d\n", DefineAndRoundFraction(5,0));
            printf("%d\n", DefineAndRoundFraction(-5,0));
            printf("%d\n", DefineAndRoundFraction(0,-5));
            printf("%d\n", DefineAndRoundFraction(0,5));
            printf("%d\n", DefineAndRoundFraction(-5,5));
            printf("%d\n", DefineAndRoundFraction(-5,-5));
        break;
        case 5:
            printf("%d\n", FnDivide(-5,1));
        break;
        case 6:
            printf("%d\n", FnDivide(1,-1));
        break;
        case 7:
            printf("%d\n", FnDivide(5,2));
        break;
        case 8:
            printf("%d\n", FnDivide(-1,1));
        break;
        case 9:
            printf("%d\n", FnDivide(50,-5));
        break;
        case 10:
            printf("%d\n", FnTimes(-10,-2));
        break;
        case 11:
            printf("%d\n", FnTimes(2,10));
        break;
        case 12:
            printf("%d\n", FnTimes(5,-1));
        break;
        case 13:
            printf("%d\n", DefineAndRoundFraction(2,3));
            printf("%d\n", DefineAndRoundFraction(2,2));
        break;
        case 14:
            printf("%d\n", DefineAndRoundFraction(0,2));
        break;
   }
}
