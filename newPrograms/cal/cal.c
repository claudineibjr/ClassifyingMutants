/* Introduction to Software Testing
// Authors: Paul Ammann & Jeff Offutt
// Chapter 5, section 5.2, page 190
*/

#include <stdio.h>

      /*Skip month 0. */
int daysIn[] = {0, 31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

/***********************************************************
// Calculate the number of Days between the two given days in
// the same year.
// preconditions : day1 and day2 must be in same year
//               1 <= month1, month2 <= 12
//               1 <= day1, day2 <= 31
//               month1 <= month2
//               The range for year: 1 ... 10000
//***********************************************************/

int cal (int month1, int day1, int month2,
                       int day2, int year)
{
   int numDays, i;

   if (month2 == month1) /* in the same month */
      numDays  = day2 - day1;
   else
   {
      /* Are we in a leap year? */
      int m4 = year % 4;
      int m100 = year % 100;
      int m400 = year % 400;
      if ((m4 != 0) || ((m100 == 0) && (m400 != 0)))
         daysIn[2] = 28;
      else
         daysIn[2] = 29;

      /* start with days in the two months */
      numDays = day2 + (daysIn[month1] - day1);

      /* add the days in the intervening months */
      for ( i = month1 + 1; i <= month2-1; i++)
         numDays = daysIn[i] + numDays;
   }
   return (numDays);
}

void main (int argc, char *argv[])
{  /* Driver program  for cal */
   int month1, day1, month2, day2, year, T;

   month1 = atoi(argv[1]);
   day1 = atoi(argv[2]);
   month2 = atoi(argv[3]);
   day2 = atoi(argv[4]);
   year = atoi(argv[5]);

   T = cal (month1, day1, month2, day2, year);

   printf("Result is: %d  \n", T);
}

