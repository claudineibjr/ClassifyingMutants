#include <stdio.h>


/*
Write a method to implement *, - , / operations. You should use only the + operator.
*/

/* Flip a positive sign to negative, or a negative sign to pos */
int FnNegate(int a) {
	int neg = 0;
	int d = a < 0 ? 1 : -1;
	while (a != 0) {
		neg += d;
		a += d;
	}
	return neg;
}

/* Subtract two numbers by negating b and adding them */
int FnMinus(int a, int b) {
	return a + FnNegate(b);
}

/* Check if a and b are different signs */
int DifferentSigns(int a, int b) {
	return ((a < 0 && b > 0) || (a > 0 && b < 0)) ? 1 : 0; 
}

/* Return absolute value */
int abss(int a) {
	if (a < 0) return FnNegate(a);
	else return a;
}

/* Multiply a by b by adding a to itself b times */
int FnTimes(int a, int b) {
	int sum = 0;
	int iter;
	if (a < b) return FnTimes(b, a); // algo is faster if b < a
	for (iter = abss(b); iter > 0; --iter){
		sum+=a;
	}
	if (b < 0) sum = FnNegate(sum);
	return sum;
}

/* returns 1, if a/b >= 0.5, and 0 otherwise */
int DefineAndRoundFraction(int a, int b) {
	if(FnTimes(abss(a), 2) >= abss(b)) return 1;
	else return 0;
}

/* Divide a by b by literally counting how many times does b go into
 * a. That is, count how many times you can subtract b from a until
 * you hit 0. */
int FnDivide(int a, int b){ 
	int quotient = 0;
	int divisor = 0;
	int divend; /* dividend */
	if (b == 0) {
		printf("ERROR: Divide by zero.");
		return 0;
	}
	divisor = FnNegate(abss(b));
	divend = abss(a);
	for (divend = abss(a); divend >= abss(divisor); divend += divisor)
	{
		++quotient;
	}

	if (DifferentSigns(a, b)==1) quotient = FnNegate(quotient);
	return quotient;
}
	

int main (int argc, char *argv[])
{ 
	driver(argc, argv);
	return 0;
}

//
//void main(int argc, char *argv[]) {
//
//	int i=0;
//	for (i = 0; i < 100; i++){
//		int a = randomInt(10);
//		int b = randomInt(10);
//		int ans = FnMinus(a, b);
//		if (ans != a - b) {
//			printf("ERROR");
//		}
//		printf("%d - %d = %d \n",a,b,ans);
//	}
//	int j=0;
//	for (j = 0; j < 100; j++){
//		int a = randomInt(10);
//		int b = randomInt(10);
//		int ans = FnTimes(a, b);
//		if (ans != a * b) {
//			printf("ERROR");
//		}
//		printf("%d * %d = %d \n",a,b,ans);
//	}
//
//
//	int k=0;
//	for (k = 0; k < 100; k++) {
//		int a = randomInt(10) + 1;
//		int b = randomInt(10) + 1;
//		printf("%d / %d = ",a,b);
//
//		int ans = FnDivide(a, b);
//
//		if (ans != a/b) {
//			printf("ERROR");
//
//		}
//		printf("%d\n",ans);
//
//
//	}
//}
