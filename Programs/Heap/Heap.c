/*************************************************************************
 *  Compilation:  javac Heap.java
 *  Execution:    java Heap < input.txt
 *  Dependencies: StdOut.java StdIn.java
 *  Data files:   http://algs4.cs.princeton.edu/24pq/tiny.txt
 *                http://algs4.cs.princeton.edu/24pq/words3.txt
 *  
 *  Sorts a sequence of strings from standard input using heapsort.
 *
 *  % more tiny.txt
 *  S O R T E X A M P L E
 *
 *  % java Heap < tiny.txt
 *  S O R T E X A M P L E A               [ one string per line ]
 *
 *  % more words3.txt
 *  bed bug dad yes zoo ... all bad yet
 *
 *  % java Heap < words3.txt
 *  all bad bed bug dad ... yes yet zoo   [ one string per line ]
 *
 *************************************************************************/
#include <stdio.h>
/***********************************************************************
* Helper functions for comparisons and swaps.
* Indices are "off-by-one" to support 1-based indexing.
**********************************************************************/
int less01(char *pq[], int i, int j) {
	if(strcmp(pq[i-1],pq[j-1]) < 0)
		return 1;
	else return 0;
}



void exch(char *pq[], int i, int j) {
    char *swap;
	swap = pq[i-1];
    pq[i-1] = pq[j-1];
    pq[j-1] = swap;
}

/* is v < w ?*/
int less02(char *v, char *w) {
	if(strcmp(v,w) < 0)
		return 1;
	else return 0;
}



/***********************************************************************
* Helper functions to restore the heap invariant.
**********************************************************************/
void sink(char *pq[], int k, int N) {
int j;

    while (2*k <= N) {
        j = 2*k;
        if (j < N && less01(pq, j, j+1)) j++;
        if (!less01(pq, k, j)) break;
        exch(pq, k, j);
        k = j;
    }
}


void sort(char *pq[],int length)
{
	int k;
	for (k=length/2;k>=1;k--)
		sink(pq,k,length);
	while (length>1)
	{
		exch(pq,1,length--);
		sink(pq,1,length);
	}

}



/***********************************************************************
*  Check if array is sorted - useful for debugging
***********************************************************************/
int isSorted(char *a[],int length) {
	int i ;
    for (i = 1; i < length; i++)
        if (less02(a[i], a[i-1])) return 0;
    return 1;
}


/* print array to standard output*/
void show(char *a[],int length) {
	int i;
    for (i = 0; i < length; i++) {
		printf("%s\n",a[i]);
    }
}


#define  MAX 	25
char *a[MAX];

/* Read strings from standard input, sort them, and print.*/
void main(int argc, char *argv[]) {
int i;
	for(i=1;i<argc;i++) 
	{
		a[i-1] = argv[i]; 
	} 

	printf("%d\n", isSorted(a, argc-1) );
	sort(a,argc-1);
	printf("***************** Sorted Strings *****************\n");
        show(a,argc-1);
}
