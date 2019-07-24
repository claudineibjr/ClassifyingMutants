#include <stdio.h>



void merge(int *list1,int list1len, int *list2, int list2len, int *temp);


void arraycopy(int *src,int srcstart,int *dest,int deststart, int length)
{
	int i = 0;
	while(i<length)
	{
		dest[deststart+i] = src[srcstart+i];
		i++;
	}

}




/** The method for sorting the numbers */
void mergeSort(int *list,int len) {
int firstHalf[len/2];
int secondHalfLength = len - len/2;
int secondHalf[secondHalfLength];
	if (len > 1) {
		/* Merge sort the first half*/

		arraycopy(list,0,firstHalf,0, len/2);

		mergeSort(firstHalf, len/2);

		/*  Merge sort the second half*/

		arraycopy(list, len/2, secondHalf, 0, secondHalfLength);

		mergeSort(secondHalf, secondHalfLength);

		/* Merge firstHalf with secondHalf into list*/
		merge(firstHalf, len/2, secondHalf,secondHalfLength, list);
	}
}

/** Merge two sorted lists */
void merge(int *list1,int list1len, int *list2, int list2len, int *temp) {
	int current1 = 0; /* Current index in list1*/
	int current2 = 0; /* Current index in list2*/
	int current3 = 0; /* Current index in temp*/

	while (current1 < list1len && current2 < list2len ) {
		if (list1[current1] < list2[current2])
			temp[current3++] = list1[current1++];
		else
			temp[current3++] = list2[current2++];
	}

	while (current1 < list1len) 
		temp[current3++] = list1[current1++];

	while (current2 < list2len) 
		temp[current3++] = list2[current2++];
}


/** A test method */
void main(int argc, char *argv[]) {
	
    if ( argc > 1 && strcmp("-", argv[1]) == 0 )
    {

        driver(atoi(argv[2]), argc, argv);
    }
    else
    {
        driver(0, argc, argv);
    }

}

