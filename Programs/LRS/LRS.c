/*************************************************************************
 *  Compilation:  javac LRS.java
 *  Execution:    java LRS < file.txt
 *  Dependencies: StdIn.java
 *  
 *  Reads a text corpus from stdin, replaces all consecutive blocks of
 *  whitespace with a single space, and then computes the longest
 *  repeated substring in that corpus. Suffix sorts the corpus using
 *  the system sort, then finds the longest repeated substring among 
 *  consecutive suffixes in the sorted order.
 * 
 *  % java LRS < mobydick.txt
 *  ',- Such a funny, sporty, gamy, jesty, joky, hoky-poky lad, is the Ocean, oh! Th'
 * 
 *  % java LRS 
 *  aaaaaaaaa
 *  'aaaaaaaa'
 *
 *  % java LRS
 *  abcdefg
 *  ''
 *
 *************************************************************************/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX	100


int minn(int a, int b)
{
	return a > b ? b : a;
}

void sort(char *a[],int sum) /*sort func*/ 
{
	char *temp; 
	int i,j; 
	for(i=0;i<sum-1;i++) 
		for(j=i+1;j<sum;j++) 
			if(strcmp(a[i],a[j])>0) 
			{
				temp=a[i]; 
				a[i]=a[j]; 
				a[j]=temp; 
			} 
} 




/*sub String*/ 
char *substr(char *str,int star,int len)
{
	char *st;
	int l;
	l = len - star;
	st = malloc(sizeof(char) * (l+1));
	strncpy(st, &str[star], l);
	st[l] = '\0';
	return st;
}

/* return the longest common prefix of s and t*/
char *lcp(char s[], char t[]) {
    int i = 0, n;

    n = minn(strlen(s), strlen(t));
    for (i = 0; i < n; i++) {
        if (s[i] != t[i])
		{
			return substr(s, 0, i);
		}
     }
     return substr(s, 0, n);
}



/* return the longest repeated string in s*/
char *LRS(char s[]) {

    /* form the N suffixes*/
	char *pp[MAX], *lrs = "", *p;
        int N  = strlen(s);
        int i=0;
	for (i = 0; i < N; i++) 
	{   
		p = substr(s,i,N);
		pp[i] = p;
	}

	sort(pp,N);

	for (i = 0; i < N-1; i++)
	{
		char *str;
		
		str = lcp(pp[i], pp[i+1]);
		
		if(strlen(str) > strlen(lrs))
		{
			lrs = str;
		}

	}
        return lrs;
}


/* read in text, replacing all consecutive whitespace with a single space
    then compute longest repeated substring*/
void main(int argc, char *argv[]) {
  
/*
	String s = StdIn.readAll();
    s = s.replaceAll("\\s+", " ");
    StdOut.println("'" + lrs(s) + "'");
*/
	char * s;
	if (argc <= 1 )
		s = "";
	else
		s = argv[1];

	printf("String: %s\n",LRS(s));

}
