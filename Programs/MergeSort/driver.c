

#include <stdio.h>


int g_list[100], g_aux[100], g_aux2[100];

void driver(int tc_number, int argc, char *argv[])
{
int i;

    switch (tc_number)
    {
        case 0:
            for (i = 1; i < argc; i++)
                    g_list[i-1] = atoi(argv[i]);
                    
            mergeSort(g_list, argc-1);
            for (i = 0; i < argc-1 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        break;
        case 1:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            arraycopy(g_list, 0, g_aux, i / 2, i - i / 2);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_aux[i]);
            printf("\n");
        break;
        case 2:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            arraycopy(g_list, 0, g_aux, i / 2, -i - i / 2);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_aux[i]);
            printf("\n");
        break;
        case 3:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            arraycopy(g_list, 0, g_aux, 0, i / 2);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_aux[i]);
            printf("\n");
        break;
        case 4:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            arraycopy(g_list, 0, g_aux, -1, i / 2);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_aux[i]);
            printf("\n");
        break;
        case 5:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            arraycopy(g_list, -1, g_aux, -1, i / 2);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_aux[i]);
            printf("\n");
        break;
        case 6:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            mergeSort(g_list, -argc-1);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        break;
        case 7:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            merge(g_list, -argc-1, g_list, -argc-1, g_list);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        break;
        case 8:
        {
            int p[] = {0, 1, 2, 3}, q[] = {5, 6, 7};
            merge(p, 4, q, 3, g_list);
            for (i = 0; i < 7 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        }
        break;
        case 9:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            merge(g_list, -argc-1, g_list, argc/2, g_list);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        break;
        case 10:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            merge(g_list, argc/2, g_list, -argc-1, g_list);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        break;
        case 11:
            for (i = 3; i < argc; i++)
                    g_list[i-3] = atoi(argv[i]);
            i = argc - 3;
            merge(g_list, 0, g_list, argc-3/2, g_list);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
            merge(g_list, argc-3/2, g_list, 0, g_list);
            for (i = 0; i < argc-3 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        break;
        case 12:
        {
            int n1, n2, i, j;
            n1 = atoi(argv[3]);
            for (i = 0; i < n1; i++)
                    g_aux[i] = atoi(argv[i+4]);
            n2 = atoi(argv[i+4]);
            for (j = 0; j < n2; j++)
                    g_aux2[j] = atoi(argv[i+5+j]);

            merge(g_aux, n1, g_aux2, n2, g_list);
            for (i = 0; i < n1 + n2 ; i++)
                    printf("%d ", g_list[i]);
            printf("\n");
        }
        break;
    }
}
