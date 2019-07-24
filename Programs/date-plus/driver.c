


#include <stdio.h>

extern int argc;
extern char **argv;

void driver(int tc_number, int Argc, char *Argv[])
{
int i;

    switch (tc_number)
    {
        case 0:
            argc = Argc;
            argv = Argv;
            incrdate();
            printdate();
            break;
    }
}

