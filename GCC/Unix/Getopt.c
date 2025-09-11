#include <stdio.h>
#include <unistd.h>
/*
int getopy(int argc, char *const argv[], const char *optstring)
{
    extern char *optarg;
    extern int optind, opterr, optopt;
}
*/
int main()
{
    int c;
    while ((c = getopt(argc, argv, ":abf:")) != -1)
}