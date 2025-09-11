#include <stdio.h>
#include <unistd.h>
/*
int execve(const *filename, const char *argv[], const char *envp[])
*/
int main(int argc, int *argv[])
{
    char *newcode;
    if (argc < 3)
    {
        exit(1);
    }
    newcode = argv[1];
    execv(newcode, argv + 2);
    printf("Error!\n");
    exit(1);
}