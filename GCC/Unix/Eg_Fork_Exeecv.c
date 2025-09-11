#include <stdio.h>
#include <unistd.h>
int main(int argc, int *argv[])
{
    int pid;
    pid = fork();
    if (pid < 0)
    {
        exit(0);
    }
    else if (pid == 0)
    {
        char *newcode;
        if (argc < 2)
        {
            exit(1);
        }
        newcode = argv[1];
        execv(newcode, argv + 1);
        printf("Error!\n");
        exit(1);
    }
    else
    {
        sleep(2);
        printf("Dine\n");
        system("ls");
        exit(0);
    }
}