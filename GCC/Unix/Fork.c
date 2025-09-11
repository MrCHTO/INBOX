//进程
#include <stdio.h>
#include <unistd.h>
int main()
{
    int parent_pid, child_pid, old_parent_pid, new_parent_pid;
    parent_pid = getpid();
    child_pid = fork();
    if (child_pid < 0)
    {
        printf("Create chile errno!\n");
    }
    else if (child_pid == 0)
    {
        old_parent_pid = getppid();
        sleep(2);
        //父进程被释放，子进程由systemd接管，pid=1
        new_parent_pid = getppid();
    }
    else
    {
        sleep(1);
        exit(0);
    }
    printf("%d\n%d\n%d\n%d\n", parent_pid, child_pid, old_parent_pid, new_parent_pid);
    exit(0);
}