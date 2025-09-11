#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h> //windows t=ms
// #include <unistd.h> //linux t=s
int main()
{
    clock_t start, end;
    start = clock();
    Sleep(3000);
    end = clock();
    printf("time = %f \n", ((double)end - start) / CLK_TCK);
    system("pause");
}