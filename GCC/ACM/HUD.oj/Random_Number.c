#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void Random_Number(int arr[], int length, int range)
{
    srand((unsigned int)time(NULL));
    for (int i = 0; i < length; i++)
    {
        arr[i] = rand() % range + 1;
    }
}
int main()
{
    int a[10];
    Random_Number(a, 10, 10);
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
    system("pause");
}