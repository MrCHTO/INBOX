#include <stdio.h>
#include <stdlib.h>
int test[100];
int num = 0, len = 0;
void Insert_Sort(int arr[], int length)
{
    for (int i = 1; i < length; i++)
    {
        int temp = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > temp)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}
int main()
{
    printf("please input number:\n");
    scanf("%d", &num);
    for (int i = 0; i < num; i++)
    {
        printf("please input the %d:\n", i + 1);
        scanf("%d", &test[i]);
    }
    Insert_Sort(test, num);
    for (int i = 0; i < num; i++)
    {
        printf("%d ", test[i]);
    }
    printf("\n");
    system("pause");
}
