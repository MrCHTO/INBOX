#include <stdio.h>
#include <stdlib.h>
int test[100];
int num = 0, len = 0;
void Quick_Core(int arr[], int start, int end)
{
    int left, right, temp, mid;
    if (start > end)
    {
        return;
    }
    mid = arr[start];
    left = start;
    right = end;
    while (left != right)
    {
        while (arr[right] >= mid && left < right)
        {
            right--;
        }
        while (arr[left] <= mid && left < right)
        {
            left++;
        }
        if (arr[left] >= arr[right])
        {
            temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
        }
    }
    // left==right
    arr[start] = arr[left];
    arr[left] = mid;
    Quick_Core(arr, start, left - 1);
    Quick_Core(arr, left + 1, end);
}
void Quick_Sort(int arr[], int length)
{
    Quick_Core(arr, 0, length - 1);
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
    Quick_Sort(test, num);
    for (int i = 0; i < num; i++)
    {
        printf("%d ", test[i]);
    }
    printf("\n");
    system("pause");
}
