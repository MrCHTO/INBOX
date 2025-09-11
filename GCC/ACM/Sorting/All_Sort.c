#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void Bubble_Sort(int arr[], int length)
{
    int temp;
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
void Bubble_New_Sort(int arr[], int length)
{
    int temp, flag;
    for (int i = 0; i < length; i++)
    {
        flag = 1;
        for (int j = 0; j < length - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                flag = 0;
            }
        }
        if (flag == 1)
        {
            break;
        }
    }
}
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
void Select_Sort(int arr[], int length)
{
    int min, temp;
    for (int i = 0; i < length - 1; i++)
    {
        min = i;
        for (int j = i + 1; j < length; j++)
        {
            if (arr[min] > arr[j])
            {
                min = j; //把值最小的下标记下来
            }
        }
        if (min != i)
        {
            temp = arr[i];
            arr[i] = arr[min];
            arr[min] = temp;
        }
    }
}

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
    int Random_Arr[50000];
    int Random_Backup[50000];
    int length, range;
    clock_t S_BS, E_BS, S_NBS, E_NBS, S_IS, E_IS, S_QS, E_QS, S_SS, E_SS;
    printf("please input the length:");
    scanf("%d", &length);
    printf("please input the range:");
    scanf("%d", &range);
    Random_Number(Random_Arr, length, range);
    memcpy(Random_Backup, Random_Arr, sizeof(int) * length);
    S_BS = clock();
    Bubble_Sort(Random_Arr, length);
    E_BS = clock();
    for (int i = 0; i < length; i++)
    {
        printf("%d ", Random_Arr[i]);
    }
    printf("\n");
    memcpy(Random_Arr, Random_Backup, sizeof(int) * length);
    S_NBS = clock();
    Bubble_New_Sort(Random_Arr, length);
    E_NBS = clock();
    for (int i = 0; i < length; i++)
    {
        printf("%d ", Random_Arr[i]);
    }
    printf("\n");
    memcpy(Random_Arr, Random_Backup, sizeof(int) * length);
    S_IS = clock();
    Insert_Sort(Random_Arr, length);
    E_IS = clock();
    for (int i = 0; i < length; i++)
    {
        printf("%d ", Random_Arr[i]);
    }
    printf("\n");
    memcpy(Random_Arr, Random_Backup, sizeof(int) * length);
    S_QS = clock();
    Quick_Sort(Random_Arr, length);
    E_QS = clock();
    for (int i = 0; i < length; i++)
    {
        printf("%d ", Random_Arr[i]);
    }
    printf("\n");
    memcpy(Random_Arr, Random_Backup, sizeof(int) * length);
    S_SS = clock();
    Select_Sort(Random_Arr, length);
    E_SS = clock();
    for (int i = 0; i < length; i++)
    {
        printf("%d ", Random_Arr[i]);
    }
    printf("\n");
    memcpy(Random_Arr, Random_Backup, sizeof(int) * length);
    printf("Bubble_Sort time is %f s\n", ((double)E_BS - S_BS) / CLK_TCK);
    printf("Bubble_New_Sort time is %f s\n", ((double)E_NBS - S_NBS) / CLK_TCK);
    printf("Insert_Sort time is %f s\n", ((double)E_IS - S_IS) / CLK_TCK);
    printf("Quick_Sort time is %f s\n", ((double)E_QS - S_QS) / CLK_TCK);
    printf("Select_Sort time is %f s\n", ((double)E_SS - S_SS) / CLK_TCK);
    //system("pause");
}