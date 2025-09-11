#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int Box[100][2] = {'0'};
float Ratio[100] = {'0'};
int Number = 0;
int Range = 0;
int Capacity = 0;
int Sum = 0;
float Answer[100][3] = {'0'};

void Bubble_Sort1(float arr[])
{
    int temp_i;
    float temp_f;
    for (int i = 0; i < Number; i++)
    {
        for (int j = 0; j < Number - i - 1; j++)
        {
            if (arr[j] < arr[j + 1])
            {
                {
                    temp_f = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp_f;
                }
                {
                    temp_i = Box[j][0];
                    Box[j][0] = Box[j + 1][0];
                    Box[j + 1][0] = temp_i;
                }
                {
                    temp_i = Box[j][1];
                    Box[j][1] = Box[j + 1][1];
                    Box[j + 1][1] = temp_i;
                }
                if (arr[j] == arr[j + 1] && Box[j][1] < Box[j + 1][1])
                {
                    {
                        temp_i = Box[j][0];
                        Box[j][0] = Box[j + 1][0];
                        Box[j + 1][0] = temp_i;
                    }
                    {
                        temp_i = Box[j][1];
                        Box[j][1] = Box[j + 1][1];
                        Box[j + 1][1] = temp_i;
                    }
                }
            }
        }
    }
}
void Bubble_Sort2(int arr[][2])
{
    int temp_i;
    float temp_f;
    for (int i = 0; i < Number; i++)
    {
        for (int j = 0; j < Number - i - 1; j++)
        {
            if (arr[j][1] > arr[j + 1][1])
            {
                {
                    temp_i = arr[j][1];
                    arr[j][1] = arr[j + 1][1];
                    arr[j + 1][1] = temp_i;
                }
                {
                    temp_i = arr[j][0];
                    arr[j][0] = arr[j + 1][0];
                    arr[j + 1][0] = temp_i;
                }
                {
                    temp_f = Ratio[j];
                    Ratio[j] = Ratio[j + 1];
                    Ratio[j + 1] = temp_f;
                }
            }
            if (arr[j][1] == arr[j + 1][1] && Ratio[j] < Ratio[j + 1])
            {
                {
                    temp_i = Box[j][0];
                    Box[j][0] = Box[j + 1][0];
                    Box[j + 1][0] = temp_i;
                }
                {
                    temp_f = Ratio[j];
                    Ratio[j] = Ratio[j + 1];
                    Ratio[j + 1] = temp_f;
                }
            }
        }
    }
}
void Random_Number(int arr[][2], int length, int range)
{
    srand((unsigned int)time(NULL));
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            arr[i][j] = rand() % range + 1;
        }
    }
}
void Create_Question(int arr[][2])
{
    printf("please input the article number:");
    scanf("%d", &Number);
    printf("please input the article range:");
    scanf("%d", &Range);
    printf("please input the Capcity:");
    scanf("%d", &Capacity);
    Random_Number(arr, Number, Range);
    printf("done... \n");
    printf("\nForm:\n");
    for (int j = 0; j < 2; j++)
    {
        if (j == 0)
        {
            printf("Value:\t");
        }
        if (j == 1)
        {
            printf("Weight:\t");
        }
        for (int i = 0; i < Number; i++)
        {
            printf("%.2f\t", (float)Box[i][j]);
        }
        printf("\n");
    }
}
void Solvation_Ratio(int arr1[][2], float arr2[])
{
    for (int i = 0; i < Number; i++)
    {
        arr2[i] = (float)arr1[i][0] / (float)arr1[i][1];
    }
    printf("Ratio:\t");
    for (int i = 0; i < Number; i++)
    {
        printf("%.2f\t", Ratio[i]);
    }
}
void Show_form()
{
    for (int j = 0; j < 2; j++)
    {
        if (j == 0)
        {
            printf("Value:\t");
        }
        if (j == 1)
        {
            printf("Weight:\t");
        }
        for (int i = 0; i < Number; i++)
        {
            printf("%.2f\t", (float)Box[i][j]);
        }
        printf("\n");
    }
    printf("Ratio:\t");
    for (int i = 0; i < Number; i++)
    {
        printf("%.2f\t", Ratio[i]);
    }
    printf("\n");
}
void Show_answer()
{
    Sum = 0;
    int k = 0;
    for (int i = 0; i < Number; i++)
    {
        Sum = Sum + Box[i][1];
        if (Sum <= Capacity)
        {
            Answer[k][0] = Box[i][0];
            Answer[k][1] = Box[i][1];
            Answer[k][2] = Ratio[i];
            k++;
        }
        if (Sum > Capacity)
        {
            Sum = Sum - Box[i][1];
        }
    }
    for (int j = 0; j < 3; j++)
    {
        if (j == 0)
        {
            printf("Value:\t");
        }
        if (j == 1)
        {
            printf("Weight:\t");
        }
        if (j == 2)
        {
            printf("Ratio:\t");
        }
        for (int i = 0; i < k; i++)
        {
            printf("%.2f\t", Answer[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    clock_t start, end;

    Create_Question(Box);

    start = clock();

    Solvation_Ratio(Box, Ratio);
    printf("\n");

    Bubble_Sort2(Box);
    printf("\n");

    printf("Weight first:");
    printf("\n");
    Show_form();
    printf("\n");
    printf("Weight first answer is:\n");
    Show_answer();
    printf("\n");

    Bubble_Sort1(Ratio);

    printf("Ratio first:");
    printf("\n");
    Show_form();
    printf("\n");
    printf("Ratio first answer is:\n");
    Show_answer();
    printf("\n");

    end = clock();
    printf("the time is: %f ms\n", ((double)end - start));
    system("pause");
}