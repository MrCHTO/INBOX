#include <stdio.h>
#include <string.h>
int number = 0;
int dnumber = 0;
char password[100];
char dpassword[100];
void input()
{
    int i, k;
    printf("please input the password:\n");
    gets(password);
    number = strlen(password);
    printf("please input the key:\n");
    scanf("%d", &k);
    for (i = 0; i < number; i++)
    {
        if (password[i] == 32)
        {
            password[i] = 123;
        }
        if (password[i] > 64 && password[i] < 91)
        {
            password[i] = password[i] - 65;
            password[i] = password[i] + k;
            password[i] = password[i] % 52;
        }
        if (password[i] > 96 && password[i] < 123)
        {
            password[i] = password[i] - 71;
            password[i] = password[i] + k;
            password[i] = password[i] % 52;
        }
    }
    for (i = 0; i < number; i++)
    {
        if (password[i] >= 0 && password[i] <= 25)
        {
            password[i] = password[i] + 65;
        }
        if (password[i] >= 26 && password[i] <= 52)
        {
            password[i] = password[i] + 71;
        }
        if (password[i] == 123)
        {
            password[i] = 32;
        }
    }
    printf("the dpassword is:\n");
    puts(password);
    getchar();
}

void output()
{
    int i, k;
    printf("please input the dpassword:\n");
    gets(dpassword);
    dnumber = strlen(dpassword);
    printf("please input the key:\n");
    scanf("%d", &k);
    for (i = 0; i < number; i++)
    {
        if (dpassword[i] == 32)
        {
            dpassword[i] = 123;
        }
        if (dpassword[i] > 64 && dpassword[i] < 91)
        {
            dpassword[i] = dpassword[i] - 65;
            dpassword[i] = dpassword[i] - k;
            if (dpassword[i] < 0)
            {
                dpassword[i] = dpassword[i] + 52;
            }
        }
        if (dpassword[i] > 96 && dpassword[i] < 123)
        {
            dpassword[i] = dpassword[i] - 71;
            dpassword[i] = dpassword[i] - k;
        }
    }
    for (i = 0; i < number; i++)
    {
        if (dpassword[i] >= 0 && dpassword[i] <= 25)
        {
            dpassword[i] = dpassword[i] + 65;
        }
        if (dpassword[i] >= 26 && dpassword[i] <= 52)
        {
            dpassword[i] = dpassword[i] + 71;
        }
        if (dpassword[i] == 123)
        {
            dpassword[i] = 32;
        }
    }
    printf("the dpassword is:\n");
    puts(dpassword);
    getchar();
}

int main()
{
    input();
    printf("\n");
    output();
    system("pause");
}