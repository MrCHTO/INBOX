#include <stdio.h>
#include <string.h>
int number = 0, knumber = 0, dnumber = 0;
int nkey[100];
char key[100], password[100], dpassword[100];
void input()
{
    int i, j = 0;
    printf("please input the key:\n");
    gets(key);
    knumber = strlen(key);
    printf("please input the password:\n");
    gets(password);
    number = strlen(password);
    for (i = 0; i < knumber; i++)
    {
        nkey[i] = key[i] - 97;
    }
    for (i = 0; i < number; i++)
    {
        if (password[i] == 32)
        {
            password[i] = 123;
        }
        if (password[i] > 96 && password[i] < 123)
        {
            password[i] = password[i] - 97;
            password[i] = password[i] + nkey[j % knumber];
            password[i] = password[i] % 26;
            j++;
        }
    }
    for (i = 0; i < number; i++)
    {
        if (password[i] >= 0 && password[i] <= 25)
        {
            password[i] = password[i] + 97;
        }
        if (password[i] == 123)
        {
            password[i] = 32;
        }
    }
    printf("the dpassword is:\n");
    puts(password);
}
void output()
{
    int i, j = 0;
    printf("please input the key:\n");
    gets(key);
    knumber = strlen(key);
    printf("please input the dpassword:\n");
    gets(dpassword);
    dnumber = strlen(dpassword);
    for (i = 0; i < knumber; i++)
    {
        nkey[i] = key[i] - 97;
    }
    for (i = 0; i < dnumber; i++)
    {
        if (dpassword[i] == 32)
        {
            dpassword[i] = 123;
        }
        if (dpassword[i] > 96 && dpassword[i] < 123)
        {
            dpassword[i] = dpassword[i] - 97;
            dpassword[i] = dpassword[i] - nkey[j % knumber];
            if (dpassword[i] < 0)
            {
                dpassword[i] = dpassword[i] + 26;
            }
            j++;
        }
    }
    for (i = 0; i < dnumber; i++)
    {
        if (dpassword[i] >= 0 && dpassword[i] <= 25)
        {
            dpassword[i] = dpassword[i] + 97;
        }
        if (dpassword[i] == 123)
        {
            dpassword[i] = 32;
        }
    }
    printf("the password is:\n");
    puts(dpassword);
}
int main()
{
    input();
    printf("\n");
    output();
    getchar();
}
