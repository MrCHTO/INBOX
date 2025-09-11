#include <stdio.h>
#include <string.h>
struct node
{
    char name[23];
    int qimo;
    int banji;
    char ganbu, xibu;
    int lunwen;
} a[110];
int main()
{
    int i, j, t, n, sum[110], max, SUM;

    memset(sum, 0, sizeof(sum));
    scanf("%d", &n);
    for (i = 0; i < n; i++)
    {
        scanf("%s %d %d %c %c %d", a[i].name, &a[i].qimo, &a[i].banji, &a[i].ganbu, &a[i].xibu, &a[i].lunwen);
        if (a[i].qimo > 80 && a[i].lunwen > 0)
            sum[i] += 8000;
        if (a[i].qimo > 85 && a[i].banji > 80)
            sum[i] += 4000;
        if (a[i].qimo > 90)
            sum[i] += 2000;
        if (a[i].qimo > 85 && a[i].xibu == 'Y')
            sum[i] += 1000;
        if (a[i].banji > 80 && a[i].ganbu == 'Y')
            sum[i] += 850;
    }
    max = sum[0];
    SUM = 0;
    for (i = 0; i < n; i++)
    {
        SUM += sum[i];
        if (max < sum[i])
        {
            max = sum[i];
            j = i;
        }
    }
    printf("%s\n%d\n%d\n", a[j].name, max, SUM);
    system("pause");
}
