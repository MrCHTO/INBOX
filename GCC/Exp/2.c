#include <stdio.h>
#include <math.h>
#define float double
void main()
{
    /*考生在此设计程序*/
    int n, sign = 1;
    double s = 1, t = 1;
    for (n = 1; n <= 18; n++)
    {
        sign = -sign;
        t *= 3;
        s = s + sign * t / (2 * n + 1);
    }
    printf("%.5lf\n", s);
    system("pause");
}
