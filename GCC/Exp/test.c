#include <stdio.h>
int c(int a, int b)
{
	return a + b;
}
int main()
{
	int x, y;
	scanf("%d%d", &x, &y);
	printf("%d", c(x, y));
}
