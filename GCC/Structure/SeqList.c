#include <stdio.h>
#include <stdlib.h>
#define MAXSIZE 100
//顺序表封装
typedef struct node
{
    int Data[MAXSIZE];
    int length;
} SeqList, *PSeqList;
//顺序表初始化
PSeqList Init_SeqList(void)
{
    PSeqList PL;
    PL = (PSeqList)malloc(sizeof(SeqList));
    if (PL)
    {
        PL->length = 0;
    }
    printf("顺序表初始化成功！\n");
    return PL;
}
//求顺序表长度
int Length_List(SeqList L)
{
    printf("求顺序表长度成功！\n");
    return (L.length);
}
//顺序表检索
int Location_SeqList(SeqList L, int x)
{
    int i = 0;
    while (i < L.length && L.Data[i] != x)
    {
        i++;
    }
    if (i >= L.length)
    {
        return 0;
    }
    else
    {
        printf("顺序表检索成功！\n");
        return i + 1;
    }
}
//顺序表的插入
int Insert_SeqList(PSeqList PL, int i, int x)
{
    int j;
    if (PL == 0)
    {
        printf("表不存在！\n");
        return (-2);
    }
    if (PL->length >= MAXSIZE)
    {
        printf("表溢出！\n");
        return (-1);
    }
    if (i < 1 || PL->length >= MAXSIZE)
    {
        printf("插入位置不合法！\n");
        return (0);
    }
    for (j = PL->length - 1; j >= i - 1; j--)
    {
        PL->Data[j + 1] == PL->Data[j];
    }
    PL->Data[i - 1] = x;
    PL->length++;
    printf("顺序表的插入成功！\n");
    return (1);
}
//顺序表的删除
int Delete_SeqList(PSeqList PL, int i)
{
    int j;
    if (PL == 0)
    {
        printf("表不存在！\n");
        return (-1);
    }
    if (i < 1 || PL->length >= MAXSIZE)
    {
        printf("删除位置不合法！\n");
        return (0);
    }
    for (j = i; j < PL->length - 1; j++)
    {
        PL->Data[j - 1] == PL->Data[j];
    }
    PL->length--;
    printf("顺序表的删除成功！\n");
    return (1);
}
//顺序表的操作
int main()
{
    PSeqList a = Init_SeqList();
    printf("%d\n", Length_List(*a));
    Insert_SeqList(a, 1, 1);
    printf("%d\n", Length_List(*a));
}