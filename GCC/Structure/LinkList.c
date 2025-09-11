#include <stdio.h>
#include <stdlib.h>
//节点申明
typedef struct node
{
    int data;
    struct node *next;
} Lnode, *PLnode;
//创建头节点
PLnode Create_LinkList(void)
{
    PLnode H;
    H = (PLnode)malloc(sizeof(Lnode));
    if (H)
    {
        H->next = NULL;
    }
    return H;
}
//销毁单链表
void Destroy_LinkList(PLnode *H)
{
    PLnode p, q;
    p = *H;
    while (p)
    {
        q = p;
        p = p->next;
        free(q);
    }
    *H = NULL;
}
//求表长
int Length_LinkList(PLnode H)
{
    PLnode p = H;
    int count = -1;
    while (p)
    {
        p = p->next;
        count++;
    }
    return (count);
}
//按照序号查找操作
PLnode Local_LinkList_Pos(PLnode H, int i)
{
    PLnode p = H;
    int j = 0;
    while (p && j < i)
    {
        p = p->next;
        j++;
    }
    if (j != i || !p)
    {
        printf("参数i不存在或者单链表不存在！\n");
        return (NULL);
    }
    return p;
}
//按照值查找操作
PLnode Local_LinkList_Value(PLnode H, int x)
{
    PLnode p = H->next;
    while (p && p->data != x)
    {
        p - p->next;
    }
    return p;
}
//插入
PLnode Insert_LinkList(PLnode H, int i, int x)
{
    PLnode p, q;
    p = Local_LinkList_Pos(H, i - 1);
    if (!p)
    {
        printf("i有误！\n");
        return (0);
    }
    q = (PLnode)malloc(sizeof(Lnode));
    if (!q)
    {
        printf("申请空间失败！\n");
        return (0);
    }
    q->data = x;
    q->next = p->next;
    p->next = q;
    return (1);
}
//删除
int Del_LinkList(PLnode H, int i)
{
    PLnode p, q;
    if (H == NULL || H->next == NULL)
    {
        printf("链表不存在或者空表不能删除！\n");
        return (0);
    }
    p = Local_LinkList_Pos(H, i - 1);
    if (p == NULL || p->next == NULL)
    {
        printf("参数i错误！\n");
        return (0);
    }
    q = p->next;
    p->next = q->next;
    free(q);
    return (1);
}
//链表的操作
int main()
{
    PLnode Head;
    Head = Create_LinkList();
    printf("%d\n", Length_LinkList(Head));
    Destroy_LinkList(&Head);
}