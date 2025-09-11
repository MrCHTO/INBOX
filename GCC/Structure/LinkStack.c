#include <stdio.h>
#include <stdlib.h>
//节点结构定义
typedef struct node
{
    int data;
    struct node *next;
} Stacknode, *PStacknode;
//栈结构的定义
typedef struct
{
    PStacknode top;
} LinkStack, *PLinkStack;
//初始化栈
PLinkStack Init_LinkStack(void)
{
    PLinkStack s;
    s = (PLinkStack)malloc(sizeof(LinkStack));
    if (s)
    {
        s->top = NULL;
    }
    return s;
}
//判断栈
int Empty_LinkStack(PLinkStack s)
{
    return (s->top == NULL);
}
//入栈
int Push_LinkStack(PLinkStack s, int x)
{
    PStacknode p;
    p = (PStacknode)malloc(sizeof(Stacknode));
    if (!p)
    {
        printf("内存溢出！\n");
        return (0);
    }
    p->data = x;
    p->next = s->top;
    s->top = p;
    return (1);
}
//出栈
int Pop_LinkStack(PLinkStack s, int *x)
{
    PStacknode p;
    if (Empty_LinkStack(s))
    {
        printf("栈空！\n");
        return (0);
    }
    *x = s->top->data;
    p = s->top;
    s->top = s->top->next;
    free(p);
    return (1);
}
//取栈顶元素
int GetTop_LinkStack(PLinkStack s, int *x)
{
    if (Empty_LinkStack(s))
    {
        printf("栈空！\n");
        return (0);
    }
    *x = s->top->data;
    return (1);
}
//销毁栈
void Destory_LinkStack(PLinkStack *LS)
{
    PStacknode p, q;
    if (*LS)
    {
        p = (*LS)->top;
        while (p)
        {
            q = p;
            p = p->next;
            free(q);
        }
        free(*LS);
    }
    *LS = NULL;
    return;
}
int main()
{
    PLinkStack s;
    int x;
    int y = 9;
    s = Init_LinkStack();
    printf("%d\n", Empty_LinkStack(s));
    Push_LinkStack(s, y);
    printf("%d\n", Empty_LinkStack(s));
    GetTop_LinkStack(s, &x);
    printf("%d\n", x);
    Destory_LinkStack(&s);
}