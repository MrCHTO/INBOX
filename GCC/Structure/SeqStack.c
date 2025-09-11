#include <stdio.h>
#include <stdlib.h>
#define MAXSIZE 100
//栈的封装
typedef struct
{
    int data[MAXSIZE];
    int top; //指明当前栈顶位置
} SeqStack, *PSeqStack;
//栈的初始化
PSeqStack Init_SeqStack(void)
{
    PSeqStack s;
    s = (PSeqStack)malloc(sizeof(SeqStack));
    if (s)
    {
        s->top = -1;
        return s;
    }
}
//判断栈是否为空
int Empty_SeqStack(PSeqStack s)
{
    if (s->top == -1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
//入栈
int Push_SeqStack(PSeqStack s, int x)
{
    if (s->top == MAXSIZE - 1)
    {
        return 0;
    }
    else
    {
        s->top++;
        s->data[s->top] = x;
        return 1;
    }
}
//出栈
int Pop_SeqStack(PSeqStack s, int *x)
{
    if (Empty_SeqStack(s))
    {
        return 0;
    }
    else
    {
        *x = s->data[s->top];
        s->top--;
        return 1;
    }
}
//取栈顶元素
int GetTop_SeqStack(PSeqStack s, int *x)
{
    if (Empty_SeqStack(s))
    {
        return 0;
    }
    else
    {
        *x = s->data[s->top];
        return 1;
    }
}
//销毁栈
void Destory_SeqStack(PSeqStack *s)
{
    if (*s)
    {
        free(*s);
    }
    *s = NULL;
    return;
}
int main()
{
    PSeqStack s;
    int x;
    int y = 9;
    s = Init_SeqStack();
    printf("%d\n", Empty_SeqStack(s));
    Push_SeqStack(s, y);
    printf("%d\n", Empty_SeqStack(s));
    GetTop_SeqStack(s, &x);
    printf("%d\n", x);
    Destory_SeqStack(&s);
}