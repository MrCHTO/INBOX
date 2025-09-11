#include <stdio.h>
#include <stdlib.h>
typedef struct node
{
    int date;
    struct node *next;
} node;
node *init_node()
{
    printf("Start initializing nodes... ----------");
    node *n = (node *)malloc(sizeof(node));
    if (n == NULL)
    {
        printf("Failed to initialize node\n");
        exit(0);
    }
    printf("The initialization node is complete\n");
    return n;
}
int empty(node *phead)
{
    if (phead == NULL)
    {
        printf("The list is empty！\n");
        return 1;
    }
    else
    {
        printf("The list is not empty ！\n");
        return 0;
    }
}
node *creat_LinkList(int n)
{
    node *phead, *pend;
    phead = init_node();
    pend = init_node();
    printf("\n");
    phead = NULL;
    pend = NULL;
    for (int i = 0; i < n; i++)
    {
        node *n = init_node();
        printf("Please enter the node Data：");
        scanf("%d", &n->date);
        n->next = NULL;
        if (empty(phead))
        {
            phead = n;
            pend = n;
            printf("Header node added successfully！\n");
        }
        else
        {
            pend->next = n;
            pend = n;
            printf("node added successfully！\n");
        }
        printf("\n");
    }
    return phead;
}
int main()
{
    node *LinkList, *pmov;
    int number;
    printf("please input number:");
    scanf("%d", &number);
    pmov = LinkList = creat_LinkList(number);
    while (pmov != NULL)
    {
        printf("%d ", pmov->date);
        pmov = pmov->next;
    }
    printf("\n");
    system("pause");
}