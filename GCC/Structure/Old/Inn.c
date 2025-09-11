#include <stdio.h>
#include <stdlib.h>
typedef struct node
{
    int date;
    struct node *next;
} node;
typedef struct inn
{
    node *head;
    node *tail;
    int length;
} inn;
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
inn *init_inn()
{
    printf("Start initializing the inn... ----------");
    inn *q = (inn *)malloc(sizeof(inn));
    if (q == NULL)
    {
        printf("Failed to initialize the inn\n");
        exit(0);
    }
    q->head = NULL;
    q->tail = NULL;
    printf("The initialization inn is complete\n");
    return q;
}
int empty(inn *q)
{
    if (q->head == NULL)
    {
        printf("The inn is empty！\n");
        return 1;
    }
    else
    {
        printf("The inn is not empty！\n");
        return 0;
    }
}
void push(inn *q)
{
    int data;
    node *n = init_node();
    q->length++;
    printf("Please enter the node Data:");
    scanf("%d", &data);
    n->date = data;
    n->next = NULL;
    if (empty(q))
    {
        q->head = n;
        q->tail = n;
        printf("Header node added successfully！\n");
    }
    else
    {
        n->next = q->head;
        q->head = n;
        printf("node added successfully ！\n");
    }
}
void pop(inn *q)
{
    node *n = q->head;
    printf("The node data is: %d\n", n->date);
    if (empty(q))
    {
        printf("The queue is NULL！\n");
        return;
    }
    if (q->head == q->tail)
    {
        q->head = NULL;
        q->tail = NULL;
        free(n);
        printf("The node is released and empty！\n");
    }
    else
    {
        q->head = q->head->next;
        free(n);
        printf("Nodes are released！\n");
    }
    q->length--;
}
int main()
{
    int nn;
    printf("please input number:");
    scanf("%d", &nn);
    inn *q;
    q = init_inn();
    printf("\n");
    for (int i = 0; i < nn; i++)
    {
        push(q);
        printf("\n");
    }
    printf("you have %d node in this inn ! \n", q->length);
    for (int i = 0; i < nn; i++)
    {
        pop(q);
        printf("\n");
    }
    //system("pause");
}
