#include <stdio.h>
#include <stdlib.h>
typedef struct node
{
    int date;
    struct node *next;
} node;
typedef struct queue
{
    node *head;
    node *tail;
    int length;
} queue;
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
queue *init_queue()
{
    printf("Start initializing the queue... ----------");
    queue *q = (queue *)malloc(sizeof(queue));
    q->length = 0;
    if (q == NULL)
    {
        printf("Failed to initialize the queue\n");
        exit(0);
    }
    q->head = NULL;
    q->tail = NULL;
    printf("The initialization queue is complete\n");
    return q;
}
int empty(queue *q)
{
    if (q->head == NULL)
    {
        printf("The queue is empty！\n");
        return 1;
    }
    else
    {
        printf("The queue is not empty！\n");
        return 0;
    }
}
void push(queue *q)
{
    int data;
    node *n = init_node();
    printf("Please enter the node Data:");
    scanf("%d", &data);
    n->date = data;
    n->next = NULL;
    q->length++;
    if (empty(q))
    {
        q->head = n;
        q->tail = n;
        printf("Header node added successfully！\n");
    }
    else
    {
        q->tail->next = n;
        q->tail = n;
        printf("node added successfully ！\n");
    }
}
void pop(queue *q)
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
    queue *q;
    q = init_queue();
    printf("\n");
    for (int i = 0; i < nn; i++)
    {
        push(q);
        printf("\n");
    }
    printf("you have %d node in this queue! \n", q->length);
    for (int i = 0; i < nn; i++)
    {
        pop(q);
        printf("\n");
    }
    system("pause");
}
