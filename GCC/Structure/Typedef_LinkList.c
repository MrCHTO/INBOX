#include <stdio.h>
#define MAXSIZE 100
typedef struct
{
    int data;
    int next;
} SNode;
typedef struct
{
    SNode sp[MAXSIZE];
    int SL;
} StList, *PStList;
