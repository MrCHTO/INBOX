#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
int main(){
	printf("%d\n",EXIT_SUCCESS);
	printf("%d\n",EXIT_FAILURE);
	printf("fater pid is %d\n",getpid());
	pid_t pid;
	pid = fork();
	printf("son pid is %d\n",pid);
	if(pid>0){
		printf("father\n");
		exit(1);
	}
	else if(pid==0){
		printf("son\n");
		exit(1);
	}
	else{
		printf("error\n");
		exit(1);
	}
	exit(EXIT_SUCCESS);
}

