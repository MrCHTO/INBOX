#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>

int main(){
	FILE *rfp;
	char buffer[BUFSIZ + 1];
	int cread;
	memset(buffer,'\0',sizeof(buffer));
	rfp = popen("ls -l","r");
	if (rfp != NULL){
		cread = fread(buffer,sizeof(char),BUFSIZ,rfp);
		if (cread > 0){
			printf("output:\n%s\n",buffer);
		}
		pclose(rfp);
		exit(EXIT_SUCCESS);
	}
	exit(EXIT_FAILURE);
}
