#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
int main(int argc, char *argv[])
{
	char buffer[BUFSIZ + 1];
	char file_des[2];
	pid_t cpid;
	char somed_ata[2];
	int flag = 0;
	memset(buffer, '\0', sizeof(buffer));
	if (argc == 3)
	{
		if (strcmp(argv[1], "cat") == 0)
		{
			flag = 0;
		}
		else
		{
			flag = 1;
		}
		if (flag == 0)
		{
			cpid = fork();
			if (cpid == -1) //Creat Error
			{
				exit(0);
			}
			else if (cpid == 0) //Creat Child
			{
				write(file_des[1], argv, strlen(argv));
			}
			else //Creat Father
			{
				sprintf(buffer, "%s", file_des[0]);
				execvp("cat", buffer);
			}
		}
		else
		{
			exit(0);
		}
	}
	else
	{
		exit(0);
	}
}
