#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>

int main()
{
    int sfd, l;
    struct sockaddr_in add;
    int result;
    char ss[256];
    char rs[256];

    sfd = socket(AF_INET, SOCK_STREAM, 0);

    add.sin_family = AF_INET;
    add.sin_addr.s_addr = inet_addr("127.0.0.1");
    add.sin_port = htons(9734);
    l = sizeof(add);

    result = connect(sfd, (struct sockaddr *)&add, l);

    if (result == -1)
    {
        perror("oops: client");
        exit(1);
    }

    printf("send your message: ");
    fflush(stdout);
    fgets(ss, sizeof(ss), stdin);

    write(sfd, ss, sizeof(ss));
    read(sfd, rs, sizeof(rs));
    printf("server send = %s\n", rs);
    close(sfd);
    exit(0);
}
