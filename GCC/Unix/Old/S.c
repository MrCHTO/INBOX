#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

int main()
{
    int ssockfd, csockfd;
    int sl, cl;
    struct sockaddr_in server_address;
    struct sockaddr_in client_address;
    char ss[256];
    char *ans = "OK!";

    ssockfd = socket(AF_INET, SOCK_STREAM, 0);

    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_address.sin_port = htons(9734);
    sl = sizeof(server_address);
    bind(ssockfd, (struct sockaddr *)&server_address, sl);

    listen(ssockfd, 5);
    while (1)
    {
        char ch;

        printf("server waiting...\n");

        cl = sizeof(client_address);
        csockfd = accept(ssockfd,
                         (struct sockaddr *)&client_address, &cl);

        printf("client join\n");
        printf("client_add = %s\tclient_port = %u\n", inet_ntoa(client_address.sin_addr), htons(client_address.sin_port));

        read(csockfd, ss, sizeof(ss));
        printf("client send: %s\n", ss);
        write(csockfd, ans, sizeof(ans));
        close(csockfd);
    }
    close(ssockfd);
}
