#include <stdio.h>
#include <string.h>
#include <unistd.h>

//llllhhrrrrrhrrrCCWCCWCCWCCWCCWh
char* moves[] = {"left", "left", "left", "left", "hard", "hard", "right", "right", "right", "right", "right", "hard", "right", "right", "right", "ccw", "ccw", "ccw", "ccw", "ccw", "hard", "hard"};
int main()
{
    printf("yeetus\n");
    printf("option seed 1\n");
    printf("ready\n");
    fflush(stdout);
    char* str;
    size_t len;
    getline(&str, &len, stdin);
    if (strcmp(str, "ack\n")) return -1;
    int x = 0;
    int y = 0;
    while (1)
    {
        getline(&str, &len, stdin);
        if (!strcmp(str, "move\n"))
        {
            if (x >= 22) printf("move hard\n");
            else printf("move %s\n", moves[x]);
            fflush(stdout);
            x++;
            while (1)
            {
                getline(&str, &len, stdin);
                if (!strncmp(str, "repeatedmove", 12))
                {
                    sleep(1);
                    printf("ready\n");
                    fflush(stdout);
                    break;
                }
                if (!strcmp(str, "gameover\n"))
                {
                    if (y < 1)
                    {
                        printf("option seed 1\n");
                        printf("option garbage 1-5 2-6\n");
                        printf("ready\n");
                        fflush(stdout);
                        y++;
                        x = 0;
                        break;
                    }
                    else
                    {
                        printf("kill\n");
                        fflush(stdout);
                        break;
                    }
                }
            }
        }
    }
}
