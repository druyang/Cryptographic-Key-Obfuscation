/**
 * generates a binary file from unsigned integer text
 *
 * Usage: ./binary-generator [input file] [output file]
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(const int argc, const char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: ./binary-generator [inputfile] [outputfile]\n");
        return -1;
    }

    FILE *readfile;
    if ((readfile = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error: input file not readable\n");
        return -2;
    }

    FILE *writefile;
    if ((writefile = fopen(argv[2], "wb")) == NULL) {
        fprintf(stderr, "Error: input file not readable\n");
        return -2;
    }

    char line[100];
    int32_t i;
    for (;;) {
        fgets(line, sizeof(line), readfile);
        if (feof(readfile)) break;
        i = (int32_t)atoi(line);
        fwrite(&i, sizeof(i), 1, writefile);
    }
    fclose(writefile);
    fclose(readfile);
}
