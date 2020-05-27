#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char *argv[])
{
    if (argc != 2) {
        perror("need a file argument");
        return -1;
    }

    // open() gives the system a call to identify a file on disk
    // describe its location with an int, called a "file descriptor (fd)",
    // and give certain permissions to the program
    int fd = open(argv[1], O_RDONLY, S_IWUSR | S_IRUSR);
    struct stat sb;

    // fstat() reads a file descriptor and tries to get its length in bytes as a long int
    if (fstat(fd, &sb) == -1) {
        perror("bad file");
        return -1;
    }

    printf("file size is %lld \n", sb.st_size);

    // mmap asks the OS to provision a chunk of disk storage out to contiguous (read aligned, coalesced) RAM
    // this is the reverse of using 'swap space' to cache some RAM out to disk when under memory pressure
    // we give it the fd file descriptor and the size of the file to tell the OS which chunk of disk to allocate as memory
    // and also give it certain permissions
    // this is a direct array of data so we can cast it to whatever form we like, in this case bytes
    // and then we can address the pointer as an array as we're familiar with
    uint8_t *file_in_memory = (uint8_t *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    printf("****PRINTING FILE AS BYTES****\n\n");

    for (int i = 0; i < sb.st_size; i++) {
        printf("%x", file_in_memory[i]);
    }
    printf("\n\n");

    printf("****PRINTING FILE AS CHARS****\n\n");

    for (int i = 0; i < sb.st_size; i++) {
        printf("%c", (char)file_in_memory[i]);
    }
    printf("\n\n");

    printf("****PRINTING FILE AS INTS****\n\n");

    for (int i = 0; i < sb.st_size; i++) {
        printf("%d", (int)file_in_memory[i]);
    }
    printf("\n\n");

    // if at any point we wanted to break this out as a 2d array, we just need to have a predetermined line length
    // and could then index into it as a 2d array by using offsets

    // this is like free() and fclose() but for mmap() and open()
    munmap(file_in_memory, sb.st_size);
    close(fd);

    return 0;
}
