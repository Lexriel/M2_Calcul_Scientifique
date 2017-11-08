# ifndef LIBRARY_1
# define LIBRARY_1

# include <stdlib.h>
# include <stdio.h>
# include <string.h>

struct book_s{
  char* title;
  char* author;
  int num;
};

typedef struct book_s book;

typedef book* pointer_book;

extern void readbooks(int, FILE*, pointer_book*);
extern void findnumber(int, pointer_book*);
extern void findtitle(char[], pointer_book*);
extern void findauthor(char[], pointer_book*);
extern int t_ret(char[], pointer_book*);
extern int a_ret(char[], pointer_book*);
extern void insertbook(pointer_book*);
extern void suppressbook(pointer_book*);

# endif
