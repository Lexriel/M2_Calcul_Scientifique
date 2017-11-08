# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include "library1.h"
# define SIZE_MAX 9999


void readbooks(int m, FILE* fp, pointer_book* library)
{
  char c1[30], c2[30];
  int n, i;
  if ((m>SIZE_MAX) || (m<0))
    {
      printf("error");
      exit(1);
    }

  else
    {
      for(i=0; i<m; i++)
	{
	  library[i]=(pointer_book) malloc(sizeof(book));
	  library[i]->author=(char*) malloc(30*sizeof(char));
	  library[i]->title=(char*) malloc(30*sizeof(char));

	  /* first possility */
	  fscanf(fp, "%d %s %s", &n, c1, c2);
	  strcpy(library[i]->author,c1);
	  strcpy(library[i]->title,c2);
	  library[i]->num = n;

	  /* second possibility:
	  fscanf(fp, "%d %s %s", &n, library[i]->author, library[i]->title);
	  library[i]->num = n;
	                      */

	}
    }

}

void findnumber(int k, pointer_book* library)
{
  if((k>SIZE_MAX) || (k<0))
    {
      printf("error");
      exit(1);
    } 
  printf("The book number %d you are looking for is:\n %d %s %s \n", library[k]->num, library[k]->num, library[k]->author, library[k]->title);
}


void findauthor(char author[30], pointer_book* library)
{
  int found=0, i=0;
  while ((i<=SIZE_MAX) && (found == 0))
    {
      if (strcmp(library[i]->author, author) == 0)
	{
	  found=1;
	  printf("The %s's book you are looking for is:\n %d %s %s \n", library[i]->author, library[i]->num, library[i]->author, library[i]->title);
	}
      i++;
    }
  if (found==0)
    printf("error");
}

int a_ret(char a[30], pointer_book* library)
{
  int found=0, i=0;
  while ((i<=SIZE_MAX) && (found == 0))
    {
      if (strcmp(library[i]->author, a) == 0)
	{
	  found=1;
	}
      i++;
    }
  if (found==0)
    printf("error");
  return library[i]->num;
}

void findtitle(char title[30], pointer_book* library)
{
  int found=0, i=0;
  while ((i<=SIZE_MAX) && (found == 0))
    {
      if (strcmp(library[i]->title, title) == 0)
	{
	  found=1;
	  printf("The book untitled %s you are looking for is:\n %d %s %s \n", library[i]->title, library[i]->num, library[i]->author, library[i]->title);
	}
      i++;
    }
  if (found==0)
    printf("error");
}

int t_ret(char t[30], pointer_book* library)
{
  int found=0, i=0;
  while ((i<=SIZE_MAX) && (found == 0))
    {
      if (strcmp(library[i]->title, t) == 0)
	{
	  found=1;
	}
      i++;
    }
  if (found==0)
    printf("error");
	  return library[i]->num;
}

void insertbook(pointer_book* library)
{
  int m, i=0;
  char a[30], t[30];
  printf("You want to insert a book.\n Give me his number, his author and his title:\n (separated by spaces) \n");
  scanf("%d %s %s", &m, a, t);
  while((library[i] != NULL) && (i<SIZE_MAX))
    i++;
  if(i!=SIZE_MAX)
    {
      library[i]=(pointer_book) malloc(sizeof(book));
      library[i]->author=(char*) malloc(30*sizeof(char));
      library[i]->title=(char*) malloc(30*sizeof(char));
      
      library[i]->num=m;
      library[i]->author=a;
      library[i]->title=t;
      printf("Congratulations ! You managed to insert your first book !\n");
    }
  else
    printf("Sorry, there is no place for your book.\n");
  return ;
}

void suppressbook(pointer_book* library)
{
  char a1, t1;
  int p1, p2, q;
  printf("Do you want to remove a book ? (yes:1, no: other letters) \n");
  scanf("%d",&q);
 if (q != 1)
    exit(1);
  printf("Are your sure ?\n");
  scanf("%d", &q);
  if (q != 1)
    exit(1);
  printf("Very sure ?\n");
  scanf("%d", &q);
  if (q != 1)
    exit(1);
  printf("Really ?\n");
  scanf("%d", &q);
  if (q != 1)
    exit(1);
  printf("Ok.\n");

  printf("To remove a book, you have the following choices:\n 1: To give its number.\n 2: To give its title.\n 3: To give its author.\n 4: To hit Valentin.\n (key 1, 2,3 or 4)\n");
  scanf("%d", &p1);
  switch (p1)
    { 
    case 1 :
      {
	printf("Number of the book ?\n");
	scanf("%d", &p2);
	free(library[p2]->author);
	free(library[p2]->title);
	free(library[p2]);
	library[p2]->author=NULL;
	library[p2]->title=NULL;
	printf("The book number %d is dead... :-( \n",p2);
	break;
      }
    case 2 :
      {
	printf("Title of the book ?\n");
	scanf("%s", &t1);
	p2=t_ret("t1", library);
	free(library[p2]->author);
	free(library[p2]->title);
	free(library[p2]);
	library[p2]->author=NULL;
	library[p2]->title=NULL;
	printf("The book number %d is dead... :-( \n",p2);
	break;
      }
    case 3 :
      {
	printf("Author of the book ?\n");
	scanf("%s", &a1);
	p2=a_ret("a1", library);
	free(library[p2]->author);
	free(library[p2]->title);
	free(library[p2]);
	library[p2]->author=NULL;
	library[p2]->title=NULL;
	printf("The book number %d is dead... :-( \n",p2);
	break;
      }
    case 4 :
      {
	printf("Bim, bam, boum !");
	break;
      }
    default :
      {
	printf("You don't know how to use a PC, go out !");
	break;
      }
    }
}


int main()
{
  pointer_book* library;
  FILE* fp;
  library = (pointer_book*) calloc(10000, sizeof(pointer_book));
  fp=fopen("biglib.txt","r");
  readbooks(100, fp, library);
  findnumber(12, library);
  findtitle("moiitnck", library);
  findauthor("DRVMHLUBBR", library);
  //  insertbook(library);
  suppressbook(library);
  fclose(fp);
  return 0;
}

/*
int main()
{
  FILE * fp;
  fp=fopen("biglib.txt","r");
  readbooks(10,fp);
  fclose(fp);
  return 0;
}
*/

