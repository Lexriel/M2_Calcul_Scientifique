# include "date.h"

int main()
{
  time_t timestamp;

      timestamp = time(NULL);
      t = localtime(&timestamp);

      printf("The processor 0 sent the date \"%s, %s the %d %d\".\n", day[t->tm_wday], month[t->tm_mon], t->tm_mday, 1900 + t->tm_year);

  return 0;
}
