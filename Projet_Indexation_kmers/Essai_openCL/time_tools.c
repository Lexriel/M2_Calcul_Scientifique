
#include <time.h>
#include <sys/time.h>

struct timeval tv1,tv2;
struct timezone tz;

void time_start()
{
  printf("\n== Time reset\n");
  gettimeofday(&tv1, &tz);
}


void time_stop()
{
  gettimeofday(&tv2, &tz);
  float diff=(tv2.tv_sec-tv1.tv_sec) + (float) (tv2.tv_usec-tv1.tv_usec) / 1000000L ;
  
  printf("== Wall-clock time: %3.3f s\n\n", diff);

}
