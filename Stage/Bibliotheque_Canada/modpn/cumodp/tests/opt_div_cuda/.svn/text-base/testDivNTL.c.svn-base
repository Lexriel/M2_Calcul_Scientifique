#include <NTL/lzz_p.h>
#include <NTL/lzz_pX.h>



NTL_CLIENT

int main(int argc, char *argv[]){
  int deg, prime, deg_2;

  if(argc > 1) deg = atoi(argv[1]);
  if(argc > 2) deg_2 = atoi(argv[2]);
  if(argc > 3) prime = atoi(argv[3]);

  zz_p::init(prime);
  zz_pX a, b, c;
  zz_p deg_p;

  random(a, deg);
  random(b, deg_2);

  double u = GetTime();

  div(c, a, b);

  u = GetTime() - u;

  cout<<deg<<" "<<deg_2<<" "<<u<< endl;
 return 0;
}
