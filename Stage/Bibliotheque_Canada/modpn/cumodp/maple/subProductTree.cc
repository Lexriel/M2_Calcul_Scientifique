
#include<iostream>
#include <ctime>
#include<cmath>

#define BASE_1 31

using namespace std;



typedef int            sfixn;
typedef unsigned int   usfixn;





void getPrimeLevel(int *p, int *k)
{
	ifstream ifs( "primeLevel.dat", ifstream::in  ); 
	
	ifs>>*p;
	ifs>>*k;

	ifs.close();

}


int main(int argc, char *argv[])
{	
	int p, k;	

	getPrimeLevel(&p, &k);

	
	return 0;
}
	
	
