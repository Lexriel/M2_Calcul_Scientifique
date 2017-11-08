main()
{
int x, p=0;
scanf("%lu",&x);
for(;x>0;p++) 
  x&=x-1;
printf("%d\n",p);
}
