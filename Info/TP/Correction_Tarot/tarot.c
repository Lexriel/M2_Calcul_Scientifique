# include "tarot.h"

int is_an_oudler(tarot_card c)
{
  return ( (c.tc_color == EXCUSE) || ( (c.tc_color == TRUMP) && ( (c.tc_value == 1) || (c.tc_value == 21) ) ) ) ? 1 : 0;
}

tarot_card best_card(tarot_card a, tarot_card b, color asked)
{
  if (a.tc_color == TRUMP)
    {
      if (b.tc_color == TRUMP)
	{
	  if (a.tc_value>b.tc_value)
	    return a;
	  if (a.tc_value<b.tc_value)
	    return b;
	  printf("Trying to compare two same cards\n");
	  exit(EXIT_FAILURE);
	}
      return a;
    }
  if (b.tc_color == TRUMP)
    return b;
  if (a.tc_color == asked)
    {
      if (b.tc_color == asked)
	{
	  if (a.tc_value>b.tc_value)
	    return a;
	  if (a.tc_value<b.tc_value)
	    return b;
	  printf("Trying to compare two same cards\n");
	  exit(EXIT_FAILURE);
	}
      return a;
    }
  if (b.tc_color == asked)
    return b;
  printf("Trying to compare two cards but none of them has the correct color\n");
  exit(EXIT_FAILURE);
}

int trick_winner(tarot_card trick[], color asked)
{
  tarot_card winner;
  int is_winner=-1;
  int seen=0;
  while (seen != NB_PLAYERS)
    {
      if (is_winner != -1)
	{
	  winner=best_card(trick[seen],winner,asked);
	  if ( ( ( winner.tc_color == trick[seen].tc_color) && ( winner.tc_value == trick[seen].tc_value ) ) )
	    is_winner=seen;
	}
      else
	if ( (trick[seen].tc_color == asked) || ( trick[seen].tc_color == TRUMP ) )
	  {
	    is_winner=seen;
	    winner=trick[seen];
	  }
      seen++;
    }
  if (is_winner != -1)
    return is_winner;
  printf("None of the card can win the trick\n");
  exit(EXIT_FAILURE);
}

void add2won(tarot_card t[], won *p, int n)
{
  int i,j=p->nb;
  for (i=0; i<n; i++)
    p->cards[j+i]=t[i];
  p->nb+=n;
  return;
}

void trick_winner_and_store(tarot_card c[], won p[], int first)
{
  color asked = c[first].tc_color == EXCUSE ? c[(first+1)%NB_PLAYERS].tc_color : c[first].tc_color;
  int i, j, win = trick_winner(c,asked), excuse = 0;
  tarot_card cc[NB_PLAYERS-1];
  for (i=0; i<NB_PLAYERS; i++)
    if (c[i].tc_color == EXCUSE)
      {
	excuse=1;
	add2won(c+i,p+i,1);
	(p+i)->owe_one=1;
	for (j=0; j<i; j++)
	  cc[j]=c[j];
	for (j=i+1; j<NB_PLAYERS; j++)
	  cc[j-1]=c[j];
	add2won(cc,p+win,NB_PLAYERS-1);
	(p+win)->owe_one=-1;
	break;
      }
  if (!excuse)
    add2won(c,p+win,NB_PLAYERS);
  return;
}

void print_card(tarot_card t)
{
  int c=t.tc_color, v=t.tc_value;
  switch (c){
  case 0: printf("CLUB "); break;
  case 1: printf("DIAMOND "); break;
  case 2: printf("HEART "); break;
  case 3: printf("SPADE "); break;
  case 4: printf("TRUMP "); break;
  default: printf("EXCUSE\n");
  }
  if (c<=3)
    {
      switch (v){
      case 11: printf("JACK\n"); break;
      case 12: printf("KNIGHT\n"); break;
      case 13: printf("QUEEN\n"); break;
      case 14: printf("KING\n"); break;
      default:	printf("%d\n",v);
      }
    }
  if (c==4)
    printf("%d\n",v);
  return;
}

void print_won(won p)
{
  int i;
  printf("This player won %d cards. They are:\n",p.nb);
  for (i=0; i<p.nb; i++)
    print_card(p.cards[i]);
  if (p.owe_one == 1)
    printf("He's supposed to give back one card\n");
  if (p.owe_one == -1)
    printf("He's supposed to get back one more card\n");
}

