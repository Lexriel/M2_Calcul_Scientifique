# include "tarot.h"

int main()
{
  int i;
  tarot_card t[NB_PLAYERS]={{CLUB,11},{TRUMP,12},{DIAMOND,QUEEN},{TRUMP,5}};
  tarot_card t2[NB_PLAYERS]={{CLUB,9},{CLUB,2},{DIAMOND,JACK},{SPADE,5}};
  tarot_card t3[NB_PLAYERS]={{CLUB,8},{SPADE,2},{HEART,QUEEN},{EXCUSE}};
  tarot_card c={DIAMOND,14};
  won p[NB_PLAYERS];
  for (i=0; i<NB_PLAYERS; i++)
    {
      p[i].nb=0;
      (p+i)->owe_one=0;
    }
  print_card(t3[3]);
  c=best_card(t[2],c,DIAMOND);
  print_card(c);
  c=best_card(t2[1],t2[0],CLUB);
  print_card(c);
  c=t[trick_winner(t,HEART)];
  print_card(c);
  c=t2[trick_winner(t2,CLUB)];
  print_card(c);
  c=t2[trick_winner(t2,DIAMOND)];
  print_card(c);
  trick_winner_and_store(t,p,0);
  trick_winner_and_store(t2,p,0);
  trick_winner_and_store(t3,p,4);
  for (i=0; i<NB_PLAYERS; i++)
    print_won(p[i]);
  c=t2[trick_winner(t2,TRUMP)];
  print_card(c);
  return 0;
}
