# ifndef TAROT_H
# define TAROT_H

# include <stdio.h>
# include <stdlib.h>

# define NB_PLAYERS 4

enum color_e {CLUB, DIAMOND, HEART, SPADE, TRUMP, EXCUSE};

typedef enum color_e color;

enum {JACK=11, KNIGHT, QUEEN, KING};

typedef struct tarot_card_s
{
  color tc_color;
  int tc_value;
} tarot_card;

typedef struct won_player_s
{
  tarot_card cards[78];
  int nb;
  int owe_one; /* this is for dealing with the excuse ; 1 means one has to give a simple card, -1 means we're suppose to get one */
} won;

extern int is_an_oudler(tarot_card);
extern tarot_card best_card(tarot_card, tarot_card, color);
extern int trick_winner(tarot_card[], color);
extern void add2won(tarot_card[], won*, int);
extern void trick_winner_and_store(tarot_card[], won[], int);
extern void print_card(tarot_card);
extern void print_won(won);

# endif /* TAROT_H */
