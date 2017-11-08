# ifndef FRENCH_TAROT_1_H
# define FRENCH_TAROT_1_H

enum color {heart=1, spade=2, diamond, club, trump, excuse};

typedef enum color color;

enum value {as=1, jockey=11, cavalier, queen, roy};

typedef enum value value;

struct french_tarot
{ 
  color col;
  value val;
};

typedef struct french_tarot card;

int is_it_an_oudler(card);

# endif
