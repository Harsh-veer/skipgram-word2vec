#include "parser.h"

long long int table_size = 100000000;
int *table;

void initUnigram(){
  printf("Creating unigram table\n");
  double train_words_pow = 0;
  double d1,power = 0.75;
  table = (int *)malloc(table_size*sizeof(int));
  int a=0;
  for(a=0;a<vocab_size;a++) train_words_pow += pow(vocab[a].cnt, power);
  int i=0;
  d1 = pow(vocab[i].cnt, power)/train_words_pow;

  for(a=0;a<table_size;a++){
    table[a]=i;
    if( a/(double)table_size > d1 ){
      i++;
      d1 += pow(vocab[i].cnt,power)/train_words_pow;
    }
    if( i >= vocab_size ) i = vocab_size-1;
  }
}
