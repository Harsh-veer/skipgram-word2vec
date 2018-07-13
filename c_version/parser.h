#define MAX_WORD_LENGTH 15
#define MAX_VOCAB_SIZE 50000000

long long int file_size;
struct vocab_word{
  char *word;
  int cnt;
};
struct vocab_word *vocab;
long long int vocab_size = 0;
const int vocab_hash_size = 30000000;
int *vocab_hash;
long long train_words = 0;
int min_count = 3;

int getHash(char *word){
  unsigned long long int hash = 0;
  int i=0;
  for(i=0;i<strlen(word);i++) hash = hash * 257 + word[i];
  hash = hash%vocab_hash_size;
  return hash;
}

void readWord(char *word, FILE *fin){
  int a = 0;
  char ch;
  while(!feof(fin)){
    ch = fgetc(fin);
    if (ch=='\n' || ch==' ' || ch=='\t'){
      if (a>0){
        if (ch=='\n')ungetc(ch,fin);
        break;
      }
      if (ch=='\n'){
        strcpy(word,(char *)"</s>");
        return;
      }else continue;
    }
    word[a] = ch;
    a++;
    if (a>=MAX_WORD_LENGTH-1) a--;
  }
  word[a] = 0;
}

int searchVocab(char *word){
  int hash = getHash(word);
  while (1){
    if (vocab_hash[hash]==-1) return -1;
    if ( strcmp(vocab[vocab_hash[hash]].word, word)==0 ) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
}

void addToVocab(char *word){
  strcpy(vocab[vocab_size].word,word);
  vocab[vocab_size].cnt=1;
  vocab_size++;
  int hash = getHash(word);
  while(vocab_hash[hash]!=-1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size-1;
}

void createVocab(FILE *fin){
  addToVocab( (char *)"</s>" );

  char word[MAX_WORD_LENGTH];
  while(!feof(fin)){
    if (vocab_size >= MAX_VOCAB_SIZE){
      vocab_size = MAX_VOCAB_SIZE-1;
      break;
    }
    readWord(word,fin);
    int pos=searchVocab(word);
    if (pos==-1){
      addToVocab(word);
    }
    else{
      vocab[pos].cnt+=1;
    }
  }
}

int vocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cnt - ((struct vocab_word *)a)->cnt;
}

int readWordIndex(FILE *fin){
  char word[MAX_WORD_LENGTH];
  readWord(word, fin);
  if (feof(fin)) return -1;
  return searchVocab(word);
}

void sortVocab(){
  int a,size;
  unsigned int hash;
  qsort(&vocab[1], vocab_size-1, sizeof(struct vocab_word), vocabCompare);
  for (a=0; a<vocab_hash_size;a++) vocab_hash[a] = -1;
  size=vocab_size;
  train_words = 0;
  for (a=0; a<size; a++){
    if ( (vocab[a].cnt < min_count) && (a!=0) ){
      vocab_size--;
      free(vocab[a].word);
    }else{
      hash=getHash(vocab[a].word);
      while(vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cnt;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

void VocabFromTrainFile(){
  FILE *fin;
  fin = fopen("raw.txt","r");
  if (fin==NULL){
    printf("Training file not found\n");
    exit(0);
  }
  fseek(fin,0,SEEK_END);
  file_size = (long long int)ftell(fin);
  fseek(fin,0,SEEK_SET);

  printf("Reading file, creating vocabulary...\n");
  createVocab(fin);
  sortVocab();
  printf("Vocab:%lld\n",vocab_size);
}
