#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<pthread.h>
#include "unigram.h"

#define MAX_SENTENCE_LENGTH 1000

const int window = 5;
const long long int feature_size = 100;
double alpha = 0.025, starting_alpha = 0.0;
const int negK = 5;
const float thresh = 0.001;
float *W1, *W2;
int next_random = 1;
int num_threads = 12;
int word_count_actual = 0;
int iter = 5;
int classes = 100;

float sgm(float x){
  if (x > 10) return 1.0;
  else if (x < -10) return 0.0;
  else return 1/(1 + exp(-x));
}

void initNet(){
  printf("Initializing network\n");
  int x = posix_memalign((void **)&W1, 128, vocab_size * feature_size * sizeof(float));
  int y = posix_memalign((void **)&W2, 128, vocab_size * feature_size * sizeof(float));

  for(x=0;x<vocab_size;x++)
    for(y=0;y<feature_size;y++)
      W2[x * feature_size + y] = 0;

  for(x=0;x<vocab_size;x++){
    for(y=0;y<feature_size;y++){
      next_random= next_random* (unsigned long long)25214903917 + 11;
      W1[x * feature_size + y] = ((next_random& 0xFFFF)/(float)65536) - 0.5;
    }
  }

}

void *trainThread(void *id){
  // printf("chkpt3\n");
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1,l2,c,target,label,local_iter = iter;
  unsigned long long next_random = (long long)id;
  float f,g;
  float *de = (float *)calloc(feature_size, sizeof(float)); // stores derivative of W2, and some other products
  FILE *fn = fopen("raw.txt","r");
  fseek(fn, file_size/num_threads * (long long)id, SEEK_SET);
  while (1){
    // printf("chkpt4\n");
    if (word_count - last_word_count > 10000){
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;

      alpha = starting_alpha * (1 - word_count_actual / (float)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0){
      while(1){
        word = readWordIndex(fn);
        if (feof(fn)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0)break;
        float ran = (sqrt(vocab[word].cnt/(thresh * train_words)) + 1) * (thresh * train_words) / vocab[word].cnt;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        if(ran < (next_random & 0xFFFF) / (float)65536) continue;
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fn) || (word_count > train_words / num_threads)){
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter==0)break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fn, file_size/(long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word==-1) continue;
    for(c=0; c<feature_size;c++) de[c]=0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    for (a = b; a < window*2 + 1 - b; a++) if(a!=window){
      c = sentence_position - window + a;
      if (c<0 || c>=sentence_length) continue;
      last_word = sen[c];
      if (last_word == -1)continue;
      l1 = last_word * feature_size;
      for (c=0;c<feature_size;c++)de[c]=0;
      if (negK > 0){
        for(d=0;d<negK + 1;d++){ // +1 for actual context, rest for negative samples
          if (d==0){
            target = word;
            label = 1;
          }else{
            next_random = next_random * (unsigned long long)25214903917 + 11;
            long long indx  = (next_random >> 16) % table_size;
            target = table[indx]; // <--- caused segmentation error
            if (target==0) target = next_random % (vocab_size - 1) + 1;
            if (target==word) continue;
            label = 0;
          }
          l2 = target * feature_size;
          f = 0;
          for (c=0;c<feature_size;c++) f += W1[c + l1] * W2[c + l2];
          g = ( label - sgm(f) ) * alpha;
          for (c=0;c<feature_size;c++) de[c] += g * W2[c + l2];
          for (c=0;c<feature_size;c++) W2[c + l2] += g * W1[c + l1];
        }
        for (c=0;c<feature_size;c++) W1[c + l1] += de[c];
      }
    }
    sentence_position++;
    if(sentence_position >= sentence_length){
      sentence_length = 0;
      continue;
    }
  }
  fclose(fn);
  free(de);
  pthread_exit(NULL);
}

void trainNet(){
  long long int a, b;
    starting_alpha = alpha;
  pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  VocabFromTrainFile();
  initNet();
  if (negK > 0) initUnigram();
    printf("Training\n");
  for (a = 0; a < num_threads; a++) pthread_create(&threads[a], NULL, trainThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(threads[a], NULL);

  printf("Training done, saving embeddings to file\n");
  FILE *fo;
  fo = fopen("embeddings", "w");
  fprintf(fo, "%lld %lld\n",vocab_size, feature_size);
  for (a=0;a<vocab_size;a++){
    fprintf(fo,"%s ",vocab[a].word);
    for(b=0;b<feature_size;b++)fprintf(fo, "%lf", W1[a * feature_size + b]);
    fprintf(fo, "\n");
  }

  fclose(fo);

  printf("Computing K-means\n");
  int c=0,d=0;
  fo = fopen("Kclasses","w");
  int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    float closev, x;
    float *cent = (float *)calloc(classes * feature_size, sizeof(float));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * feature_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < feature_size; d++) cent[feature_size * cl[c] + d] += W1[c * feature_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < feature_size; c++) {
          cent[feature_size * b + c] /= centcn[b];
          closev += cent[feature_size * b + c] * cent[feature_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < feature_size; c++) cent[feature_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < feature_size; b++) x += cent[feature_size * d + b] * W1[c * feature_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);

    fclose(fo);
}

int main(int *argc, int **argv){
  vocab = (struct vocab_word *)calloc(MAX_VOCAB_SIZE, sizeof(struct vocab_word));
  int i=0;
  for (i=0;i<MAX_VOCAB_SIZE;i++){
    vocab[i].word = (char *)malloc(MAX_WORD_LENGTH * sizeof(char));
  }

  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  for(i=0;i<vocab_hash_size;i++)vocab_hash[i]=-1;

  trainNet();

  return 0;
}
