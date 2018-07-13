CC=gcc
CFLAGS= -lm -pthread
DEPS = unigram.h parser.h
OBJ = skipgram.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

skipgram: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
	rm -rf $(OBJ)

clean:
	rm -rf skipgram
