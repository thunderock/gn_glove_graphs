#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make
# if [ ! -e text8 ]; then
#   if hash wget 2>/dev/null; then
#     wget http://mattmahoney.net/dc/text8.zip
#   else
#     curl -O http://mattmahoney.net/dc/text8.zip
#   fi
#   unzip text8.zip
#   rm text8.zip
# fi
# declare -A arguments
# arguments=([dim]=16 [para]=8)
DIMENSION=16
PARA=8
CORPUS=text8
#CORPUS=$2
VOCAB_FILE=vocab16.txt
COOCCURRENCE_FILE=cooccurrence"$DIMENSION".bin
COOCCURRENCE_SHUF_FILE=cooccurrence"$DIMENSION".shuf.bin
BUILDDIR=build
SAVE_FILE=vectors"$DIMENTION"-"$PARA"
VERBOSE=2
MEMORY=500.0
VOCAB_MIN_COUNT=0
VECTOR_SIZE=16
MAX_ITER=100
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=100
VOCAB_HASH_FILE=hashdump"$DIMENTION".txt
MALE_WORD_FILE=../wordlist/male_word_file_new.txt
FEMALE_WORD_FILE=../wordlist/female_word_file_new.txt

python ../scripts/random_walk.py -n 1000000 -o $CORPUS -d airport -w $FEMALE_WORD_FILE -m $MALE_WORD_FILE

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -hash-dump-file $VOCAB_HASH_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -vocab-hash-file $VOCAB_HASH_FILE -male-word-file $MALE_WORD_FILE -female-word-file $FEMALE_WORD_FILE
#       if [[ $? -eq 0 ]]
#       then
#           if [ "$1" = 'matlab' ]; then
#               matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
#           elif [ "$1" = 'octave' ]; then
#               octave < ./eval/octave/read_and_evaluate_octave.m 1>&2 
#           else
               # python2 eval/python/evaluate.py --vocab_file $VOCAB_FILE --vectors_file $SAVE_FILE".txt"
#           fi
#       fi
    fi
  fi
fi
