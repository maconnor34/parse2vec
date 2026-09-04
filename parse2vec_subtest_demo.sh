#!/bin/sh

train_file=$1
tune_file=$2
test_file=$3
form=$4
ppn5=$5
def6=$6

dictionary_end=$7
if [ $dictionary_end = "_" ]; then
dictionary_end=""
fi

dim=$8
neg_form=$9
epo1=${10}
epo0=${11}
epo3=${12}
parser=${13}
randmode=${14}
ragfiletrain=${15}
ragfiletest=${16}

if [ $parser = "_" ]; then
parser=""
fi

if [ $randmode = "-" ]; then
randmode=""
fi

rm prop_noun_set.txt
touch prop_noun_set.txt

y1="0"

for x1 in 5;
  do
  for w in "v48-48-32";
  #for w in "16";
    do

output_fold0="none"
output_fold0_test="none"
input_fold0=$output_fold0
input_fold0_test=$output_fold0_test

#for fold0 in "_w";
for fold0 in "_wG.fky!ztxujdpb";
      do
      fold=$fold0"-"$train_file"_"

      z=$ppn5
      for def1 in "word_only_word_src_lit_only";
        do
        def1=$def6
        for str in "parse";
          do

          for xn in "_xp4ns2igo2sk4b";
          #for xn in "_x0";
            do

output_io="none"
output_io_test="none"

            for io in "_irosf";
              do
input_io=$output_io
input_io_test=$output_io_test

              for xav in "_xav";
                do

                for load in "noload_savemodel_pretrain"$randmode;
                  do

yn="_y1"
x0n=$xn
if [ $x0n = "_xo" ]; then
x0n="_xn"
elif [ $x0n = "_xq" ]; then
x0n="_xp"
fi

x=$x1
y=$y1
def=$def1
def0=$def1
epochs=$epo1
epochlabel=$epo1
prevlabel=$epo3
epoch3s=$epo3

w2v="_t2v"
w2v0="_t2v"

if [ $load = "loadword_savemodel_pretrain"$randmode ]
then
w2v="_w2v"
fi

model_pre=$str"_"$def"_"$z"_"$form$xn$yn$io"_"$dim"_"$fold0"-"$train_file"_#"$epochlabel"#_model_pretrain"$w2v$xav".pickle"
loadmode=$load
load0="noload_savemodel_pretrain"$randmode

echo "\n\n"$str"_"$def" "$z" "$x" "$y" pretrain"
cmd1="python3 parse2vec"$parser".py text_vectors_"$train_file".json "$str"_"$def0"_"$z"_"$form$xn$yn$io"_"$loadmode$w2v$xav" none "$model_pre" none none none "$dim" "$fold" none none "$w" "$epochs
echo $cmd1
eval $cmd1

loadmode="loadmodel_savemodel_tuning"
loadmode0="loadmodel_savemodel_tuning"

for epoch in "1";
  do
  for loadmode2 in "zero_shotw";
    do

loadmode=$loadmode0
loadmode1=$loadmode2

loadmode="loadmodel_nosave_testset"
if [ "$randmode" = "randmode" ]
then
loadmode="loadmodel_nosave_testset1"
fi

echo "\n\n"$str"_word "$z" "$x" "$y" "$loadmode" test "$form$xn$yn" "$loadmode1
cmd1="python3 parse2vec"$parser".py text_vectors_"$test_file".json "$str"_"$def"_"$z"_"$form$xn$yn$io"_"$loadmode"_"$loadmode1$w2v$xav" "$model_pre" none none none none "$dim" "$fold" none none "$w" "$epoch
echo $cmd1
eval $cmd1


done
done

done
done
done
done
done
done
done
done
done




