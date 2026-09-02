#!/bin/sh

prog="spacy_sub_demo8.py"
vers="ad"
tmp="tmp"


corp="../Corpora_test_newsgroups/"
#./spacy_sub_run.sh $prog $corp 2 0-1 "text_vectors_test1_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 3 0-2 "text_vectors_test2_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 5 0-4 "text_vectors_test4_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 9 5-8 "text_vectors_test8_spacy"$vers"_n20.json"

#cp "text_vectors_test1_spacy"$vers"_n20.json."$tmp".csv" "text_vectors_test1_spacy"$vers"_n20.json."$tmp"a.csv"
#cp "text_vectors_test2_spacy"$vers"_n20.json."$tmp".csv" "text_vectors_test2_spacy"$vers"_n20.json."$tmp"a.csv"
#cp "text_vectors_test4_spacy"$vers"_n20.json."$tmp".csv" "text_vectors_test4_spacy"$vers"_n20.json."$tmp"a.csv"
#cp "text_vectors_test4_spacy"$vers"_n20.json."$tmp".csv" "text_vectors_test8_spacy"$vers"_n20.json."$tmp"a.csv"
#cat "text_vectors_test8_spacy"$vers"_n20.json."$tmp".csv" >> "text_vectors_test8_spacy"$vers"_n20.json."$tmp"a.csv"

corp="../Corpora_train_newsgroups/"
./spacy_sub_run.sh $prog $corp 2 0-1 "text_vectors_train1_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 3 0-2 "text_vectors_train2_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 5 0-4 "text_vectors_train4_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 9 0-8 "text_vectors_train8_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 17 0-16 "text_vectors_train16_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 33 17-32 "text_vectors_train32_spacy"$vers"_n20.json"
#./spacy_sub_run.sh $prog $corp 49 33-48 "text_vectors_train48_spacy"$vers"_n20.json" &
#./spacy_sub_run.sh $prog $corp 53 49-52 "text_vectors_train52_spacy"$vers"_n20.json" &

#cp "text_vectors_train4_spacy"$vers"_n20.json."$tmp".csv" "text_vectors_train8_spacy"$vers"_n20.json."$tmp"a.csv"
#cat "text_vectors_train8_spacy"$vers"_n20.json."$tmp".csv" >> "text_vectors_train8_spacy"$vers"_n20.json."$tmp"a.csv"
#cp "text_vectors_train8_spacy"$vers"_n20.json."$tmp"a.csv" "text_vectors_train16_spacy"$vers"_n20.json."$tmp"a.csv"
#cat "text_vectors_train16_spacy"$vers"_n20.json."$tmp".csv" >> "text_vectors_train16_spacy"$vers"_n20.json."$tmp"a.csv"
#cp "text_vectors_train16_spacy"$vers"_n20.json."$tmp"a.csv" "text_vectors_train32_spacy"$vers"_n20.json."$tmp"a.csv"
#cat "text_vectors_train32_spacy"$vers"_n20.json."$tmp".csv" >> "text_vectors_train32_spacy"$vers"_n20.json."$tmp"a.csv"

corp="../Corpora_test_ag/"
./spacy_sub_run.sh $prog $corp 1 0-0 "text_vectors_test1_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 2 0-1 "text_vectors_test2_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 4 0-3 "text_vectors_test4_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 7 0-6 "text_vectors_test7_spacy"$vers"_ag_news.json"

#cp "text_vectors_test1_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_test1_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_test2_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_test2_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_test4_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_test4_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_test4_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_test7_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_test7_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_test7_spacy"$vers"_ag_news.json."$tmp"a.csv"

corp="../Corpora_train_ag/"
#./spacy_sub_run.sh $prog $corp 2 0-1 "text_vectors_train1_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 3 0-2 "text_vectors_train2_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 5 0-4 "text_vectors_train4_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 9 5-8 "text_vectors_train8_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 17 9-16 "text_vectors_train16_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 33 17-25 "text_vectors_train32a_spacy"$vers"_ag_news.json" &
#./spacy_sub_run.sh $prog $corp 33 26-32 "text_vectors_train32b_spacy"$vers"_ag_news.json" &
#./spacy_sub_run.sh $prog $corp 65 32-36 "text_vectors_train64a_spacy"$vers"_ag_news.json" &
#./spacy_sub_run.sh $prog $corp 65 37-40 "text_vectors_train64b_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 65 41-41 "text_vectors_train64c_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 65 42-44 "text_vectors_train64d_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 65 45-48 "text_vectors_train64e_spacy"$vers"_ag_news.json" &
#./spacy_sub_run.sh $prog $corp 65 49-52 "text_vectors_train64f_spacy"$vers"_ag_news.json" &
#./spacy_sub_run.sh $prog $corp 65 53-56 "text_vectors_train64g_spacy"$vers"_ag_news.json" &
#./spacy_sub_run.sh $prog $corp 65 57-60 "text_vectors_train64h_spacy"$vers"_ag_news.json" &
#./spacy_sub_run.sh $prog $corp 65 61-65 "text_vectors_train64i_spacy"$vers"_ag_news.json"

#./spacy_sub_run.sh $prog $corp 0 0-0 "text_vectors_train32_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 0 0-0 "text_vectors_train64_spacy"$vers"_ag_news.json"
#./spacy_sub_run.sh $prog $corp 100 0-100 "text_vectors_train100_spacy"$vers"_ag_news.json"

#cp "text_vectors_train1_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_train1_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_train2_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_train2_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_train4_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_train4_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_train4_spacy"$vers"_ag_news.json."$tmp".csv" "text_vectors_train8_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train8_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train8_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_train8_spacy"$vers"_ag_news.json."$tmp"a.csv" "text_vectors_train16_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train8_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train16_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_train16_spacy"$vers"_ag_news.json."$tmp"a.csv" "text_vectors_train32_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train32a_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train32_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train32b_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train32_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cp "text_vectors_train32_spacy"$vers"_ag_news.json."$tmp"a.csv" "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64a_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64b_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64c_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64d_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64e_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64f_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64g_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64h_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"
#cat "text_vectors_train64i_spacy"$vers"_ag_news.json."$tmp".csv" >> "text_vectors_train64_spacy"$vers"_ag_news.json."$tmp"a.csv"

corp="../Corpora_bblm_gutenberg/"
#./spacy_sub_run.sh $prog $corp 2 0-1 "text_vectors_train1_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 3 0-2 "text_vectors_train2_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 5 0-4 "text_vectors_train4_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 9 0-8 "text_vectors_train8_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 17 9-16 "text_vectors_train16_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 33 17-32 "text_vectors_train32_spacy"$vers"_bbllm_gutenberg.json"

#./spacy_sub_run.sh $prog $corp 34 33-33 "text_vectors_test1_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 36 34-35 "text_vectors_test2_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 40 36-39 "text_vectors_test4_spacy"$vers"_bbllm_gutenberg.json"
#./spacy_sub_run.sh $prog $corp 48 40-47 "text_vectors_test8_spacy"$vers"_bbllm_gutenberg.json"

