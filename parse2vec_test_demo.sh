#!/bin/sh

dim=128
epoch1=8
epoch2=8
epoch3=8

ppn1="p0_id0"
script1="./parse2vec_subtest_demo.sh"
parser="_demo_v1_54"

type1="_spacyac"
type2="d_spacyc"
type3="_spacyac"
typeRN="_n20"
typeRT="_ag_news"
#typeTrain="_ag_news"
typeTrain="_n20"
typeTest="_ag_news"
#typeTest="_n20"
form="transformer10fqx_early_fusion"
neg_form="negsmpl0sqx_early_fusion"

fine_tune="train8"$type1$typeTest
#test1="test8"$type1$typeTest
#test1="test7"$type3$typeTest
test1="test1"$type3$typeTest
ragXn="train2"
ragXt="train2"

$script1 train1$type1$typeTrain $fine_tune $test1 $form $ppn1 word_only_word_src_lit_only $type2 $dim $neg_form $epoch1 $epoch1 $epoch3 $parser - $ragXn$type1$typeRN $ragXt$type1$typeRT

#$script1 train2$type1$typeTrain $fine_tune $test1 $form $ppn1 word_only_word_src_lit_only $type2 $dim $neg_form $epoch1 $epoch1 $epoch3 $parser - $ragXn$type1$typeRN $ragXt$type1$typeRT

#$script1 train4$type1$typeTrain $fine_tune $test1 $form $ppn1 word_only_word_src_lit_only $type2 $dim $neg_form $epoch1 $epoch1 $epoch3 $parser - $ragXn$type1$typeRN $ragXt$type1$typeRT

#$script1 train8$type1$typeTrain $fine_tune $test1 $form $ppn1 word_only_word_src_lit_only $type2 $dim $neg_form $epoch1 $epoch1 $epoch3 $parser - $ragXn$type1$typeRN $ragXt$type1$typeRT

#$script1 train16$type1$typeTrain $fine_tune $test1 $form $ppn1 word_only_word_src_lit_only $type2 $dim $neg_form $epoch1 $epoch1 $epoch3 $parser - $ragXn$type1$typeRN $ragXt$type1$typeRT

#$script1 train32$type1$typeTrain $fine_tune $test1 $form $ppn1 word_only_word_src_lit_only $type2 $dim $neg_form $epoch1 $epoch1 $epoch3 $parser - $ragXn$type1$typeRN $ragXt$type1$typeRT

#$script1 train64$type1$typeTrain $fine_tune $test1 $form $ppn1 word_only_word_src_lit_only $type2 $dim $neg_form $epoch1 $epoch1 $epoch3 $parser - $ragXn$type1$typeRN $ragXt$type1$typeRT

