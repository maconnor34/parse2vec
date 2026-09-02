#python3 spacy_sub_demo5.py Root_Words.csv Suffix_Dictionary.csv Suffix_Expansion.csv Word_Gender.csv ../Corpora 3 diff

import spacy
import itertools as it, glob
import os
import sys
import json
import subprocess
import copy

puncts = {".":"PERIOD","..":"PERIOD_PERIOD","...":"PERIOD_PERIOD_PERIOD","?":"QUESTION_MARK","!":"EXCLAMATION_MARK",",":"COMMA",";":"SEMI_COLON",":":"COLON","(":"OPEN_PARENTHESIS",")":"CLOSE_PARENTHESIS","[":"OPEN_BRACKET","]":"CLOSE_BRACKET","{":"OPEN_BRACE","}":"CLOSE_BRACE","-":"DASH","--":"DASH_DASH","\'":"SINGLE_QUOTE","\"":"DOUBLE_QUOTE"}

hypoth0 = {"zxwvbcdf":"0"}

hypoth1 = {"zxwvbcdf":"0"}

hypoth = {"zxwvbcdf":"0"}

subst_word = "to"
subst_word_UNK = "zxwvbcdf"

num_unk = 11
num_unk_max = 11
num_kno = 9

def multiple_file_types(*patterns):
    files1 = sorted(it.chain.from_iterable(glob.glob(pattern) for pattern in patterns))
    directory_path = '.' # Use '.' for the current directory or specify your path
    files2 = [os.path.join(directory_path, f) for f in files1]
    files3 = [f for f in files2 if os.path.isfile(f)]
    files3.sort(key=os.path.getsize)
    return files3

def get_corefs(spans,sent_i):
    corefs = {}
    if "coref_clusters_"+str(sent_i) in spans:
        corefH = ""
        for c in spans["coref_clusters_"+str(sent_i)]:
            c_str = str(c)
            if len(c_str) > 3 and c_str[-2:] == "\'s": c_str = c_str[:-2]
            while len(c_str) > 2 and not c_str[-1].isalpha(): c_str = c_str[:-1]
            c_lower = c_str.lower()
            if c_lower.isalpha():
                if c_lower in ["he","his","him","himself","she","her","hers","herself","it","its","itself","you","your","yours","yourself","yourselves","we","our","ours","ourselves","us","they","their","theirs","them","themselves","i","me","mine","my","myself","this","that","these","those"]:
                    if corefH != "":
                        if c_lower not in corefs:
                            corefs[c_lower] = []
                        corefs[c_lower].append(corefH)
                else:
                    case1 = c_str.split(" ")[-1]
                    case2 = c_lower.split(" ")[-1]
                    if case1 !=  case2:
                        corefH = case2
    return corefs

def get_objects_line(doc,doc2,doc_num=0,primary_doc=True,mode="orig"):
    last_token = None
    last_punct = None
    objects_line = []
    ii = doc_num * 1000
    last_word = ""
    sent_i = 1
    corefs = None
    if primary_doc and doc2 is not None:
        corefs = get_corefs(doc2.spans,sent_i)

    for token_i in range(len(doc)):
        token = doc[token_i]
        tmp = {}
        if primary_doc:
            if token.text in [".","?","!"]:
                sent_i += 1
                if doc2 is not None:
                    corefs = get_corefs(doc2.spans,sent_i)
        str1 = token.text.lower().strip()
        if primary_doc:
            if str1 in puncts:
                punct = puncts[str1]
                if last_token is not None:
                    last_token["next_punct_"+punct] = "metadata"
                last_punct = punct
                continue
        if str1 == "." and last_word == "label": continue
        last_word = str1
        has_num = False
        for n in "0123456789":
            has_num = has_num or n in str1
        if has_num:
            try:
                num1 = int(str1)
                if num1 > 100 or num1 < 0:
                    str1 = "NUM"
            except:
                str1 = "NUM"
                has_num = False
        elif str1 == "." or str1 == "," or "\'" in str1:
            continue
        elif str1.isalpha():
            pass
        else:
            continue

        tmp["ID"] = str(ii)
        ii += 1
        tmp["STR"] = str1
        tmp["IDX"] = token.idx
        if primary_doc and last_punct is not None:
            tmp["last_punct_"+last_punct] = "metadata"
        last_punct = None
        tmp[token.tag_] = "pos2"
        tmp[token.pos_] = "pos"
        if token.dep_ != "":
            tmp[token.dep_] = "metadata"
        tmp["_dep_"] = token.dep_
        tmp["_head_"] = token.head.text
        tmp["_text_"] = token.text

        if primary_doc:
            if str1 == "NUM":
                tmp["lemma_NUM"] = "metadata"
            else:
                tmp["lemma_"+token.lemma_.lower()] = "metadata"
            str_morph = str(token.morph)
            if str_morph != "" and mode != "rag":
                fields = str_morph.split("|")
                for f in fields:
                    tmp["morph_"+f] = "metadata"
        if token.ent_type_ != "" and mode != "rag":
            tmp["ent_type_"+token.ent_type_] = "metadata"

        if str1 in bases:
            tmp[bases[str1]] = "word"
        else:
            tmp[str1] = "word"
        if primary_doc:
            if str1 in suffixes:
                suffix1 = suffixes[str1]
                if suffix1 in expansions:
                    for s in expansions[suffix1]:
                        tmp[s] = "sfx"

        if primary_doc:
            if str1 in genders and genders[str1] != "":
                tmp[genders[str1]] = "metadata"

            if doc2 is not None:
                corefH = ""
                if str1 in corefs:
                    heads = corefs[str1]
                    for i in range(len(heads)):
                        if heads[i] != "":
                            corefH = heads[i]
                            heads[i] = ""
                            break
                    corefs[str1] = heads
                if corefH != "":
                    tmp["Coreference"] = corefH

        objects_line.append(tmp)
        last_token = tmp

    for i in range(len(objects_line)):
        id1 = objects_line[i]["ID"]
        head = objects_line[i]["_head_"]
        dep = objects_line[i]["_dep_"]
        text = objects_line[i]["_text_"]
        if head != text:
            for j in range(1,10):
                if i-j < 0: continue
                id2 = objects_line[i-j]["ID"]
                if head == objects_line[i-j]["_text_"]:
                    objects_line[i-j]["h"+dep] = "metadata-"+str(id1)
                    objects_line[i][dep] = "metadata-"+str(id2)
                    break
                if i + j >= len(objects_line): continue
                id3 = objects_line[i+j]["ID"]
                if head == objects_line[i+j]["_text_"]:
                    objects_line[i+j]["h"+dep] = "metadata-"+str(id1)
                    objects_line[i][dep] = "metadata-"+str(id3)
                    break
    for i in range(len(objects_line)):
        if "_head_" in objects_line[i]:
            del objects_line[i]["_head_"]
        if "_dep_" in objects_line[i]:
            del objects_line[i]["_dep_"]
        if "_text_" in objects_line[i]:
            del objects_line[i]["_text_"]

    return objects_line

def is_all_z(feature):
    return subst_word in feature.lower()

    if "zz" in feature or "Zz" in feature:
        return True
    maxZ = len(feature)
    if maxZ > 2:
        maxZ = 2
    for m in range(maxZ):
        if feature[m] not in ["z","Z"]:
            return False
    return True

def is_all_z2(feature2,feature):
    return subst_word in feature2.lower()

    if "zz" in feature2 or "Zz" in feature2:
        return True
    if is_all_z(feature2):
        return True
    maxZ = len(feature2)
    for m in range(maxZ):
        if m >= len(feature):
            if feature2[m] in ["z","Z"]:
                return True
        elif feature[m] not in ["z","Z"] and feature2[m] in ["z","Z"]:
            return True
    return False


def get_diff(line,line2):
    additions = {}
    missing = {}

    for l in line:
        if "lemma_" in l or "_Case" in l or "_punct_" in l or "Concept_" in l or "IDX" in l: continue
        if l not in line2:
            missing[l] = line[l]
    for l in line2:
        if is_all_z(l) or "IDX" in l:
            continue
        if l not in line:
            additions[l] = line2[l]

    return additions, missing

id_i = 1
bases = {}
suffixes = {}
expansions = {}
genders = {}
with open(sys.argv[1], "r") as fi:
  lines = fi.readlines()
  for l in lines:
    fields = l.strip().split(",")
    bases[fields[1]] = fields[0]
with open(sys.argv[2], "r") as fi:
  lines = fi.readlines()
  for l in lines:
    fields = l.strip().split(",")
    suffixes[fields[0]] = fields[1]
with open(sys.argv[3], "r") as fi:
  lines = fi.readlines()
  for l in lines:
    fields = l.strip().split(",")
    while len(fields) > 2 and fields[-1] == "": fields.pop()
    expansions[fields[0]] = fields[1:]
with open(sys.argv[4], "r") as fi:
  lines = fi.readlines()
  for l in lines:
    fields = l.strip().split(",")
    genders[fields[0]] = fields[1]

start_finish = sys.argv[7]
if start_finish == "-":
    file_start = 0
    file_end = int(sys.argv[6])
elif "-" in start_finish:
    fields = start_finish.split("-")
    file_start = int(fields[0])
    file_end = int(fields[1])

mode = sys.argv[8]

print(file_start,file_end,"start end")

nlp2 = nlpX = None
nlp = spacy.load("en_core_web_lg")  #w2v
if mode == "orig":
    nlp2 = spacy.load('en_coreference_web_trf')
    hypoth = {}
elif mode == "diff":
    nlpX = spacy.load("en_core_web_lg", exclude=["lemmatizer"])
elif mode == "rag":
    hypoth = {}
files=[]
i = 0
for f in multiple_file_types(sys.argv[5]+"/*segment.txt"):
  if i >= int(sys.argv[6]):
    pass
    break
  files.append(f)
  i+=1

subwords = {}

if mode == "orig":
    fo = open(sys.argv[9]+".tmp.csv", "w")
elif mode == "diff":
    fo1 = open(sys.argv[9]+".tmp1.csv", "w")
elif mode == "rag":
    fo = open(sys.argv[9]+".tmp2.csv", "w")

texts = []
objects = []
for f_index in range(len(files)):
  f_path = files[f_index]
  f_str = str(f_index)
  if f_index < file_start:
      print(f_path+"---------skip before start", f_index, file_start, file_end)
      continue
  if f_index > file_end:
      print(f_path+"---------skip after end", f_index, file_start, file_end)
      break
  print(f_path+"---------", f_index, file_start, file_end)

  fh=open(f_path,"r")
  lines=fh.readlines()
  fh.close()
  all_lines = "".join(lines)
  all_lines2 = lines
  texts = []
  texts.append(all_lines)
  objects_file = []
  for line_index in range(len(all_lines2)):
    line = all_lines2[line_index]
    objects_line = []
    doc = nlp(line)
    doc2 = None
    if mode == "orig":
        doc2 = nlp2(line)
        objects_line = get_objects_line(doc,doc2,line_index,True,mode)
    elif mode == "rag":
        objects_line = get_objects_line(doc,None,line_index,True,mode)
    else:
        objects_line = get_objects_line(doc,None,line_index,False,mode)

    len1 = len(doc)
    docX = []
    for subst_word in hypoth:
        subst_index = hypoth[subst_word]
        docUNK = []
        linesUNK = []
        for j in range(num_unk):
            lineX = copy.deepcopy(line)
            for k in range(len(objects_line)-1,-1,-1):
                objects_line[k]["IDX"+str(j)] = objects_line[k]["IDX"]
            for k in range(len(objects_line)-1,-1,-1):
                if k % num_unk == j:
                    if True:
                        s_word = subst_word
                        pre_space = False
                        #this handled         yesterday.An    becoming     yesterday.qwerasdfzxcv        which was 1 word
                        if lineX[objects_line[k]["IDX"]] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                            s_word = s_word[0].upper()+s_word[1:]
                        #this handles         haven't    becoming    qwerasdfzxcvn't     which was 1 word
                        if lineX[objects_line[k]["IDX"]].isalpha() and (objects_line[k]["IDX"] + len(objects_line[k]["STR"]) + 1) < len(lineX) and (lineX[objects_line[k]["IDX"] + len(objects_line[k]["STR"])].isalpha() or ( lineX[objects_line[k]["IDX"] + len(objects_line[k]["STR"])] == "\'" and lineX[objects_line[k]["IDX"] + len(objects_line[k]["STR"]) + 1].isalpha())):
                            s_word = s_word + " "
                        #this handles 12cm
                        elif k+1 < len(objects_line) and objects_line[k+1]["IDX"] == objects_line[k]["IDX"] + len(objects_line[k]["STR"]):
                            s_word = s_word + " "
                        if k-1 >= 0 and objects_line[k]["IDX"] == objects_line[k-1]["IDX"] + len(objects_line[k-1]["STR"]):
                            s_word = " " + s_word
                            pre_space = True
                        len_diff = len(s_word) - len(objects_line[k]["STR"])
                        idx_start = objects_line[k]["IDX"]
                        lineX = lineX[:objects_line[k]["IDX"]] + s_word + lineX[objects_line[k]["IDX"]+len(objects_line[k]["STR"]):]
                        for k1 in range(len(objects_line)):
                            if pre_space and objects_line[k1]["IDX"] == idx_start:
                                objects_line[k1]["IDX"+str(j)] += 1
                            elif objects_line[k1]["IDX"] > idx_start:
                                objects_line[k1]["IDX"+str(j)] += len_diff
            linesUNK.append(lineX)
            tmpX = nlpX(lineX)
            docUNK.append(tmpX)
        objects_lineUNK = []
        objects_maps = []
        for n in range(num_unk):
            idx2 = "IDX"+str(n)
            adjusted = False
            tmp = get_objects_line(docUNK[n], None,line_index, False,mode)
            tmp_copy = copy.deepcopy(tmp)

            for i in range(len(objects_line)):
                if i > len(objects_line) - 3:
                    continue
                if i >= len(tmp):
                    continue
                if objects_line[i][idx2] == tmp[i]["IDX"] and objects_line[i]["STR"] == tmp[i]["STR"]:
                    continue
                if i+1 < len(tmp) and objects_line[i][idx2] == tmp[i+1]["IDX"]:
                    if is_all_z(tmp[i]["STR"]):
                        del tmp[i+1]
                    else:
                        del tmp[i]
                    adjusted = True
                if i+2 < len(tmp) and objects_line[i][idx2] == tmp[i+2]["IDX"]:
                    if is_all_z(tmp[i]["STR"]):
                        del tmp[i+1]
                        del tmp[i+2]
                    elif is_all_z(tmp[i+1]["STR"]):
                        del tmp[i]
                        del tmp[i+2]
                    else:
                        del tmp[i]
                        del tmp[i+1]
                    adjusted = True
                elif i+1 < len(objects_file) and i < len(tmp) and objects_line[i+1][idx2] == tmp[i]["IDX"]:
                    if i % num_unk == n:
                        s_word = subst_word
                        if objects_line[i]["STR"][0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                            s_word = s_word[0].upper()+s_word[1:]
                        new_i = {"STR":s_word, "ID":"999", s_word:"word", "IDX":objects_line[i][idx2]}
                        tmp.insert(i,new_i)
                    else:
                        tmp.insert(i,objects_line[i])
                    adjusted = True
            for i in range(1,len(objects_line)+1):
                if i > len(objects_line) - 3:
                    continue
                if i >= len(tmp):
                    continue
                if objects_line[-i][idx2] == tmp[-i]["IDX"] and objects_line[-i]["STR"] == tmp[-i]["STR"]:
                    continue
                if i+1 < len(tmp) and objects_line[-i][idx2] == tmp[-i-1]["IDX"]:
                    if is_all_z(tmp[-i]["STR"]):
                        del tmp[-i-1]
                    else:
                        del tmp[-i]
                    adjusted = True
                elif i+2 < len(tmp) and objects_line[-i][idx2] == tmp[-i-2]["IDX"]:
                    if is_all_z(tmp[-i]["STR"]):
                        del tmp[-i-1]
                        del tmp[-i-2]
                    elif is_all_z(tmp[-i-1]["STR"]):
                        del tmp[-i]
                        del tmp[-i-2]
                    else:
                        del tmp[-i]
                        del tmp[-i]
                    adjusted = True
                elif i+1 < len(objects_line) and i < len(tmp) and objects_line[-i-1][idx2] == tmp[-i]["IDX"]:
                    if (len(objects_line) - i) % num_unk == n:
                        s_word = subst_word
                        if objects_line[-i]["STR"][0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                            s_word = s_word[0].upper()+s_word[1:]

                        new_i = {"STR":s_word, "ID":"999", s_word:"word", "IDX":objects_line[-i][idx2]}
                        tmp.insert(-i,new_i)
                    else:
                        tmp.insert(-i,objects_line[-i])
                    tmp.insert(-i,objects_line[-i])
                    adjusted = True
            if adjusted and False:
                words1 = []
                for k in range(len(objects_line)): words1.append(objects_line[k][idx2])
                words1 = []
                for k in range(len(objects_line)): words1.append(objects_line[k]["STR"])
                words1 = []
                for k in range(len(tmp)): words1.append(tmp[k]["IDX"])
                words1 = []
                for k in range(len(tmp)): words1.append(tmp[k]["STR"])
                words1 = []
                for k in range(len(tmp_copy)): words1.append(tmp_copy[k]["IDX"])
                words1 = []
                for k in range(len(tmp_copy)): words1.append(tmp_copy[k]["STR"])
                words1 = []
                for k in range(len(docUNKmap)): words1.append(docUNKmap[k])
            objects_lineUNK.append(tmp)
        for i in range(len(objects_line)):
            unk_i = i % num_unk
            found = False
            unk_0 = False
            for k in range(num_unk):
                if k == 4 and unk_0:
                    found = True
                    break
                unk_i2 = (unk_i + k) % num_unk
                idx2 = "IDX"+str(unk_i2)
                idx = objects_line[i][idx2]
                if i < len(objects_lineUNK[unk_i2]):
                    idxU = objects_lineUNK[unk_i2][i]["IDX"]
                    unk_0 = unk_0 or (idx == idxU and objects_lineUNK[unk_i2][i]["STR"] == "NUM" and k == 0)
                    if idx == idxU and is_all_z(objects_lineUNK[unk_i2][i]["STR"]):
                        unk_i = unk_i2
                        found = True
                        break
                unk_i2 = (unk_i + unk_i - k) % num_unk
                idx2 = "IDX"+str(unk_i2)
                idx = objects_line[i][idx2]
                if i < len(objects_lineUNK[unk_i2]):
                    idxU = objects_lineUNK[unk_i2][i]["IDX"]
                    if idx == idxU and is_all_z(objects_lineUNK[unk_i2][i]["STR"]):
                        unk_i = unk_i2
                        found = True
                        break
            words1 = []
            for k in range(i): words1.append(objects_line[k]["STR"])
            words1.append("|")
            words1.append(objects_line[i]["STR"])
            words1.append("|")
            for k in range(i+1,len(objects_line)): words1.append(objects_line[k]["STR"])

            words2 = []
            ii = i
            if ii > len(objects_lineUNK[unk_i]) - 1:
                ii = len(objects_lineUNK[unk_i]) - 1
            for k in range(ii): words2.append(objects_lineUNK[unk_i][k]["STR"])
            words2.append("|")
            words2.append("|")
            if i+1 < len(objects_lineUNK[unk_i]) - 1:
                for k in range(i+1,len(objects_lineUNK[unk_i])): words2.append(objects_lineUNK[unk_i][k]["STR"])
            unk_i_warn = False
            if not found and objects_line[i]["STR"].isalpha() and objects_line[i]["STR"] != "NUM":
                if False:
                    print("",i,unk_i,line,"\n",i,unk_i,linesUNK[unk_i],"----------------unk_i-\n")
                    print(i,words1,"not found 1")
                    print(i,words2,"not found UNK----------------------", unk_i)
                    print(objects_line[i])
                    if i < len(objects_lineUNK[unk_i]):
                        print(objects_lineUNK[unk_i][i])
                    unk_i_warn = True
            idx2 = "IDX"+str(unk_i)

            for j in range(int(num_unk / 2)):
                if i-j > 0 and i-j < len(objects_lineUNK[unk_i]):
                    obj_id_neg = str(objects_line[i-j]["ID"])
                    line1 = objects_line[i-j]
                    line1_z = is_all_z(line1["STR"])
                    line2 = objects_lineUNK[unk_i][i-j]
                    line2_z = is_all_z2(line2["STR"],line1["STR"])
                    additions, missing = get_diff(line1,line2)
                    for jj in additions:
                        if additions[jj] == "word" and not line2_z and not line1_z and jj.isalpha() and jj != "NUM":
                            pass
                        else:
                            if "Suffix" in jj or "morph_" in jj or additions[jj] == "word":
                                continue
                            if j != 0 or subst_index != "0":
                                if j != 0 or additions[jj] not in ["pos","pos2"]:
                                    fo1.write(f_str+","+str(line_index)+","+obj_id_neg+","+subst_index+"_+_"+str(-j)+"_"+jj+",mod\n")

                    for jj in missing:
                        if j != 0 and missing[jj] == "word" and not line2_z and not line1_z and jj.isalpha() and jj != "NUM":
                            pass
                        else:
                            if "Suffix" in jj or "morph_" in jj or missing[jj] == "word":
                                continue
                            if j != 0:
                                fo1.write(f_str+","+str(line_index)+","+obj_id_neg+","+subst_index+"_-_"+str(-j)+"_"+jj+",mod\n")
                    if j == 0 and len(missing) == 0 and "X" not in line1 and "punct" not in line1 and "NUM" not in line1 and not line1_z:
                        pass
                if j == 0: continue
                if i+j < len(objects_line) and i+j < len(objects_lineUNK[unk_i]):
                    obj_id_pos = str(objects_line[i+j]["ID"])
                    line1 = objects_line[i+j]
                    line1_z = is_all_z(line1["STR"])
                    line2 = objects_lineUNK[unk_i][i+j]
                    line2_z = is_all_z2(line2["STR"],line1["STR"])
                    additions, missing = get_diff(line1,line2)
                    for jj in additions:
                        if additions[jj] == "word" and not line2_z and not line1_z and jj.isalpha() and jj != "NUM":
                            pass
                        else:
                            if "Suffix" in jj or "morph_" in jj or additions[jj] == "word":
                                continue
                            fo1.write(f_str+","+str(line_index)+","+obj_id_pos+","+subst_index+"_+_"+str(j)+"_"+jj+",mod\n")

                    for jj in missing:
                        if j != 0 and missing[jj] == "word" and not line2_z and not line1_z and jj.isalpha() and jj != "NUM":
                            pass
                        else:
                            if "Suffix" in jj or "morph_" in jj or missing[jj] == "word":
                                continue
                            fo1.write(f_str+","+str(line_index)+","+obj_id_pos+","+subst_index+"_-_"+str(j)+"_"+jj+",mod\n")

    if mode == "orig" or mode == "rag":
        for i in range(len(objects_line)):
            objects_line[i]["line"] = str(line_index)
            objects_line[i]["file"] = f_str

        objects_file.extend(objects_line)

  if mode == "orig" or mode == "rag":
    for o in objects_file:
        o_lst = []
        for p in o.keys():
            if "," in p or p == "" or p == "IDX":
                continue
            o_lst.append(p+":"+str(o[p]))
        fo.write(",".join(o_lst)+"\n")

if mode == "orig" or mode == "rag":
    fo.close()
else:
    fo1.close()
