#!/bin/bash

python3_cmd=python3.6

stage=0
use_gpu=cuda:0

model=bert
model_path=/home/M10815022/Models/bert-base-cased
save_path=./models/bert-devout-devin

train_datasets="HotpotQA_train NaturalQuestions_train NewsQA_train SearchQA_train SQuAD_train TriviaQA_train"
dev_datasets="BioASQ_dev DROP_dev DuoRC_dev RACE_dev RelationExtraction_dev TextbookQA_dev"
test_datasets="HotpotQA_test NaturalQuestions_test NewsQA_test SearchQA_test SQuAD_test TriviaQA_test"


if [ $stage -le 0 ]; then
  echo "====================================================="
  echo "     Convert data from MRQA-format to FGC-format     "
  echo "====================================================="
  mkdir -p dataset
  for split in dev test train; do
    echo "Converting $split set to FGC-format..."
    split_path=./MRQA-Shared-Task-2019/$split

    for fpath in $split_path/*; do
      fname=`cut -d'/' -f4 <<< $fpath`
      dataset=`cut -d'.' -f1 <<< $fname`
      output_path=dataset/${dataset}_${split}.json
      $python3_cmd scripts/convert_mrqa_to_fgc.py $fpath $output_path
    done
  done
  echo "Done."
fi


if [ $stage -le 1 ]; then
  echo "======================"
  echo "     Prepare data     "
  echo "======================"
  rm -rf data
  for split in train dev test; do
    for dir in passage passage_no_unk question question_no_unk answer span; do
      mkdir -p data/$split/$dir
    done
  done
  echo "Preparing dev set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path dev $dev_datasets || exit 1
  echo "Preparing test set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path test $test_datasets || exit 1
  echo "Preparing train set..."
  $python3_cmd scripts/prepare_${model}_data.py $model_path train $train_datasets || exit 1
fi


if [ $stage -le 2 ]; then
  echo "================================="
  echo "     Train and test QA model     "
  echo "================================="
  if [ -d $save_path ]; then
    echo "'$save_path' already exists! Please remove it and try again."; exit 1
  fi
  mkdir -p $save_path
  $python3_cmd scripts/train_${model}.py $use_gpu $model_path $save_path
fi
