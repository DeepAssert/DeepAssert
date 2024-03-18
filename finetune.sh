source=test
target=assert
lr=3e-5
batch_size=12
beam_size=10
source_length=512
target_length=128
output_dir=saved_models/$source-$target/
train_file=data/train.$source.txt,data/train.$target.txt
dev_file=data/eval.$source.txt,data/eval.$target.txt
epochs=80
pretrained_model=microsoft/graphcodebert-base

mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=0 python -W ignore run.py \
--do_train \
--do_eval \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs 2>&1| tee $output_dir/train.log
