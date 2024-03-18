source=test
target=assert
lr=2e-5
batch_size=128
beam_size=10
source_length=512
target_length=128
output_dir=saved_models/$source-$target/
# test_file=data/test.6.$source.txt,data/test.6.$target.txt
test_file=data/test_best-bleu.src,data/test_best-bleu.gold
pretrained_model=microsoft/graphcodebert-base
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin

CUDA_VISIBLE_DEVICES=1 python run.py \
--do_test \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--load_model_path $load_model_path \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log