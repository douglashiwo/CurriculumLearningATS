





CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/essay_train_8.xlsx  --valid_file data/essay_valid_8.xlsx --test_file data/essay_test_8.xlsx --test_result_file excel/lr00002_sub_8_test-a.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --num_cls 1 --use_features 1 --features_num 24  > result/lr00002_sub_8_test-a.txt

sleep 3s

CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/essay_train_8.xlsx  --valid_file data/essay_valid_8.xlsx --test_file data/essay_test_8.xlsx --test_result_file excel/lr00002_sub_8_test-b.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --num_cls 1 --use_features 1 --features_num 24  > result/lr00002_sub_8_test-b.txt

sleep 3s

CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/essay_train_8.xlsx  --valid_file data/essay_valid_8.xlsx --test_file data/essay_test_8.xlsx --test_result_file excel/lr00002_sub_8_test-c.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --num_cls 1 --use_features 1 --features_num 24  > result/lr00002_sub_8_test-c.txt

sleep 3s

CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/essay_train_8.xlsx  --valid_file data/essay_valid_8.xlsx --test_file data/essay_test_8.xlsx --test_result_file excel/lr00002_sub_8_test-d.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --num_cls 1 --use_features 1 --features_num 24  > result/lr00002_sub_8_test-d.txt

sleep 3s

CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/essay_train_8.xlsx  --valid_file data/essay_valid_8.xlsx --test_file data/essay_test_8.xlsx --test_result_file excel/lr00002_sub_8_test-e.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --num_cls 1 --use_features 1 --features_num 24  > result/lr00002_sub_8_test-e.txt

sleep 3s







