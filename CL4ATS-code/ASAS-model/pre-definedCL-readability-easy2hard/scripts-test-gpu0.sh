





CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/cl_train_1.xlsx  --valid_file data/cl_valid_1.xlsx --test_file data/cl_test_1.xlsx   --test_result_file excel/lr00002_sub_1_test-a.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --best_model bert_lr00002_sub_1.bert  --num_cls 4  > result/lr00002_sub_1_test-a.txt


CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/cl_train_1.xlsx  --valid_file data/cl_valid_1.xlsx --test_file data/cl_test_1.xlsx   --test_result_file excel/lr00002_sub_1_test-b.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --best_model bert_lr00002_sub_1.bert  --num_cls 4  > result/lr00002_sub_1_test-b.txt


CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/cl_train_1.xlsx  --valid_file data/cl_valid_1.xlsx --test_file data/cl_test_1.xlsx   --test_result_file excel/lr00002_sub_1_test-c.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --best_model bert_lr00002_sub_1.bert  --num_cls 4  > result/lr00002_sub_1_test-c.txt


CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/cl_train_1.xlsx  --valid_file data/cl_valid_1.xlsx --test_file data/cl_test_1.xlsx   --test_result_file excel/lr00002_sub_1_test-d.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --best_model bert_lr00002_sub_1.bert  --num_cls 4  > result/lr00002_sub_1_test-d.txt


CUDA_VISIBLE_DEVICES=0 python -u code.py  --train_file data/cl_train_1.xlsx  --valid_file data/cl_valid_1.xlsx --test_file data/cl_test_1.xlsx   --test_result_file excel/lr00002_sub_1_test-e.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00002  --best_model bert_lr00002_sub_1.bert  --num_cls 4  > result/lr00002_sub_1_test-e.txt










