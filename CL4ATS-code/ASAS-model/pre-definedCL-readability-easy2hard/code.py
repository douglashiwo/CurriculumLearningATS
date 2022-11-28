# Import BeautifulSoup
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import time, os, random, bs4
from transformers import AdamW
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import roc_curve, roc_auc_score, auc,precision_score, f1_score, accuracy_score, recall_score
from datasets import Dataset
import argparse
import ml_metrics as metrics
from torch.utils.data import WeightedRandomSampler



def re_weighting_sequence(target_series, current_epoch, total_epoch = 10 ):

    """
    Here we require target_series containes only two types of 
    element \in {0,1}, where 1 represents positive and 0 represents negative.
    total_epoch = 10
    current_epoch = 2
    target_series = [1,1,1,0,1,1,1,1,0,0]
    
    This strategy means first negative and then later positive.
    
    """

    #positive_pro = (current_epoch * 1.0 / total_epoch)
    #negative_pro = 1.0 - positive_pro
    #res_series = [ positive_pro if e != 1 else negative_pro for e in target_series ]
    
    easy_threshold = min(1.0, current_epoch * 1.0 / total_epoch) 
    
    res_series = [ 1 if e <= easy_threshold else 0 for e in target_series ]
    #res_series = [ 1 if 1 - e <= easy_threshold else 0 for e in target_series ]
    
    return res_series


    
    
def data_preparing(data_src, batch_size=1, shuffle = True, return_dataset = False):
    
    '''
    The key point here is to adapt your data.file(e.g. csv excel) 
    as to fit the input format of 'torch.utils.data.DataLoader'
    which will facilitate your later experiments
    '''
    
    #data_src = os.path.join('G:\\我的云端硬盘\\BERT-GRADER\\dataset\\sciEntsBank-two ways only', 'science_train.xlsx')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    #df = pd.read_excel('short-answer-grading.xlsx')
    df = pd.read_excel(data_src)
    #df['answertext'] = df['answertext'].astype(str) # donot know why but it just solved the problem.
    
    dataset = Dataset.from_pandas(df)
    
    
    tokenized_datasets = dataset.map(
                                lambda x: tokenizer(x["studentanswer"], padding="max_length", truncation=True)
                                 , batched = True
                                )
    tokenized_datasets = tokenized_datasets.map(
                                    lambda x : {'labels' : x['score1']}
                                    )


    '''
    tokenized_datasets = tokenized_datasets.map(
                                    lambda x : {'train_test' : 'train' if random.random()<0.8 else 'test'}
                                    )
    '''

    # keep the following cols only
    
    if return_dataset is True:
        return tokenized_datasets
    
    cols_to_remove = [col for col in tokenized_datasets.features.keys() if col not in ('attention_mask','input_ids','token_type_ids','labels')]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)

    tokenized_datasets.set_format("torch")

    dataloader = DataLoader(tokenized_datasets, shuffle=shuffle, batch_size = batch_size)
    #eval_dataloader = DataLoader(test_dataset, batch_size = batch_size)

    return dataloader   
    

def eval_model(model, eval_dataloader,num_cls=4,device = torch.device("cpu"), data_src = None, result_file = None):
    '''
    save and load model --
    #torch.save(model,'bert-cls.bert')   
    #new_model = torch.load(model_file) 
    '''
    #new_model = torch.load(model_file)   
    #metric= load_metric("accuracy") # datasets.list_metrics()  can show all metrics
    
    all_predictions = []
    all_labels = []
    result_logits = []
    
    t1 = time.time()
    
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        #result_logits.append(  )# logits.view(-1,4)[:,1].tolist()
        result_logits = result_logits +logits.view(-1,num_cls).tolist()
        #negative_logits = positive_logits +logits.view(-1,2)[:,0].tolist()
        predictions = torch.argmax(logits, dim=-1)
        all_predictions = all_predictions + predictions.view(-1).tolist()
        all_labels = all_labels + batch["labels"].view(-1).tolist()
    print('------------------------------------------------------------------------')
    print('pred and label info:')
    print('number of elements in pred:', len(all_predictions))
    #print('number of positive elements in pred:',len([i for i in all_predictions if int(i)==1 ]))
    print('number of elements in labels:', len(all_labels))
    #print('number of positive elements in labels:',len([i for i in all_labels if int(i)==1 ]))
    print('precision: ', round(precision_score(all_labels,all_predictions,average ='weighted' ),5) )
    print('recall: ', round(recall_score(all_labels,all_predictions,average ='weighted'),5))
    print('f1_score: ', round(f1_score(all_labels,all_predictions,average ='weighted'),5))
    print('accuracy: ', round(accuracy_score(all_labels,all_predictions),5))
    QWK = round(metrics.quadratic_weighted_kappa(all_labels,all_predictions),5)
    print('Quadratic_Weighted_Kappa: ', QWK )
    #print('****************check here about the shape and examples*******************')
    #print('shape of result_logits--',torch.Tensor(result_logits).shape)
    #print('samples from result_logits',result_logits[0:2])
    sfmx_prob = torch.nn.Softmax(dim = -1)(torch.Tensor(result_logits)).tolist() # -1 means the inmost dimension
    AUCval = round(roc_auc_score(all_labels, sfmx_prob, average = 'weighted', multi_class = 'ovo'),5)
    print('auc: ', AUCval)
    print('time cost for evaluation: {} secs'.format(int(time.time()-t1)))
    
    if data_src is not None and result_file is not None:        
    
        df = pd.read_excel(data_src)
        
        df['predictions'] = pd.Series(all_predictions)
        
        df.to_excel(result_file)
    
    return QWK


'''
---------------------------------------------------------------------------------------------
Have a look at model parameter's name THEN decide which to FREEZE.

for name, value in model.named_parameters():
    print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
'''
def freeze_model(model):

    freezing_para = [name for name, value in model.named_parameters()][:-2] # last two are CLS layer, need not freezing.
    for name, value in model.named_parameters():
        if name in freezing_para:
            value.requires_grad = False
        else:
            value.requires_grad = True

"---------------------------------------------------------------------------------------------"




def loader_return(tokenized_datasets, current_epoch, total_epoch):

    target_col = tokenized_datasets['readability_difficulty']

    cols_to_remove = [col for col in tokenized_datasets.features.keys() if col not in ('attention_mask','input_ids','token_type_ids','labels')]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove) 

    sampler = WeightedRandomSampler(weights = re_weighting_sequence(target_col, current_epoch = current_epoch, total_epoch = total_epoch), 
                                    num_samples = len(tokenized_datasets),
                                    replacement = True
                                   )
    tokenized_datasets.set_format("torch")
    dataloader = DataLoader(tokenized_datasets, batch_size = 1, sampler = sampler)
      
    return dataloader




def train_model(model,learning_rate, best_model,train_dataloader, valid_dataloader,test_dataloader,
                num_epochs=5, accumulation_steps=16, num_cls = 4,valid_data_src = None, 
                test_data_src = None, valid_result_file = None, test_result_file = None,
                device = torch.device("cpu")):
    
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    
    #train_dataloader_current = loader_return(train_dataloader, 1, num_epochs)

    num_training_steps = num_epochs * len(train_dataloader)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.to(device)
    #best_model = None  # we refer to AUC to deterimine the best model in VALID
    best_QWK_valid = None
    print('-+-+-+metrics before training+-+VALID+-+')
    eval_model(model = model, 
               eval_dataloader = valid_dataloader, num_cls = num_cls,
               device = device
                )
    print('-+-+-+metrics before training+-+TEST+-+')
    eval_model(model = model, 
               eval_dataloader = test_dataloader, num_cls = num_cls,
               device = device
                )            
    print('-+-+-+END+-+-+-+-+')
    
    total_training_num = len(train_dataloader)
    
    for w, epoch in enumerate(range(num_epochs)):
        #if epoch==3:
        #    freeze_model(model)
        t1 = time.time()
        cnt = 0 
        loss_avg = []       
        
        model.train() # simply changing the mode here
        
        train_dataloader_current = loader_return(train_dataloader, w + 1, num_epochs)
        counter = 0
        for batch in train_dataloader_current:
            if counter >= int(total_training_num * 1.0 * (w + 1) * 1.0 / num_epochs):
                break
            counter += 1
            #t1 = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss / accumulation_steps  
            loss_avg.append(loss.item())

            cnt+=1
            loss.backward()

            if cnt % accumulation_steps == 0:
                if cnt%(accumulation_steps*20)==0:
                    pass
                    #print('loss for this batch: ', np.sum(loss_avg))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                loss_avg = []
                
        print('time cost for this eopoch: {} secs'.format(int(time.time()-t1)))      
        if w+1==num_epochs:
            print('----------eval info after epoch_{}-----VALID--'.format(w+1))
            vQWK = eval_model(model = model, 
                   eval_dataloader = valid_dataloader, num_cls = num_cls,
                   device = device, data_src = valid_data_src, result_file = valid_result_file
                    )
                    
            print('----------eval info after epoch_{}-----TEST--'.format(w+1))  
            eval_model(model = model, 
                   eval_dataloader = test_dataloader,  num_cls = num_cls,
                   device = device, data_src = test_data_src, result_file = test_result_file
                    )
                    
            #torch.save(model,best_model)
            
        """
        if best_QWK_valid is None or best_QWK_valid < vQWK:
            best_QWK_valid = vQWK
            if os.path.exists(best_model):
                os.remove(best_model)
                print('---old model removed---')
            print('---current best QWK is {}---'.format(best_QWK_valid))
            torch.save(model,best_model)  
        
        """
    return None
    
  
if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        
    parser = argparse.ArgumentParser(description="This is a description")
    
    parser.add_argument('--accumulation_steps',dest='accumulation_steps',required = True,type = int)
    parser.add_argument('--num_epochs',dest='num_epochs',required = True,type = int)
    parser.add_argument('--batch_size',dest='batch_size',required = True,type = int)
    parser.add_argument('--num_cls',dest='num_cls',required = True,type = int)
    parser.add_argument('--learning_rate',dest='learning_rate',required = True,type = float)    
    parser.add_argument('--train_file',dest='train_file',required = True,type = str)
    parser.add_argument('--valid_file',dest='valid_file',required = True,type = str)
    parser.add_argument('--test_file',dest='test_file',required = True,type = str)
    parser.add_argument('--best_model',dest='best_model',required = True,type = str)
    parser.add_argument('--test_result_file',dest='test_result_file', required = False, type = str)
    parser.add_argument('--valid_result_file',dest='valid_result_file', required = False, type = str)
    
    args = parser.parse_args()    
    
    num_labels = args.num_cls
    best_model = args.best_model
    train_file = args.train_file
    valid_file = args.valid_file
    test_result_file = args.test_result_file
    valid_result_file = args.valid_result_file
    test_file = args.test_file
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate  
    batch_size = args.batch_size 
    accumulation_steps = args.accumulation_steps  # batch_size * accumulation_steps will be the actual batch_size  

    print('best_model: {}'.format(best_model))
    print('train_file: {}'.format(train_file)) 
    print('valid_file: {}'.format(valid_file))
    print('test_result_file: {}'.format(test_result_file))
    print('valid_result_file: {}'.format(valid_result_file))
    print('test_file: {}'.format(test_file))
    print('num_epochs: {}'.format(num_epochs))
    print('num_cls: {}'.format(num_labels))
    print('learning_rate: {}'.format(learning_rate))
    print('batch_size: {}'.format(batch_size))
    print('accumulation_steps: {}'.format(accumulation_steps))

    train = data_preparing(data_src = train_file, batch_size = batch_size, return_dataset = True) 
    valid = data_preparing(data_src = valid_file, batch_size = batch_size,shuffle = False) 
    test = data_preparing(data_src = test_file, batch_size = batch_size,shuffle = False)
    

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels = num_labels) 

    train_model(model = model,
                    best_model=best_model,
                    learning_rate = learning_rate, 
                    train_dataloader = train,
                    valid_dataloader = valid,
                    test_dataloader = test,
                    num_cls = num_labels,
                    num_epochs = num_epochs, 
                    accumulation_steps = accumulation_steps,
                    device = device,
                    test_data_src = test_file, 
                    valid_data_src = valid_file,                    
                    valid_result_file = valid_result_file , test_result_file = test_result_file
                ) 
    