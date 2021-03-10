
# Experiment runner script

#EN='02'  #experiment_no

#python main.py \
#        --experiment_no='wcm_bertAug0.2'  \
#        --epochs=100 \
#        --epoch_step=90 \
#        --device_ids=0 \
#        --batch-size=8 \
#        --G-lr=0.001 \
#        --D-lr=0.1 \
#        --B-lr=0.001 \
#        --save_model_path='./checkpoint' \
#        --data_type='TrainTest_agNews' \
#        --data_path='../datasets/textAugment' \
#        --use_previousData=0 \
#        --model_type='MABert' \
#        --method='MultiLabelMAP' \
#        --overlength_handle='truncation' \
#        --min_tagFrequence=250  \
#        --max_tagFrequence=999999  \
#        --intanceNum_limit=999999 \
#        --data_split=999999  \
#        --test_description='' \
#        --resume=''   #  \
#        --evaluate=False


python main.py \
        --experiment_no='zyz_alldata_semiGAN_MSELoss'  \
        --epochs=400 \
        --epoch_step=380 \
        --device_ids=0 \
        --batch-size=4 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTest_programWeb_freecode_AAPD' \
        --data_path='../datasets/ProgrammerWeb' \
        --use_previousData=0 \
        --model_type='MABert' \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='truncation' \
        --min_tagFrequence=0  \
        --max_tagFrequence=999999  \
        --intanceNum_limit=999999 \
        --data_split=999999  \
        --test_description='' \
        --resume=''  \


#方法、epoch_step
# MLPBert, MABert
#batch-size：1，4，8，16
#data_type: All  TrainTest  TrainTestTextTag
#method: MultiLabelMAP semiGAN_MultiLabelMAP
#overlength_handle: truncation  skip

#苏州服务器上数据：TrainTest_programWeb_freecode_AAPD
#../datasets/AAPD
#../datasets/Freecode
#../datasets/TREC-IS

#../datasets/EUR-Lex  (TrainTestTextTag)
#../datasets/RCV2  (TrainTestTextTag)
#../datasets/stack-overflow2000 (TrainTestTextTag)

#../datasets/ag-news

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


