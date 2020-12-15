CUDA_VISIBLE_DEVICES=3 python rec_train.py \
--data config/data/Rec_data_Synthetic.yml \
--cfg config/models/Rec_MobileNetV3_LSTM_CTC.yml \
--batch-size 4 --adam
