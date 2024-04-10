from transformers import BertPreTrainedModel, BertConfig, BertModel
from torch import nn
import torch


class BERT_Model_Phrase_Extraction(BertPreTrainedModel):
    def __init__(self, df_manager, load=None, pos_weight=None, freeze=False, model_card = 'bert-base-uncased'):
        self.config = BertConfig.from_pretrained(model_card, output_attentions=True, output_hidden_states=True)
        self.freeze = freeze

        super().__init__(self.config)
        self.core = self.initialize_model(load)
        # Freeze BERT embedding layer parameters
        if freeze:
            for param in self.core.embeddings.parameters():
                param.requires_grad = False

        if pos_weight == None:
            self.pos_weight = torch.ones([self.config.num_labels]).to("cuda")
        else:
            self.pos_weight = pos_weight
        self.emotion_head = nn.Linear(self.config.hidden_size, len(df_manager.unique_emotions))
        self.trigger_head = nn.Linear(self.config.hidden_size, 1)
        self.post_init()

    def initialize_model(self, load):
        if load == None:
            return BertModel(self.config)
        else:
            print("load = ", load)
            return BertModel.from_pretrained(load, config=load+'/config.json', local_files_only=True)
        
    def search_for_chanels_in_output(input, index, output=None):
        start_chanel=None
        end_chanel=None
        for i, token in enumerate(input):
            if token == 102:
                index = index-1
            if index == 0:
                start_chanel = i
        for i, token in enumerate(input[start_chanel:]):
            if token == 102:
                end_chanel = i
        print(input[start_chanel:end_chanel])
        #return output[start_chanel:end_chanel]
            
    def forward(
        self,
        utterance_index=None,
        dialogue_ids=None,
        dialogue_mask=None,
        token_type_ids=None,
        emotions_id=None,
        triggers=None,
        return_dict=None,
    ):
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ##self.search_for_chanels_in_output(input[0], utterance_index[i], outputs[i])
        ##outputs = self.core(
        ##    input_ids=dialogue_ids,
        ##    attention_mask=dialogue_mask,
        ##    token_type_ids=token_type_ids,
        ##    return_dict=return_dict,
        ##) 
##
        #model_output = [self.search_for_chanels_in_output(input[i], utterance_index[i], outputs[i]) for i in range(len(dialogue_ids))]
        #emotion_logits = torch.mean(self.emotion_head(model_output), dim=(1))
        #trigger_logits = torch.mean(self.trigger_head(model_output), dim=(1))
        #return {"emotion_logits": emotion_logits,
        #        "trigger_logits": trigger_logits}