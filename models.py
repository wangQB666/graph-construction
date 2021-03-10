from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Variable
import torch.nn.functional as F


class MABert(nn.Module):
    def __init__(self, bert, num_classes, bert_trainable=True, device=0):
        super(MABert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes

        self.class_weight = Parameter(torch.Tensor(num_classes, 768).uniform_(0, 1), requires_grad=False).cuda(device)
        self.class_weight.requires_grad = True

        self.discriminator = Parameter(torch.Tensor(1, 768).uniform_(0, 1), requires_grad=False).cuda(device)
        self.discriminator.requires_grad = True

        self.Linear1 = nn.Linear(768, 500)
        self.Linear2 = nn.Linear(500, 1)
        self.act = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.output = nn.Softmax(dim=-1)

    # def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):
    
    #     token_feat = self.bert(ids,
    #                            token_type_ids=token_type_ids,
    #                            attention_mask=attention_mask)[0] #N, L, hidden_size
    
    #     # sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
    #     #                 / torch.sum(attention_mask, dim=1, keepdim=True)#N, hidden_size
    


    #     embed = self.bert.get_input_embeddings()
    #     tag_embedding = embed(encoded_tag)
    #     tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
    #                     / torch.sum(tag_mask, dim=1, keepdim=True)  #labels_num, hidden_size
    
    #     # tag_embedding = torch.cat((tag_embedding, feat), 0)
    
    #     masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool()
    #     attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((1-masks.byte()), torch.tensor(-np.inf))
    
    #     # similarity = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(0))
    #     # similarity = torch.sum(similarity, -1)#.unsqueeze(-1)
    #     # similarity = torch.sum(similarity, -1)
    
    #     attention = F.softmax(attention, -1)
    
    #     attention_out = torch.matmul(attention, token_feat)   # N, labels_num, hidden_size
    
    #     attention_out = attention_out * self.class_weight
    #     attention_out = torch.sum(attention_out, -1)
    
    #     # attention_out = self.Linear1(attention_out)
    #     # attention_out = self.act(attention_out)
    #     # attention_out = self.Linear2(attention_out).squeeze(-1)
    
    #     logit = torch.sigmoid(attention_out)
    
    #     # discrimate_hidden = torch.sum(
    #     #     torch.matmul(hidden_out[:, -1].squeeze(-2), self.class_weight.transpose(0, 1)), -1, keepdim=True)
    #     #
    #     # attention_out = torch.cat((attention_out, discrimate_hidden), -1)
    
    #     #################fake sample process#######
    
    #     # print(torch.sum(attention,-2))
    
    #     # fake_ids = ids.clone()#.detach() torch.Tensor(fake_ids.shape[0], fake_ids.shape[1]).uniform_(150, 1000).long().cuda(0)
    #     # # print(torch.sum(torch.sum(attention,-2) > 1, -1))
    #     # fake_ids = torch.where(torch.sum(attention,-2) > 0.2, torch.Tensor(fake_ids.shape[0], fake_ids.shape[1]).uniform_(150, 30000).long().cuda(0), fake_ids)
    #     #
    #     # # fake_ids[fake_ids > 102] -=
    #     #
    #     # feat = self.bert(fake_ids,
    #     #                        token_type_ids=token_type_ids,
    #     #                        attention_mask=attention_mask)[0]#.detach()
    
    #     feat = feat[:,:token_feat.shape[1],:] # N, L, hidden_size

    #     # fl_func = nn.MSELoss()
    #     # fl = fl_func(feat,token_feat)
    #     # feat += token_feat.detach()
    #     # feat = torch.mean(feat, 1)
    #     attention_fake = (torch.matmul(feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((1-masks.byte()), torch.tensor(-np.inf))
    
    #     # similarity_fake = (torch.matmul(feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(0))
    #     # similarity_fake = torch.sum(similarity_fake, -1)#.unsqueeze(-1)
    #     # similarity_fake = torch.sum(similarity_fake, -1)
    
    #     attention_fake = F.softmax(attention_fake, -1)
    
    #     # print('+',torch.max(attention_fake,-1)[0])
    
    #     attention_out_fake = torch.matmul(attention_fake, feat)  # N, labels_num, hidden_size
    #     # discrimate = torch.matmul(feat, tag_embedding.transpose(0, 1))
    
    #     attention_out_fake = attention_out_fake * self.class_weight
    #     attention_out_fake = torch.sum(attention_out_fake, -1)
    #     # print(attention_out_fake)
    #     # attention_out_fake = torch.sum(attention_out_fake, -1, keepdim=True)
    
    #     # discrimate_hidden = torch.sum(torch.matmul(hidden_out_fake[:,-1].squeeze(-2), self.class_weight.transpose(0, 1)),-1, keepdim=True)
    
    #     # attention_out_fake = torch.cat((attention_out_fake, discrimate_hidden), -1)
    
    #     # discrimate = torch.mean(attention_out_fake, -2, keepdim=True)
    #     # discrimate = torch.sum(discrimate, -1, keepdim=True)
    #     #################
    #     # pred = torch.cat((discrimate, logit), -1)
    
    #     # discrimate = torch.sum(torch.matmul(feat, self.class_weight.transpose(0, 1)), -1, keepdim=True)
    #     # attention_out = attention_out * self.class_weight
    
    #     # pred = torch.cat((attention_out_fake, attention_out), -2)
    #     # logit = self.Linear1(logit)
    #     # logit = self.act(logit)
    #     # logit = self.Linear2(logit).squeeze(-1)
    #     # logit = torch.sigmoid(logit)
    
    #     # logit = pred[:,self.num_classes:]
    
    #     # prob =  torch.sigmoid(attention_out_fake)
    
    #     # print('fake',torch.max(attention_out_fake, -1)[0])
    #     # print(torch.max(attention_out, -1)[0])
    #     # print(torch.sum(attention_out >= 0.5, -1))
    
    #     # print(torch.sum(attention_out_fake - attention_out > 0, -1))
    
    #     attention_out = torch.sum(attention_out, -1, keepdim=True)
    #     attention_out_fake = torch.sum(attention_out_fake, -1, keepdim=True)
    
    #     prob = torch.cat((attention_out_fake,attention_out),-1)
    #     # print(prob)
    #     prob = self.output(prob)
    
    #     # prob = torch.mean(prob,-1)
    
    #     #################
    #     # masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool()
    #     # attention = (torch.matmul(token_feat, self.discriminator.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(-np.inf))
    #     # attention = F.softmax(attention, -1)
    #     # attention_out = attention @ token_feat  # N, 1, hidden_size
    #     #
    #     # feat = feat[:, :token_feat.shape[1], :]  # N, L, hidden_size
    #     # attention_fake = (torch.matmul(feat, self.discriminator.transpose(0, 1))).transpose(1, 2).masked_fill(
    #     #     (1 - masks.byte()), torch.tensor(-np.inf))
    #     # attention_fake = F.softmax(attention_fake, -1)
    #     # attention_out_fake = attention_fake @ feat  # N, 1, hidden_size
    #     #
    #     # flatten = torch.cat((attention_out, attention_out_fake),-2)
    #     # # flatten = flatten * self.class_weight
    #     # # flatten = torch.sum(flatten, -1)
    #     #
    #     # flatten = self.Linear1(flatten)
    #     # flatten = self.act(flatten)
    #     # flatten = self.Linear2(flatten).squeeze(-1)
    #     # flatten = torch.sigmoid(flatten)
    
    
    #     #################
    
    #     # flatten = token_feat
    #     # prob = torch.cat((similarity_fake, similarity), -1)
    #     #
    #     # prob = self.output(prob)
    
    
    #     # prob = torch.max(pred[:,:self.num_classes],-1)[0] - 0.5
    #     # prob = self.relu(prob)
    
    #     return prob[:,1], logit, prob[:,0], attention

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0] #N, L, hidden_size

        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(attention_mask, dim=1, keepdim=True)#N, hidden_size

        embed = self.bert.get_input_embeddings()
        tag_embedding = embed(encoded_tag)
        tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(tag_mask, dim=1, keepdim=True)
        # print("similarity {}".format(torch.mean(torch.mean(tag_embedding, -1))))

        masks = torch.unsqueeze(attention_mask, 1)  # N, 1, L  .bool() .byte()
        attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((1-masks.byte()), torch.tensor(-np.inf))
        # attention = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill((~masks.bool()), torch.tensor(-np.inf))


        # similarity = (torch.matmul(token_feat, tag_embedding.transpose(0, 1))).transpose(1, 2).masked_fill(
        #             (1 - masks.byte()), torch.tensor(0))
        # print("similarity {}".format(torch.mean(torch.sum(similarity, -1))))

        attention = F.softmax(attention, -1) #N, labels_num, L

        # print("1 {}".format(torch.max(torch.max(attention,-1)[0],-1)[1]))
        # print("2 {}".format(torch.max(torch.max(attention, -1)[0], -1)[0]))
        attention_out = torch.matmul(attention,token_feat)   # N, labels_num, hidden_size
        # print("0 {}".format(torch.sum(token_feat, -1)))
        attention_out_feat = attention_out
        
        attention_out = attention_out * self.class_weight
        
        attention_out = torch.sum(attention_out, -1)
        # print("3 {}".format(torch.max(prob,-1)[1]))
        # print("4 {}".format(torch.max(prob, -1)[0]))

        logit = torch.sigmoid(attention_out)

        # logit = attention_out#self.output(attention_out)
        # logit = prob

        flatten = torch.sum(attention_out, -1, keepdim=True)
 
        fl_func = nn.MSELoss()
        fl = fl_func(feat,attention_out_feat)

        feat = feat * self.class_weight

        feat = torch.sum(feat, -1)

        # print(torch.cat((torch.max(prob,-1),torch.max(attention_out,-1)),-1))
        # print((attention_out_feat - feat).shape)
        # print(attention_out_feat - feat)

        prob = torch.sum(feat, -1, keepdim=True)

        prob = torch.cat((prob,flatten),-1)
        # print(prob)
        # prob = torch.sigmoid(prob)
        # prob = torch.mean(prob, -1)
        # prob = torch.max(prob, -1)[0]
        
        prob = self.output(prob)

        return fl, logit, prob[:,0], attention #[:,0]

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.class_weight, 'lr': lr},
            {'params': self.bert.parameters(), 'lr': lrp},
            {'params': self.Linear1.parameters(), 'lr': lr},
            {'params': self.Linear2.parameters(), 'lr': lr},
        ]


class Generator(nn.Module):
    def __init__(self, bert,num_classes, hidden_dim=768, input_dim=768, num_hidden_generator=2, hidden_dim_generator=2000):
        super(Generator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU(0.2) #nn.Sigmoid()#

        self.num_classes = num_classes

        input_dim += num_classes

        self.num_hidden_generator = num_hidden_generator

        # self.classes_hidden_list = []
        # for i in range(num_classes):
        self.hidden_list_generator = nn.ModuleList()
        for i in range(num_hidden_generator):
            dim = input_dim if i == 0 else hidden_dim_generator
            self.hidden_list_generator.append(nn.Linear(dim, hidden_dim_generator).cuda(0))
        # self.hidden_list_generator.append(nn.Linear(hidden_dim_generator, hidden_dim).cuda(0))
        # self.classes_hidden_list.append(self.hidden_list_generator)

        self.output = nn.Linear(hidden_dim_generator, hidden_dim)

        self.m1 = nn.BatchNorm1d(2000)

        self.add_module('bert', bert)

        for m in self.bert.parameters():
            m.requires_grad = True

        self.class_weight = Parameter(torch.Tensor(num_classes, 768).uniform_(0, 1), requires_grad=False).cuda(0)
        self.class_weight.requires_grad = True

    def forward(self, feat, encoded_tag, tag_mask):

        feat = feat.expand(feat.shape[0], self.num_classes,feat.shape[2])

        # embed = self.bert.get_input_embeddings()
        # tag_embedding = embed(encoded_tag)
        # tag_embedding = torch.sum(tag_embedding * tag_mask.unsqueeze(-1), dim=1) \
        #                 / torch.sum(tag_mask, dim=1, keepdim=True)
        # tag_embedding = tag_embedding.detach().unsqueeze(0).expand_as(feat)

        tag_embedding = torch.eye(self.num_classes).cuda(0).unsqueeze(0).expand(feat.shape[0],self.num_classes,self.num_classes)
        x = torch.cat((feat,tag_embedding),-1)
        # x = feat
  
        for i in range(self.num_hidden_generator):
            x = self.hidden_list_generator[i](x)
            x = self.act(x)
            # x = self.dropout(x)

            # x = self.classes_hidden_list[c][-1](x)
            # if c==0:
            #     y = x
            # else:
            #     y = torch.cat((y,x),1)

        # y = y.expand(y.shape[0], self.num_classes,y.shape[2])
        # y = y * self.class_weight.unsqueeze(0)
        y = self.output(x)

        return y

    def get_config_optim(self, lr):
        return [
            {'params': self.hidden_list_generator.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]

class Bert_Encoder(nn.Module):
    def __init__(self, bert, bert_trainable=True):
        super(Bert_Encoder, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

    def forward(self, ids, token_type_ids, attention_mask):
        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(attention_mask, dim=1, keepdim=True)

        return sentence_feat, token_feat

    def get_config_optim(self, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lrp},
        ]


class Discriminator(nn.Module):
    def __init__(self, num_classes, input_dim=768, num_hidden_discriminator=1, hidden_dim_discriminator=500):
        super(Discriminator, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU(0.2)#nn.ReLU()

        self.num_hidden_discriminator = num_hidden_discriminator
        self.hidden_list_discriminator = nn.ModuleList()
        for i in range(num_hidden_discriminator):
            dim = input_dim if i == 0 else hidden_dim_discriminator
            self.hidden_list_discriminator.append(nn.Linear(dim, hidden_dim_discriminator))

        self.Linear = nn.Linear(hidden_dim_discriminator, (num_classes + 1))
        self.output = nn.Softmax(dim=-1)

    def forward(self, feat):
        # x = self.dropout(feat)
        x = feat
        for i in range(self.num_hidden_discriminator):
            x = self.hidden_list_discriminator[i](x)
            x = self.act(x)
            # x = self.dropout(x)

        flatten = x
        logit = self.Linear(x)
        prob = self.output(logit)
        return flatten, logit, prob

    def get_config_optim(self, lr):
        return [
            {'params': self.hidden_list_discriminator.parameters(), 'lr': lr},
            {'params': self.Linear.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]

class MLPBert(nn.Module):
    def __init__(self, bert, num_classes, hidden_dim, hidden_layer_num, bert_trainable=True):
        super(MLPBert, self).__init__()

        self.add_module('bert', bert)
        if not bert_trainable:
            for m in self.bert.parameters():
                m.requires_grad = False

        self.num_classes = num_classes
        self.hidden_layer_num = hidden_layer_num
        self.hidden_list = nn.ModuleList()
        for i in range(hidden_layer_num):
            if i == 0:
                self.hidden_list.append(nn.Linear(768, hidden_dim))
            else:
                self.hidden_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, ids, token_type_ids, attention_mask, encoded_tag, tag_mask, feat):

        token_feat = self.bert(ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)[0]
        sentence_feat = torch.sum(token_feat * attention_mask.unsqueeze(-1), dim=1) \
                        / torch.sum(attention_mask, dim=1, keepdim=True)

        x = sentence_feat
        for i in range(self.hidden_layer_num):
            x = self.hidden_list[i](x)
            x = self.act(x)
        # y = torch.sigmoid(self.output(x))
        return y, y, y

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.bert.parameters(), 'lr': lrp},
            {'params': self.hidden_list.parameters(), 'lr': lr},
            {'params': self.output.parameters(), 'lr': lr},
        ]