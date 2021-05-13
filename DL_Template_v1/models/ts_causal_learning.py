# -*- utf-8 -*-
# author : fisherwsy

import torch
from torch import nn



class LSTM_base(nn.Module):
    def __init__(self,N,lstm_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(LSTM_base, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)

        # Layers
        self.Q_layer = nn.Linear(N,N,bias=False)  # Query是要生成的的lambda种类
        self.K_layer = nn.Linear(N,N,bias=False)  # Key是发生的事件
        self.V_layer = nn.Linear(N,N,bias=False)  # Values是事件的embedding

        self.lstm = nn.LSTM(N,lstm_output_dim,Lstm_nums_layers)
        self.output_layer = nn.Linear(lstm_output_dim,1)





    def forward(self,Input):
        seq_len,N = Input.shape

        #todo :看看lstm的时间输入顺序有没有弄反？

        # 循环，计算各个类型的lambda
        lstm_input_list = []
        for taget_type in (self.type_mat.to(self.device)):
            # 计算attention的qkv
            query = self.Q_layer(taget_type)        #   quary:(node_num,type_num），当然这里的第二个轴的维度也可以不是type_num,随便什么维度都行，不过
                                                    # 为了方便后面改成加mask矩阵的方法，这里就都统一用type_num
            keys = self.K_layer(Input)    #   keys:(seq_len,N），type_num与query同理
            values = Input
            # values = self.V_layer(event_embedding)  #   values:(seq_len,node_num,type_num），这里的第三个轴的维度同样不必要是type_num，可以随意但是
                                                    # 为了方便修改成加mask矩阵的方法，这里就直接用type_num
            # 邻接举证为F.sigmoid(torch.matmul(self.K_layer.weight.T,self.Q_layer.weight))

            attention_weight = torch.nn.functional.softplus(torch.matmul(keys,query)) # 计算attention的权重
            attention_weight = attention_weight.reshape([*attention_weight.shape, 1])  # 扩展一轴方便广播
            lstm_input = values * attention_weight  # lstm_input:(seq_len,node_num,type_num），维度与values一样
            lstm_input_list.append(lstm_input)  # 用来存多个要出入lstm的

        lstm_inputs = torch.stack(lstm_input_list,1) # lstm_inputs:(seq_len,node_num,type_num, type_num+1)  对于生成不同lambda的输入在batch轴上拼接起来，同时传
        lstm_outputs, _ = self.lstm(lstm_inputs)  # lstm_inputs:(seq_len,node_num×type_num, lstm_output_dim)
        Output = torch.squeeze(self.output_layer(lstm_outputs).view([-1,N]),-1)
        loss = nn.functional.mse_loss(Output[:-1],Input[1:])
        l1 = (torch.abs(torch.sigmoid(torch.matmul(self.K_layer.weight.T, self.Q_layer.weight)))).sum()




        #
        # lambda_after_decay = lambda_[:-1]*torch.exp(-time_diff_embedding.view(-1,1,1)*decay) + base_intensity.view([1,1,-1])
        # log_likelihood_of_event_happen = (torch.log(lambda_after_decay) * event_embedding[1:]).sum()
        # ## lambda的在各段事件间隔对应的积分,即当前事件发生事件到下一个事件发生之间的各个lambda的积分
        # integral_of_base_intensity = (time_diff_embedding.view([-1, 1]) * base_intensity.view([1, -1])).sum() * node_num
        # integral_of_lambda = (lambda_[:-1]) *(1/decay)* (1 - torch.exp(-time_diff_embedding.view(-1, 1, 1) * decay))
        # log_likelihood_of_event_not_happen = -(integral_of_lambda.sum() + integral_of_base_intensity)

        # # 计算likelihood
        # base_intensity = F.softplus(self.layer_for_base_intensity(self.type_mat))
        # ## 发生事件时，对应的lambda值(PS：还没受到正在发生的事件的影响，所以要计算上一个lambda经过衰减后的值)
        # lambda_after_decay = lambda_[:-1]*decay_[:-1]*torch.exp(-time_diff_embedding.view(-1,1,1)*decay_[:-1]) + base_intensity.view([1,1,-1])     # lambda_after_decay:(seq_len-1, node_num, type_num)  lambda乘上一个衰减系数为decay的指数衰减函数
        # log_likelihood_of_event_happen = (torch.log(lambda_after_decay) * event_embedding[1:]).sum()    # 注意event_embedding的切片，当前事件发不发生是由之前的lambda和decay决定的
        #
        # ## lambda的在各段事件间隔对应的积分,即当前事件发生事件到下一个事件发生之间的各个lambda的积分
        # integral_of_base_intensity = (time_diff_embedding.view([-1,1])*base_intensity.view([1,-1])).sum()*node_num
        # integral_of_lambda = (lambda_[:-1])*(1-torch.exp(-time_diff_embedding.view(-1,1,1)*decay_[:-1]))    # lambda_after_decay:(seq_len-1, node_num, type_num)
        #                                                                                                                 # 其中(i,j,k)表示，第i个事件发生到i+1发生之间,节点j的lambda_k的积分
        # log_likelihood_of_event_not_happen = -(integral_of_lambda.sum()+integral_of_base_intensity)


        return loss,l1

    def get_adj(self):
        return torch.relu(torch.matmul(self.K_layer.weight, self.Q_layer.weight)).detach().to('cpu')



class Linear_base_1(nn.Module):
    def __init__(self,N,lstm_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(Linear_base_1, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)

        # Layers
        self.Q_layer = nn.Linear(N,N,bias=False)  # Query是要生成的的lambda种类
        self.K_layer = nn.Linear(N,N,bias=False)  # Key是发生的事件
        self.V_layer = nn.Linear(N,N,bias=False)  # Values是事件的embedding

        self.output_layer = nn.Linear(N,1)





    def forward(self,Input):
        seq_len,N = Input.shape

        #todo :看看lstm的时间输入顺序有没有弄反？

        # 循环，计算各个类型的lambda
        lstm_input_list = []
        for taget_type in (self.type_mat.to(self.device)):
            # 计算attention的qkv
            query = self.Q_layer(taget_type)        #   quary:(node_num,type_num），当然这里的第二个轴的维度也可以不是type_num,随便什么维度都行，不过
                                                    # 为了方便后面改成加mask矩阵的方法，这里就都统一用type_num
            keys = self.K_layer(Input)    #   keys:(seq_len,N），type_num与query同理
            values = Input
            # values = self.V_layer(event_embedding)  #   values:(seq_len,node_num,type_num），这里的第三个轴的维度同样不必要是type_num，可以随意但是
                                                    # 为了方便修改成加mask矩阵的方法，这里就直接用type_num
            # 邻接举证为F.sigmoid(torch.matmul(self.K_layer.weight.T,self.Q_layer.weight))

            attention_weight = torch.nn.functional.softplus(torch.matmul(keys,query)) # 计算attention的权重
            attention_weight = attention_weight.reshape([*attention_weight.shape, 1])  # 扩展一轴方便广播
            lstm_input = values * attention_weight  # lstm_input:(seq_len,node_num,type_num），维度与values一样
            lstm_input_list.append(lstm_input)  # 用来存多个要出入lstm的

        lstm_inputs = torch.stack(lstm_input_list,1) # lstm_inputs:(seq_len,node_num,type_num, type_num+1)  对于生成不同lambda的输入在batch轴上拼接起来，同时传
        Output = lstm_inputs.sum(2)
        loss = nn.functional.mse_loss(Output[:-1],Input[1:])
        l1 = (torch.abs(torch.sigmoid(torch.matmul(self.K_layer.weight.T, self.Q_layer.weight)))).sum()



        return loss,l1

    def get_adj(self):
        return torch.relu(torch.matmul(self.K_layer.weight, self.Q_layer.weight)).detach().to('cpu')



class Linear_base_3(nn.Module):
    def __init__(self,N,lstm_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(Linear_base_3, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)

        # Layers
        self.Q_layer = nn.Linear(N,N,bias=False)  # Query是要生成的的lambda种类
        self.K_layer = nn.Linear(N,N,bias=False)  # Key是发生的事件
        self.V_layer = nn.Linear(N,N,bias=False)  # Values是事件的embedding

        self.output_layer = nn.Linear(N,N)





    def forward(self,Input):
        seq_len,N = Input.shape
        Output = torch.squeeze(self.output_layer(Input).view([-1,N]),-1)
        loss = nn.functional.mse_loss(Output[:-1],Input[1:])
        l1 = None



        return loss,l1

    def get_adj(self):
        return (self.output_layer.weight).detach().to('cpu')

class Linear_base(nn.Module):
    def __init__(self,N,lstm_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(Linear_base, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)

        # Layers
        self.causal_layer = nn.Linear(N,N,bias=False)



    def forward(self,Input):
        seq_len,N = Input.shape
        self.weight = torch.nn.functional.relu(self.causal_layer(self.type_mat)) # 维度NxN
        cl_output = Input.unsqueeze(1).expand([-1,N,-1])*self.weight.T


        Output = cl_output.sum(2)

        loss = nn.functional.mse_loss(Output[:-1],Input[1:])
        l1 = (torch.abs(self.causal_layer.weight).sum())



        return loss,l1

    def get_adj(self):
        return self.weight.detach().to('cpu')



class lstm_base(nn.Module):
    def __init__(self,N,lstm_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(lstm_base, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)

        # Layers
        self.causal_layer = nn.Linear(N,N,bias=False)
        self.causal_layer.weight = nn.Parameter(torch.Tensor(torch.ones(N,N)).float())
        self.lstm = nn.LSTM(N,lstm_output_dim,Lstm_nums_layers)
        self.output_layer = nn.Linear(lstm_output_dim,1)






    def forward(self,Input):
        seq_len,N = Input.shape
        self.weight = torch.nn.functional.relu6(self.causal_layer(self.type_mat)) # 维度NxN
        cl_output = Input.unsqueeze(1).expand([-1,N,-1])*self.weight.T

        lstm_output,_ = self.lstm(cl_output)

        Output = self.output_layer(lstm_output).squeeze(2)

        loss = nn.functional.mse_loss(Output[:-1],Input[1:])
        l1 = (torch.abs(self.causal_layer.weight).sum())




        return loss,l1

    def get_adj(self):
        return self.weight.detach().to('cpu')


class Dense_base(nn.Module):
    def __init__(self,N,lstm_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(Dense_base, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)

        # Layers
        self.causal_layer = nn.Linear(N,N,bias=False)
        self.output_layer = nn.Linear(N,1)



    def forward(self,Input):
        seq_len,N = Input.shape
        self.weight = torch.nn.functional.relu(self.causal_layer(self.type_mat)) # 维度NxN
        cl_output = Input.unsqueeze(1).expand([-1,N,-1])*self.weight.T
        Output = self.output_layer(cl_output).squeeze(2)

        loss = nn.functional.mse_loss(Output[:-1],Input[1:])
        l1 = (torch.abs(self.causal_layer.weight).sum())




        return loss,l1

    def get_adj(self):
        return self.weight.detach().to('cpu')


class MLP_base(nn.Module):
    def __init__(self,N,lstm_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(MLP_base, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)

        # Layers
        self.causal_layer = nn.Linear(N,N,bias=False)
        # self.causal_layer.weight = nn.Parameter(torch.Tensor(torch.ones(10,10)).float())
        self.hidden_layer = nn.Linear(N,6)
        self.output_layer = nn.Linear(6,1)



    def forward(self,Input):
        seq_len,N = Input.shape
        self.weight = torch.nn.functional.relu(self.causal_layer(self.type_mat)) # 维度NxN
        cl_output = Input.unsqueeze(1).expand([-1,N,-1])*self.weight.T
        hd_output = nn.functional.relu(self.hidden_layer(cl_output))
        Output = self.output_layer(hd_output).squeeze(2)

        loss = nn.functional.mse_loss(Output[:-1],Input[1:])
        l1 = (torch.abs(self.causal_layer.weight).sum())




        return loss,l1

    def get_adj(self):
        return self.weight.detach().to('cpu')


class CNN_base(nn.Module):
    def __init__(self,N,cnn_output_dim,Lstm_nums_layers,device = 'cpu'):
        super(CNN_base, self).__init__()
        '''
        :param event_embedding: 事件类型的one-hot编码，(seq_len,node_num,type_num)
        :param time_embedding:  时间的编码，(seq_len,1)，这里暂时直接用时间
        '''
        self.device = device
        self.type_mat = torch.eye(N, N).to(self.device)
        # self.cnn_kernel_size = cnn_kernel_size
        # Layers
        self.causal_layer = nn.Linear(N,N,bias=False)
        self.causal_layer.weight = nn.Parameter(torch.Tensor(torch.ones(N,N)).float())
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(N, 10, 2),
            nn.ReLU(),
            nn.Conv1d(10, cnn_output_dim, 2),
            nn.ReLU()

        )
        # self.cnn_layer = torch.nn.Conv1d(N, cnn_output_dim, cnn_kernel_size)
        nn.init.ones_(self.cnn_layer[0].weight)
        self.output_layer = nn.Linear(cnn_output_dim,1)




    def forward(self,Input):
        seq_len,N = Input.shape
        self.weight = torch.nn.functional.relu6(self.causal_layer(self.type_mat)) # 维度NxN
        cl_output = Input.unsqueeze(1).expand([-1,N,-1])*self.weight.T

        cnn_input = cl_output.permute([1,2,0])
        cnn_output = self.cnn_layer(cnn_input)
        cnn_output_ = cnn_output.permute([2,0,1])

        Output = self.output_layer(cnn_output_).squeeze(2)

        loss = nn.functional.mse_loss(Output[:-1],Input[-len(Output)+1:])
        l1 = (torch.abs(self.causal_layer.weight).sum())




        return loss,l1

    def get_adj(self):
        return self.weight.detach().to('cpu')
