import torch.nn as nn
import torch.nn.functional as F
import pdb

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        #input shape = [128, 3, 32, 32]

        #C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32))
        
        #Output shape = [128, 128, 32, 32] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32))
        
         #Output shape = [128, 512, 32, 32] , RF >> 5
        #--------------TS1 Begins-----------------------
       
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=1),
                                        )
        
        #1*1 convolution output shape = [128, 64, 32, 32], RF>> 5
        self.maxpool1= nn.MaxPool2d(2,2)
        #1*1 convolution output shape = [128, 64, 16, 16], RF>> 6
        #--------------TS1 Ends-----------------------

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32))

        #Convolution output shape = [128,128, 16, 16], RF >> 10

        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32))

        #Convolution output shape = [128,512, 16, 16], RF >> 14

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32))
        
        #Convolution output shape = [128,128, 16, 16], RF >> 18

          #--------------TS2 Begins-----------------------       
        self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=1),
                                        )
        
        #Convolution output shape = [128,64, 16, 16], RF >> 18
        
        self.maxpool2 = nn.MaxPool2d(2,2)
        #Maxpool output shape = [128,64, 8, 8], RF >> 20
          #--------------TS2 Ends-----------------------

        self.convblock8 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32))

        #Convolution output shape = [128,64, 8, 8], RF >> 28
        
        self.convblock9 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=0),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16))
        #Convolution output shape = [128,128, 6, 6], RF >> 36
      
        self.convblock10 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(16)
                                        )
        #Convolution output shape = [128,64, 4, 4], RF >> 44

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        #Convolution output shape = [128,64, 1, 1], RF >> 44

        self.convblock11 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size=1))
        

    def forward(self, x):
        #pdb.set_trace()
        #C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.maxpool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.maxpool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.globalavgpool(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = -1)

        return x
    
class model_layernorm(nn.Module):
    def __init__(self):
        super(model_layernorm, self).__init__()

        #input shape = [128, 3, 32, 32]

        #C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.GroupNorm(1,32),
                                        nn.Dropout2d(0.01)
                                        )
        
        #Output shape = [128, 128, 32, 32] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.GroupNorm(1,32),
                                        nn.Dropout2d(0.01))
        
         #Output shape = [128, 512, 32, 32] , RF >> 5
        #--------------TS1 Begins-----------------------
       
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=1),
                                        )
        
        #1*1 convolution output shape = [128, 64, 32, 32], RF>> 5
        self.maxpool1= nn.MaxPool2d(2,2)
        #1*1 convolution output shape = [128, 64, 16, 16], RF>> 6
        #--------------TS1 Ends-----------------------

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.GroupNorm(1,32),
                                        nn.Dropout2d(0.01))

        #Convolution output shape = [128,128, 16, 16], RF >> 10

        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.GroupNorm(1,32),
                                        nn.Dropout2d(0.01))

        #Convolution output shape = [128,512, 16, 16], RF >> 14

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.GroupNorm(1,32),
                                        nn.Dropout2d(0.01))
        
        #Convolution output shape = [128,128, 16, 16], RF >> 18

          #--------------TS2 Begins-----------------------       
        self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=1),
                                        )
        
        #Convolution output shape = [128,64, 16, 16], RF >> 18
        
        self.maxpool2 = nn.MaxPool2d(2,2)
        #Maxpool output shape = [128,64, 8, 8], RF >> 20
          #--------------TS2 Ends-----------------------

        self.convblock8 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.GroupNorm(1,32),
                                        nn.Dropout2d(0.01))

        #Convolution output shape = [128,64, 8, 8], RF >> 28
        
        self.convblock9 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=0),
                                        nn.ReLU(),
                                        nn.GroupNorm(1,16),
                                        nn.Dropout2d(0.01))
        #Convolution output shape = [128,128, 6, 6], RF >> 36
      
        self.convblock10 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.GroupNorm(1,16),
                                         nn.Dropout2d(0.01)
                                        )
        #Convolution output shape = [128,64, 4, 4], RF >> 44

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        #Convolution output shape = [128,64, 1, 1], RF >> 44

        self.convblock11 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size=1))
        

    def forward(self, x):
        #pdb.set_trace()
        #C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.maxpool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.maxpool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.globalavgpool(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = -1)

        return x
    
class model_groupnorm(nn.Module):
    def __init__(self):
        super(model_groupnorm, self).__init__()

        #input shape = [128, 3, 32, 32]

        #C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10

        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.GroupNorm(4,32),
                                        nn.Dropout2d(0.01))
        
        #Output shape = [128, 128, 32, 32] , RF >> 3

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.GroupNorm(4,32),
                                        nn.Dropout2d(0.01))
        
         #Output shape = [128, 512, 32, 32] , RF >> 5
        #--------------TS1 Begins-----------------------
       
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=1),
                                        )
        
        #1*1 convolution output shape = [128, 64, 32, 32], RF>> 5
        self.maxpool1= nn.MaxPool2d(2,2)
        #1*1 convolution output shape = [128, 64, 16, 16], RF>> 6
        #--------------TS1 Ends-----------------------

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.GroupNorm(4,32),
                                        nn.Dropout2d(0.01))

        #Convolution output shape = [128,128, 16, 16], RF >> 10

        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.GroupNorm(4,32),
                                        nn.Dropout2d(0.01))

        #Convolution output shape = [128,512, 16, 16], RF >> 14

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, padding= 1),
                                        nn.ReLU(),
                                        nn.GroupNorm(4,32),
                                        nn.Dropout2d(0.01))
        
        #Convolution output shape = [128,128, 16, 16], RF >> 18

          #--------------TS2 Begins-----------------------       
        self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=1),
                                        )
        
        #Convolution output shape = [128,64, 16, 16], RF >> 18
        
        self.maxpool2 = nn.MaxPool2d(2,2)
        #Maxpool output shape = [128,64, 8, 8], RF >> 20
          #--------------TS2 Ends-----------------------

        self.convblock8 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.GroupNorm(4,32),
                                        nn.Dropout2d(0.01))

        #Convolution output shape = [128,64, 8, 8], RF >> 28
        
        self.convblock9 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=0),
                                        nn.ReLU(),
                                        nn.GroupNorm(4,16),
                                        nn.Dropout2d(0.01))
        #Convolution output shape = [128,128, 6, 6], RF >> 36
      
        self.convblock10 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.GroupNorm(4,16),
                                         nn.Dropout2d(0.01)
                                        )
        #Convolution output shape = [128,64, 4, 4], RF >> 44

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        #Convolution output shape = [128,64, 1, 1], RF >> 44

        self.convblock11 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size=1))
        

    def forward(self, x):
        #pdb.set_trace()
        #C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.maxpool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.maxpool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.globalavgpool(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = -1)

        return x



        
        