# -*- coding: utf-8 -*-

"""
Created on Thu Oct  3 09:20:25 2019
By: Nguyen Le Xuan Phuoc 
@author: HadesSecurity
"""
import os
import torchvision
import torch
from torchvision import transforms
import numpy
import matplotlib.pyplot as plt
import torch.optim as optim

import tkinter.constants as Tkconstants
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import Frame, Tk, BOTH, Menu,Label,Button,Entry,END,Listbox,filedialog
from tkinter.filedialog import Open,asksaveasfilename
from PIL import ImageTk
import numpy as np
# CUDA?
cuda = torch.cuda.is_available()

#==============================================================================
# chức năng hiển thị hình ảnh    
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
    plt.show()

def showimagelabel(image,label,x0,y0):
    image = image / 2 + 0.5
    img = transforms.ToPILImage()(image)
    img = transforms.Resize(126)(img)
    img = transforms.CenterCrop(126)(img)
    img2 = ImageTk.PhotoImage(img)
    label0 = Label(frame2,height=126,width=126,image=img2)
    label0.image = img2
    label0.place(x = x0, y = y0)
    label10 = Label(frame2,text=label,fg = "red",font="Times 14 bold")
    label10.place(x = x0 + 20, y = y0 + 150)    

def load():
      filename = filedialog.askdirectory()
      a = os.listdir(filename) 
      if(a[0]=='test' and a[1]=='train' and len(a)==2):
#      messagebox.showinfo("Thông báo", "Tải dữ liệu lên thành công")
       #đường dẫn cho folder train và folder test
           global testpath,trainpath
           testpath = filename  +'/test'
           trainpath = filename +'/train'
      global frame3 
      frame3 = Frame(frame,height=520,width=1050)
      frame3.place(x=450,y=200)
      label12 = Label(frame,fg = "Red",font="Times 24 bold")
      label12.place(x=650,y=200)
      
      label9 = Label(frame,fg = "black",
                   font= "Times 20 bold")
      label9.place(x=450,y=300)    
      label10 = Label(frame,fg = "black",
                    font= "Times 20 bold")
      label10.place(x=450,y=250)
      label13 = Label(frame,fg = "black",
                    font= "Times 20 bold")
      label13.place(x=450,y=350)
      
      #một chuỗi biến đổi để xử lý trước hình ảnh:
      #Compose tạo ra một loạt các biến đổi để chuẩn bị bộ dữ liệu.
      #Torchvision đọc các bộ dữ liệu vào PILImage (định dạng hình ảnh Python). 
      #ToTensor chuyển đổi PIL Image từ phạm vi [0, 255] thành FloatTensor
      #có hình dạng (C x H x W) với phạm vi [0.0, 1.0].
      #Sau đó, chúng tôi tái chuẩn hóa đầu vào thành [-1, 1] dựa trên công thức sau với
      #         μ=standard deviation=0.5.
      # input=(input−μ)/standard deviation
      # input=(input−0.5)/0.5
      transform = transforms.Compose(
         [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    
      size_of_batch = 4
      if (entry2.get() != ""):
         size_of_batch = int(entry2.get())
    
      '''
      root (string):Thư mục gốc của tập dữ liệu
      transform (callable, optional) :Hàm / biến đổi lấy trong ảnh PIL và 
              trả về phiên bản đã chuyển đổi. Ví dụ: biến đổi.RandomCrop
      target_transform (có thể gọi, tùy chọn): Một hàm / biến đổi nhận trong mục tiêu và biến đổi nó.
      '''
      train_data = torchvision.datasets.ImageFolder(root=trainpath,
                                                    transform=transform)
      
      #Trình tải dữ liệu. Kết hợp một tập dữ liệu và bộ lấy mẫu và cung cấp một 
      #lần lặp qua tập dữ liệu đã cho.
      #DataLoader hỗ trợ cả bộ dữ liệu kiểu bản đồ và kiểu lặp có thể tải một 
      #hoặc nhiều quá trình, tùy chỉnh thứ tự tải và tự động ghép (đối chiếu) và ghim bộ nhớ.
      '''
      dataset (Dataset)[train_data] – tập dữ liệu từ đó để tải dữ liệu.
      batch_size (int, optional):Có bao nhiêu mẫu trên mỗi lô để tải (mặc định: 4).
      shuffle (bool, optional)(xáo trộn):được đặt thành True để dữ liệu được chia sẻ lại ở mỗi epoch (mặc định: False).
      sampler (Sampler, optional) (bộ lấy mẫu): xác định chiến lược để lấy mẫu từ bộ dữ liệu. 
                                                Nếu được chỉ định, xáo trộn phải là Sai.
      batch_sampler (Sampler, optional):giống như bộ lấy mẫu, nhưng trả về một loạt các chỉ số tại một thời điểm.
                                      Loại trừ lẫn nhau với batch_size, shuffle, sampler và drop_last.
      num_workers (int, optional):có bao nhiêu quy trình sử dụng để tải dữ liệu. 
                      0 có nghĩa là dữ liệu sẽ được tải trong quy trình chính. (mặc định: 0)
      collate_fn(hợp nhất một danh sách các mẫu để tạo thành một lô nhỏ của Tenor) ử dụng tải theo đợt từ bộ dữ liệu kiểu bản đồ.
      drop_last (bool, tùy chọn) - được đặt thành True để thả lô không hoàn chỉnh cuối cùng, 
                  nếu kích thước tập dữ liệu không chia hết cho kích thước lô.
                  Nếu Sai và kích thước của tập dữ liệu không chia hết cho 
                  kích thước lô thì đợt cuối cùng sẽ nhỏ hơn. (mặc định: false)
      '''
      train_data_loader = torch.utils.data.DataLoader(train_data,
                                                      batch_size=size_of_batch,
                                                      shuffle=True,num_workers=0)
    
      test_data = torchvision.datasets.ImageFolder(root= testpath , 
                                                   transform=transform)
      test_data_loader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=size_of_batch,
                                                     shuffle=True,num_workers=0)
      ## nhận được một số hình ảnh đào tạo ngẫu nhiên 
      datatrain = iter(train_data_loader)
      image, target = datatrain.next()
      datatest = iter(test_data_loader)
      imagetest, label = datatest.next()
      
      label12.config(text = "Load Dataset thành công!")
      t0="+ Gồm 4 lớp, mỗi lớp có %d"%(((len(train_data_loader)/4)+(len(test_data_loader))/4))+" ảnh"
      t1 ="+Tập train: %d"% (len(train_data_loader))+" chia đều cho 4 lớp (mỗi lớp %d"%(len(train_data_loader)/4) + " ảnh)"
      t2="+Tập test: %d"%(len(test_data_loader))+ " chia đều cho 4 lớp (mỗi lớp %d"%(len(test_data_loader)/4) + " ảnh)"
      label9.config(text = t1)
      label10.config(text = t0)
      label13.config(text = t2)
    
      global lbl124
      lbl124 = Label(frame,text = "- Một vài hình ảnh mẫu:",fg = "black",font= "Times 20 bold")
      lbl124.place(x=450,y=400)    
      
      classes =("frog", "horse", "ship", "truck")
      dataiter = iter(train_data_loader)
      images, labels = dataiter.next()
      global frame2
      frame2 = Frame(frame,height=520,width=1050)
      frame2.place(x=450,y=450)
      imshow(torchvision.utils.make_grid(images))
      
      x = -120
      y = 30
      for i in range(size_of_batch):
          x = x + 150
          if (x > 1100):
              y = y + 220
              x = x + 150
          showimagelabel(images[i],classes[labels[i]],x,y)

      
#==============================================================================
# Sao chép mạng nơ-ron từ phần Mạng nơ-ron trước đó và sửa đổi nó 
# để chụp ảnh 3 kênh (thay vì hình ảnh 1 kênh như được xác định)
#Input > Conv (ReLU) > MaxPool > Conv (ReLU) > MaxPool > FC (ReLU) > FC (ReLU) > FC (SoftMax) > 10 outputs         
#==============================================================================
# Mô hình mạng khi chạy với CiFAR 10:32 x 32 x 3
class Net(nn.Module):
    # Hàm khởi tạo một số thành phần của lớp Net
    def __init__(self):
        super(Net, self).__init__()
        
        '''torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
        #                  padding=0, dilation=1, groups=1, bias=True)
        '''
        # Lớp tích chập (Convolution layer): Conv2d (tra cứu trong PyTorch Docs)
        # Số kênh vào (in_channels=Cin) của dataset CIFAR10: 3 kênh (đỏ, xanh lá cây, xanh dương) mỗi kích thước 32x32 pixel.
        # Số kênh ra (out_channels=Cin∗K)= số bộ lọc (filter, kernel): 6 mỗi kích thước 3x5x5.
        # Kích thước 1 bộ lọc (kernel_size): 5 = (5, 5)
        # Bước dịch chuyển (stride): 1 = (1, 1)
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        '''nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)'''
        #Áp dụng pooling tối đa 2 chiều trên tín hiệu đầu vào bao gồm nhiều mặt phẳng đầu vào.
        #kernel_size - kích thước của cửa sổ để lấy tối đa
        #           một int duy nhất - trong trường hợp đó, cùng một giá trị được sử dụng cho kích thước chiều cao và chiều rộng
        #           một tuple của hai int - trong trường hợp đó, int đầu tiên được sử dụng cho kích thước chiều cao 
        #           và int thứ hai cho kích thước chiều rộng
        #stride  -stride  của cửa sổ. Giá trị mặc định là kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        
        # Lớp tích chập (Convolution layer): Conv2d (tra cứu trong PyTorch Docs)
        # Số kênh vào (in_channels) = số kênh ra của lớp trước đó: 6
        # Số kênh ra (out_channels) = số bộ lọc (filter, kernel): 16
        # Kích thước 1 bộ lọc (kernel_size): 5 = (5, 5)
        # Bước dịch chuyển (stride): 1 = (1, 1)        
        self.conv2 = nn.Conv2d(6, 16, 5)
       
        #Fully_Connected Layer (FC):Tên tiếng viết là Mạng liên kết đầy đủ
        #Lớp liên kết đầy đủ nn.Linear
        #Áp dụng phép chuyển đổi tuyến tính cho dữ liệu đến: y = xAT + b
        # Lớp kết nối đầy đủ (Fully Conectted layer): Linear (tra cứu trong PyTorchDocs)
        #có hình dạng [batch_size, Kênh = 16, height = 5, width = 5]. 
        #Để truyền kích hoạt này cho nn.Linear, bạn đang làm phẳng tenxơ này thành [batch_size, 16 * 5 * 5].
        # Số kênh vào (in_features ) = số kênh ra của lớp trước đó: 16*5*5 = 400
        # Số kênh ra (out_features ) = số bộ lọc (filter, kernel):120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)        
        # self.fc2 = nn.Linear(120, 84)
        # Lớp kết nối đầy đủ (Fully Conectted layer): Linear (tra cứu trong PyTorchDocs)
        # Số kênh vào (in_channels) = số kênh ra của lớp trước đó: 120
        # Số kênh ra (out_channels) = số bộ lọc (filter, kernel): 84 (số lớp của classifier)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        # Lớp kết nối đầy đủ (Fully Conectted layer): Linear (tra cứu trong PyTorchDocs)
        # Số kênh vào (in_channels) = số kênh ra của lớp trước đó: 84
        # Số kênh ra (out_channels) = số bộ lọc (filter, kernel):10 (số lớp của classifier)
        self.fc3 = nn.Linear(84, 10)
        '''
        Các thông số: nn.Linear
         in_features - kích thước của mỗi mẫu đầu vào
         out_features - kích thước của mỗi mẫu đầu ra
         bias - Nếu được đặt thành False, lớp sẽ học bias thêm vào. Mặc định:
        True
        Công thức:
         Đầu vào: (N, ∗,in_features), trong đó ∗ có nghĩa là bất kỳ nào
         Đầu ra: (N, ∗,out_features), trong đó tất cả các kích thước trừ kích
        thước cuối cùng có cùng dạng với đầu vào.
        '''
    
    # Mô hình mạng nơron
    def forward(self, x):
        # Kích thước ảnh đầu vào (shape): x = 32 x 32 x 3
        # Kích thước ảnh đầu ra của lớp (shape): 28 x 28 x 6 (giảm đi 4)
        x = self.pool(F.relu(self.conv1(x)))
        
        # Lớp, hàm Pooling (tra cứu trong PyTorch Docs)
        # Kích thước 1 bộ lọc (kernel_size): 2 = (2, 2)
        # Bước dịch chuyển (stride): 2
        # Kích thước ảnh đầu ra của lớp (shape): 14 x 14 x 6, (bị giảm 1 nửa)
        x = self.pool(F.relu(self.conv2(x)))
        
        # Chuyển đổi (reshape) ảnh về dạng vecto: 1 x 400
        x = x.view(-1, 16 * 5 * 5)
        
        # Kích thước ảnh đầu ra của lớp (shape): 1x120
        x = F.relu(self.fc1(x))
        
         # Kích thước vecto đầu ra của lớp (shape): 1 x 84
        x = F.relu(self.fc2(x))
        
         # Kích thước vecto đầu ra của lớp (shape): 1 x 10 (10 là số lớp của classifier)
        x = self.fc3(x)
        return x

net = Net()

#Xác định mô hình mạng nơ ron
def modul_net():
    print(net)
    global frame3 
    frame3 = Frame(frame,height=520,width=1050)
    frame3.place(x=450,y=200)
    label9.config(text = "")
    label10.config(text = "")
    label13.config(text = "")
    lbl124.config(text = "")
    frame3 = Frame(frame,height=520,width=1050)
    frame3.place(x=450,y=450)
    label12 = Label(frame,fg = "Red",font="Times 18 bold")
    label12.place(x=450,y=200)
    label12.config(text="%s"%net)
#==============================================================================    
   #hàm mất mát
def loss_function():
    global net,frame3 
    frame3 = Frame(frame,height=520,width=1050)
    frame3.place(x=450,y=200)
#    loss= torch.nn.CrossEntropyLoss(weight=None, size_average=None, 
#                                    ignore_index=-100, reduce=None,
#                                    reduction='mean')
    
#------------------------------------------------------------------------------
#    torch.nn.CrossEntropyLoss(weight=None, size_average=None, 
#                               ignore_index=-100, reduce=None,
#                                reduction='mean')
#   - weight: một trọng lượng thay đổi kích thước thủ công cho mỗi lớp. 
#             Nếu được cung cấp, phải là một thang đo kích thước C
#   - size_average:Theo mặc định, các tổn thất được tính 
#                  trung bình trên mỗi phần tử tổn thất trong lô. 
#                   Lưu ý rằng đối với một số tổn thất, có nhiều yếu tố trên mỗi mẫu.
#                   Nếu kích thước trường_bảo hiểm được đặt thành Sai, thay vào đó, 
#                   tổn thất sẽ được tính tổng cho mỗi xe buýt nhỏ. 
#                   Bỏ qua khi giảm là Sai. Mặc định: Đúng
#   - ignore_index: Chỉ định giá trị đích bị bỏ qua và không đóng góp vào gradient đầu vào.
#                    Khi size_average là True, tổn thất được tính trung bình trên các mục tiêu 
#                    không bị bỏ qua
#    - reduce : Theo mặc định, các tổn thất được tính trung bình hoặc tổng hợp trên
#                các quan sát cho mỗi xe buýt nhỏ tùy thuộc vào kích thước bảo hiểm. 
#                   Thay vào đó, khi giảm là Sai, trả về tổn thất cho mỗi phần tử thay thế 
#                   và bỏ qua size_alusive. Mặc định: Đúng
#    -reduction: Chỉ định giảm để áp dụng cho đầu ra
#------------------------------------------------------------------------------
    loss = nn.CrossEntropyLoss()
#    ex:hàm mất mát
#    input = torch.randn(3, 5, requires_grad=True)
#    target = torch.empty(3, dtype=torch.long).random_(5)
#    output = loss(input, target)
#    output.backward()
    
    '''
    -torch.optim.SGD
    Thông số:
        params(iterable) :lặp lại các tham số để tối ưu hóa hoặc xác định các nhóm tham số
        lr (float) :tỷ lệ học
        momentum (float, optional):hệ số động lượng (mặc định: 0)
        weight_decay (float, optional) :phân rã trọng lượng (hình phạt L2) (mặc định: 0)
        dampening (float, optional):giảm chấn cho đà (mặc định: 0)
        nesterov (bool, optional):cho phép đà Nesterov (mặc định: false)
    '''
    optimizer = optim.SGD(net.parameters(),
                          lr=0.001,
                          momentum=0.9)
    print (loss)
    label12 = Label(frame,fg = "Red",font="Times 18 bold")
    label12.place(x=450,y=200)
    label12.config(text = "Hàm mất mát: %s"%loss )#+" là %s"%output)
    
    label14 = Label(frame,fg = "#FF3399",font="Times 18 bold")
    label14.place(x=450,y=250)
    label14.config(text = "Tối ưu hóa:")
    
    label13 = Label(frame,fg = "#663366",font="Times 18 bold")
    label13.place(x=500,y=280)
    label13.config(text = "%s"%optimizer)


    
def training():
      
      global net,frame5 
      frame5 = Frame(frame,height=520,width=1050)
      frame5.place(x=450,y=200)
      label12 = Label(frame,fg = "Red",font="Times 24 bold")
      label12.place(x=650,y=200)
      
      label11 = Label(frame,fg = "black",font= "Times 14 bold")
      label11.place(x=1100,y=200)
      
      #một chuỗi biến đổi để xử lý trước hình ảnh:
      #Compose tạo ra một loạt các biến đổi để chuẩn bị bộ dữ liệu.
      #Torchvision đọc các bộ dữ liệu vào PILImage (định dạng hình ảnh Python). 
      #ToTensor chuyển đổi PIL Image từ phạm vi [0, 255] thành FloatTensor
      #có hình dạng (C x H x W) với phạm vi [0.0, 1.0].
      #Sau đó, chúng tôi tái chuẩn hóa đầu vào thành [-1, 1] dựa trên công thức sau với
      #         μ=standard deviation=0.5.
      # input=(input−μ)/standard deviation
      # input=(input−0.5)/0.5
      transform = transforms.Compose(
         [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    
      size_of_batch = 4
      if (entry2.get() != ""):
         size_of_batch = int(entry2.get())
         
      '''
      dataset (Dataset)[train_data] – tập dữ liệu từ đó để tải dữ liệu.
      batch_size (int, optional):Có bao nhiêu mẫu trên mỗi lô để tải (mặc định: 4).
      shuffle (bool, optional)(xáo trộn):được đặt thành True để dữ liệu được chia sẻ lại ở mỗi epoch (mặc định: False).
      sampler (Sampler, optional) (bộ lấy mẫu): xác định chiến lược để lấy mẫu từ bộ dữ liệu. 
                                                Nếu được chỉ định, xáo trộn phải là Sai.
      batch_sampler (Sampler, optional):giống như bộ lấy mẫu, nhưng trả về một loạt các chỉ số tại một thời điểm.
                                      Loại trừ lẫn nhau với batch_size, shuffle, sampler và drop_last.
      num_workers (int, optional):có bao nhiêu quy trình sử dụng để tải dữ liệu. 
                      0 có nghĩa là dữ liệu sẽ được tải trong quy trình chính. (mặc định: 0)
      collate_fn(hợp nhất một danh sách các mẫu để tạo thành một lô nhỏ của Tenor) ử dụng tải theo đợt từ bộ dữ liệu kiểu bản đồ.
      drop_last (bool, tùy chọn) - được đặt thành True để thả lô không hoàn chỉnh cuối cùng, 
                  nếu kích thước tập dữ liệu không chia hết cho kích thước lô.
                  Nếu Sai và kích thước của tập dữ liệu không chia hết cho 
                  kích thước lô thì đợt cuối cùng sẽ nhỏ hơn. (mặc định: false)
      '''    
      train_data = torchvision.datasets.ImageFolder(root=trainpath,
                                                    transform=transform)
      train_data_loader = torch.utils.data.DataLoader(train_data,
                                                      batch_size=size_of_batch,
                                                      shuffle=True,num_workers=0)
    
      test_data = torchvision.datasets.ImageFolder(root= testpath,
                                                   transform=transform)
      test_data_loader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=size_of_batch,
                                                     shuffle=True,num_workers=0)
        
      datatrain = iter(train_data_loader)
      image, target = datatrain.next()
      datatest = iter(test_data_loader)
      imagetest, label = datatest.next()
    
    
      label6 = Label(frame,fg = "#003300",font="Times 16 bold")
      label6.place(x=450,y=250)
      label7 = Label(frame,fg = "#003300",font="Times 16 bold")
      label7.place(x=450,y=280) 
      label8 = Label(frame,fg = "#003300",font="Times 16 bold")
      label8.place(x=450,y=310)
    
      classes =["frog", "horse", "ship", "truck"]
    
      print(target)
      print('Example Train: ', ' '.join('%10s' % target[j] for j in range(size_of_batch)))
      
      global net
    
      #Hàm mất mát và tối ưu hóa
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.SGD(net.parameters(),
                            lr=0.001,
                            momentum=0.9)
      
      epoch = 1
      if (entry.get() != ""):
          epoch = int(entry.get())
      
      t = ''
      for epoch in range(epoch):
          running_loss = 0.0
          for i, data in enumerate(train_data_loader, 0):
              # # nhận được các đầu vào; dữ liệu là một danh sách các [đầu vào, nhãn]
              inputs, labels = data
              
              '''
              Một số thuật toán tối ưu hóa như Conjugate Gradient và LBFGS 
              cần đánh giá lại hàm nhiều lần, do đó bạn phải vượt qua trong
              một bao đóng cho phép chúng tính toán lại mô hình của bạn.
              Việc đóng sẽ xóa độ dốc, tính toán tổn thất và trả lại.
              '''
              # Zero tham số gradient #bỏ hết đạo hàm cũ
              optimizer.zero_grad() #Xóa độ dốc của tất cả được tối ưu hóa torch.Tensor s.
              #Không có bộ đệm gradient của tất cả các tham số và backprops với độ dốc ngẫu nhiên
    
              # forward + backward + optimize
              outputs = net(inputs)
              
              loss = criterion(outputs, labels)
              '''
              Để sao lưu lỗi, tất cả những gì chúng ta phải làm là mất.backward ().
              Bạn cần phải xóa các gradient hiện có, các gradient khác sẽ được 
              tích lũy vào các gradient hiện có.
              Bây giờ chúng ta sẽ gọi loss.backward () và 
              xem xét độ dốc thiên vị của CON1 trước và sau khi lùi.
              '''
              loss.backward()
              
              #Thực hiện một bước tối ưu hóa: 
              optimizer.step()
              #step(closure): Thực hiện một bước tối ưu hóa duy nhất (cập nhật tham số).
              #             closure:Một bao đóng đánh giá lại mô hình và trả lại tổn thất.
             #              Tùy chọn cho hầu hết các tối ưu hóa.
             
    
              # print statistics
              running_loss += loss.item()
              if i % 2000 == 1999:    # in mỗi 2000 lô nhỏ               
                  t = t +'\n[Epoch: %d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000)
                  print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
#                  global listbox1
                  label11.config(text=t)
#                  listbox1 = Listbox(frame, height=8, width=40, xscrollcommand=True,selectmode=MULTIPLE)
#                  listbox1.place(x=1090,y=200)
#                  listbox1.insert(tk.END,str(t),xview_scroll(1))
#                  listbox1.update()                          
                  running_loss = 0.0
                
      label12.config(text = "Hoàn tất huấn luyện!")
      # show images
      imshow(torchvision.utils.make_grid(image))
      t = 'Tập training:    ' + ' '.join('%s,' % classes[label[j]] for j in range(size_of_batch))
      label6.config(text = t)
      outputs = net(image)
      _, predicted = torch.max(outputs, 1)
      t = 'Dự đoán: ' + ' '.join('%s,' % classes[predicted[j]] for j in range(size_of_batch))
      label7.config(text = t)
      outputs = net(image)
    
      #xem cách mạng thực hiện trên toàn bộ dữ liệu.
      correct = 0
      total = 0
      confusion_matrix = np.zeros([4,4], int)
      with torch.no_grad():
          for data in train_data_loader:
              images, labels = data
              outputs = net(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              for i, l in enumerate(labels):
                  confusion_matrix[l.item(), predicted[i].item()] += 1 
      print(total)
      model_accuracy = correct / total * 100
      print('Model accuracy on {0} test images: {1:.2f}%'.format(total, model_accuracy))
      
      print('{0:10s} - {1}'.format('Category','Accuracy'))
      for i, r in enumerate(confusion_matrix):
          print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
    
      t = 'Độ chính xác của mạng trên 2000 hình ảnh training: %d %%' % (100 * correct / total)
      label8.config(text = t)
      outputs = net(image)

      global frame2
      frame2 = Frame(frame,height=520,width=1050)
      frame2.place(x=420,y=350)
      x = -120
      y = 30
      for i in range(size_of_batch):
          x = x + 150
          if (x > 1100):
              y = y + 220
              x = -120 + 150
          showimagelabel(imagetest[i],classes[label[i]],x,y)  
          
      root1 = Tk()
      root1.title("bieudo")
      #Hiển thị biểu đồ tổng quan
      fr7=tk.Frame(root1,bg="white", borderwidth=5, relief='sunken')
      fr7.grid(row=1, column=3, rowspan=3, padx=10)
      lb3=tk.Label(fr7, text='Ma Trận Trực Quan',bg="white", font='Times 14 bold')
      lb3.grid(row=0, column=0)
      fig, ax = plt.subplots(1,1,figsize=(8,6))
      ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
      plt.ylabel('Thực tế')
      plt.yticks(range(4), classes)
      plt.xlabel('Dự đoán')
      plt.xticks(range(4), classes)
      plt.show()
      canvas = FigureCanvasTkAgg(fig, master=fr7)
      canvas.get_tk_widget().grid(row=1,column=0, sticky=Tkconstants.NSEW, rowspan=2)
      canvas.draw()
      root1.geometry("700x500")
      root1.mainloop()

    
    
def save():
    ftypes = [('PT files', '*.pt'), ('All files', '*')]
    dialog = asksaveasfilename(filetypes = ftypes,
                               defaultextension='.pt')
    global net
    if dialog != '':
        torch.save(net,dialog)
    else:
        torch.save(net,'./model/net.pt')
     
    label12.config(text = "Đã lưu!")

def test_net():
      global net,frame6 
      frame6 = Frame(frame,height=520,width=1050)
      frame6.place(x=450,y=200)
      label12 = Label(frame,fg = "Red",font="Times 24 bold")
      label12.place(x=650,y=200)
      label6 = Label(frame,fg = "#003300",font="Times 20 bold")
      label6.place(x=450,y=250)
      label7 = Label(frame,fg = "#003300",font="Times 20 bold")
      label7.place(x=450,y=280) 
      label8 = Label(frame,fg = "#003300",font="Times 20 bold")
      label8.place(x=450,y=310)
      
      #một chuỗi biến đổi để xử lý trước hình ảnh:
      #Compose tạo ra một loạt các biến đổi để chuẩn bị bộ dữ liệu.
      #Torchvision đọc các bộ dữ liệu vào PILImage (định dạng hình ảnh Python). 
      #ToTensor chuyển đổi PIL Image từ phạm vi [0, 255] thành FloatTensor
      #có hình dạng (C x H x W) với phạm vi [0.0, 1.0].
      #Sau đó, chúng tôi tái chuẩn hóa đầu vào thành [-1, 1] dựa trên công thức sau với
      #         μ=standard deviation=0.5.
      # input=(input−μ)/standard deviation
      # input=(input−0.5)/0.5
      transform = transforms.Compose(
          [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
      size_of_batch = 4
      if (entry2.get() != ""):
          size_of_batch = int(entry2.get())
    
      test_data = torchvision.datasets.ImageFolder(root='./data/test',
                                                   transform=transform)
      test_data_loader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=size_of_batch,
                                                     shuffle=True,num_workers=0)
       
      #bước đầu tiên. Hãy để chúng tôi hiển thị một hình ảnh từ bộ thử nghiệm để làm quen.
      datatest = iter(test_data_loader)
      imagetest, label = datatest.next()
    
      classes =["frog", "horse", "ship", "truck"]
      
      print(len(test_data_loader))
      print(label)  
      print('Example Train: ', ' '.join('%10s' % label[j] for j in range(size_of_batch)))
      
#      t = '=> Số lượng batch của tập test: %d' % (len(test_data_loader))  
#      label9.config(text = t)
      
      #Tiếp theo, hãy để tải lại trong mô hình đã lưu của chúng tôi 
      #(lưu ý: lưu và tải lại mô hình đã không cần thiết ở đây, 
      #chúng tôi chỉ làm điều đó để minh họa cách thực hiện):
      global net
      ftypes = [('PT files', '*.pt'), ('All files', '*')]
      dialog = Open(filetypes = ftypes)
      fl = dialog.show()
      if fl != '':
          net = net = torch.load(fl)
      else:
          net = torch.load('./model/net.pt')
      
      label12.config(text = "Load thành công!")
      #hiển thị Tạo một lưới các hình ảnh
      imshow(torchvision.utils.make_grid(imagetest))
      t = 'Tập Test:    ' + ' '.join('%s,' % classes[label[j]] for j in range(size_of_batch))
      label6.config(text = t) 
      
      #bây giờ chúng ta hãy xem mạng lưới thần kinh nghĩ những ví dụ trên là gì:
      outputs = net(imagetest)
      
      #Các đầu ra là năng lượng cho 4 lớp. Năng lượng cho một lớp càng cao, 
      #mạng càng nghĩ rằng hình ảnh thuộc về một lớp cụ thể. 
      #Vì vậy, hãy để Lôi lấy chỉ số năng lượng cao nhất:
      _, predicted = torch.max(outputs, 1)
      t = 'Dự đoán: ' + ' '.join('%s,' % classes[predicted[j]] for j in range(size_of_batch))
      label7.config(text = t) 
    
       #Chúng ta hãy xem network thực hiện trên toàn bộ dữ liệu.
      correct = 0
      total = 0
      confusion_matrix = np.zeros([4,4], int)
      with torch.no_grad():
          for data in test_data_loader:
              images, labels = data
              outputs = net(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              for i, l in enumerate(labels):
                  confusion_matrix[l.item(), predicted[i].item()] += 1 
       
      model_accuracy = correct / total * 100
      print('Model accuracy on {0} test images: {1:.2f}%'.format(total, model_accuracy))
      
      print('{0:10s} - {1}'.format('Category','Accuracy'))
      for i, r in enumerate(confusion_matrix):
          print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
      
      t = 'Độ chính xác của mạng trên 4000 hình ảnh test: %d %%' % (100 * correct / total)
      label8.config(text = t)
      
      global frame2
      frame2 = Frame(frame,height=520,width=1050)
      frame2.place(x=450,y=350)
      x = -120
      y = 30
      for i in range(size_of_batch):
          x = x + 150
          if (x > 1100):
              y = y + 220
              x = -120 + 150
          showimagelabel(imagetest[i],classes[label[i]],x,y)
    
      root1 = Tk()
      root1.title("bieudo")
      #Hiển thị biểu đồ tổng quan
      fr7=tk.Frame(root1,bg="white", borderwidth=5, relief='sunken')
      fr7.grid(row=1, column=3, rowspan=3, padx=10)
      lb3=tk.Label(fr7, text='Ma Trận Trực Quan',bg="white", font='Times 14 bold')
      lb3.grid(row=0, column=0)
      fig, ax = plt.subplots(1,1,figsize=(8,6))
      ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
      plt.ylabel('Thực tế')
      plt.yticks(range(4), classes)
      plt.xlabel('Dự đoán')
      plt.xticks(range(4), classes)
      plt.show()
      canvas = FigureCanvasTkAgg(fig, master=fr7)
      canvas.get_tk_widget().grid(row=1,column=0, sticky=Tkconstants.NSEW, rowspan=2)
      canvas.draw()
      root1.geometry("640x500")
      root1.mainloop()

#Chức năng vẽ mô hình chính xác và mất mát
#def plot_model_history(model_history):
#    fig, axs = plt.subplots(1,2,figsize=(15,5))
#    # summarize history for accuracy
#    axs[0].plot(range(1,len(model_history.history['acc' ])+1),model_history.history['acc'])
#    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
#    axs[0].set_title('Model Accuracy')
#    axs[0].set_ylabel('Accuracy')
#    axs[0].set_xlabel('Epoch')
#    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
#    axs[0].legend(['train', 'val'], loc='best')
#    # summarize history for loss
#    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
#    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
#    axs[1].set_title('Model Loss')
#    axs[1].set_ylabel('Loss')
#    axs[1].set_xlabel('Epoch')
#    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
#    axs[1].legend(['train', 'val'], loc='best')
#    plt.show()

def showHuongdan():#button hướng dẫn
    root1 = Tk()
    
    root1.title("HƯỚNG DẪN SỬ DỤNG")
    frame1 = Frame(root1)
    frame1.pack(fill=BOTH, expand=1)
    
    str1="Bước 1: Click vào nút Load & Chuẩn Hóa để load Dataset huấn luyện,sau khi load thành công Dataset màn hình sẽ hiển thị:\n"
    str2="Số lớp trong Dataset, tên của các lớp, số mẫu trong mỗi tập và hiển thị hình ảnh minh họa cho mỗi tập\n"
    str3='Bước 2: Sau khi load Dataset thành công ở bước 1, click vào nút "Xác Định Mô Hình Mạng" để\n'
    str4='màn hình sẽ hiển thị mô hình mạng nơron\n'
    str5='Bước 3:click vào nút "Xác Định Hàm Mất Mát", màn hình sẽ hiển thị hàm mất mát và tối ưu\n'
    str6='Bước 4: Tại bước này, bạn có thể thay đổi một số thông số huấn luyện theo ý muốn.\n'
    str7='Sau khi kiểm tra click vào nút Train, nhập tên để lưu lại mô hình mạng sau khi huấn luyện,\n' 
    str8='sau đó quá trình huấn luyện sẽ bắt đầu\n'
    str9="Khi quá trình huấn luyện hoàn tất sẽ có thông báo đường dẫn chứa file lưu lại và ứng dụng sẽ hiển thị đồ thị huấn luyện\n"
    str12="Bước 5: Click vào nút Test model và chọn file mô hình mạng đã được lưu trước đó\n" 
    str11='sau khi test hoàn tất ứng dụng sẽ hiển thị kết quả kiểm tra'
   
    a = str1+str2+str3+str4+str5+str6+str7+str8+str9+str12+str11
    label01 = Label(frame,fg = "black",font= "Times 20 bold")
    label01 = Label(frame1,text=a,
                    fg = "black",
                    font= "Times 14 bold").pack()
    label01.place(x=10,y=30)
    button9 = Button(frame1,
                     text = "Thoát",
                     command=root1.destroy,
                     fg = "black",
                     width=20,height=1)
    button9.place(x=30,y=250)

    root1.geometry("700x250")
    root1.mainloop()

#==============================================================================    
def main():
    root = Tk()
    root.title("BTL MÔN MẠNG NORON-NHÓM 3-16DDS06031")
    global frame
    global entry
    global entry2
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=1)
    menubar = Menu(root)
    root.config(menu=menubar)

    fileMenu = Menu(menubar)
    menubar.add_cascade(label="File",
                        menu=fileMenu)
    menubar.add_cascade(label="Help",
                        command=showHuongdan)
    fileMenu.add_command(label="Close", 
                         command=root.destroy)
    
    #tên đề tài, GVHD, thành viên nhóm
    lbl = Label(root, text ="Môn học: MẠNG NƠRON",
                font = ("Times New Roman Bold",28),
                fg="red",bg='Moccasin')
    lbl.pack(expand=True)
    lbl.place(x=450,y=10)    
    label1 = Label(frame,text = "Đề Tài: Ứng dụng mạng nơron cho bài toán phân lớp frog, horse, ship, truck",
                   fg = "blue",bg='Moccasin',
                   font = ("Times New Roman Bold",28))
    label1.place(x=50,y=50)
    label2 = Label(frame,text = "Giáo Viên Hướng Dẫn: Ngô Thanh Tú ",
                   font="Times 20 bold")
    label2.place(x=160,y=100)
    label3 = Label(frame,text = "Thành viên nhóm 3: Đỗ Tống Quốc, Tôn Nữ Nguyên Hậu, Nguyễn Lê Xuân Phước ",
                   font="Times 20 bold")
    label3.place(x=160,y=140)
    
     #Add thông số
    label4 = Label(frame,text = "Epoch",
                   fg = "red",
                   font="Times 14 bold")
    label4.place(x=10,y=200)
    label4a = Label(frame,text = "(Mặc định: 1)",
                    fg = "black",
                    font="Times 14 bold")
    label4a.place(x=150,y=250)
    label5 = Label(frame,text = "Size of batch",
                   fg = "red",
                   font="Times 14 bold")
    label5.place(x=10,y=300)
    label5a = Label(frame,text = "(Mặc định: 4)",
                    fg = "black",
                    font="Times 14 bold")
    label5a.place(x=150,y=350)
    
    #entry.get()
    entry = Entry(frame,
                  font="Times 14 bold")
    entry.place(x=150,y=200)
    entry2 = Entry(frame,
                   font="Times 14 bold")
    entry2.place(x=150,y=300)
    
    #Buton theo từng bước
    button0 = Button(frame,width=20,text = "Load & Chuẩn Hóa",
                     fg = "Dark Cyan",
                     font="Times 14 bold",
                     bg='LightYellow2',
                     command=load)
    button0.place(x=110,y=400)
    button1 = Button(frame,width=20,text = "Xác Định Mô Hình Mạng",
                     fg = "Dark Cyan",
                     font="Times 14 bold",
                     bg='LightYellow2',
                     command=modul_net)
    button1.place(x=110,y=450)
    button2 = Button(frame,width=20,text = "Xác Định Hàm Mất Mát",
                     fg = "Dark Cyan",
                     font="Times 14 bold",
                     bg='LightYellow2',
                     command=loss_function)
    button2.place(x=110,y=500)
    button3 = Button(frame,width=20,
                     text = "Train The Network",
                     fg = "Dark Cyan",
                     font="Times 14 bold",
                     bg='LightYellow2',
                     command=training)
    button3.place(x=110,y=550)
    button3 = Button(frame,width=20,
                     text = "Test The Network",
                     fg = "Dark Cyan",
                     font="Times 14 bold",
                     bg='LightYellow2',
                     command=test_net)
    button3.place(x=110,y=600)
    
    button4 = Button(frame,width=20,
                     text = "Save",
                     fg = "Dark Cyan",
                     font="Times 14 bold",
                     bg='LightYellow2',
                     command=save)
    button4.place(x=110,y=650)
    
    
    lbl1 = Label(frame,text = "Bước 1: ",
                 fg = "Purple",
                 font="Times 16 bold")
    lbl1.place(x=20,y=400)
    lbl1 = Label(frame,text = "Bước 2: ",
                 fg = "Purple",
                 font="Times 16 bold")
    lbl1.place(x=20,y=450)
    lbl1 = Label(frame,text = "Bước 3: ",
                 fg = "Purple",
                 font="Times 16 bold")
    lbl1.place(x=20,y=500)
    lbl1 = Label(frame,text = "Bước 4: ",
                 fg = "Purple",
                 font="Times 16 bold")
    lbl1.place(x=20,y=550)
    lbl1 = Label(frame,text = "Bước 5: ",
                 fg = "Purple",
                 font="Times 16 bold")
    lbl1.place(x=20,y=600)
    
    global label12,label6,label7,label8,label9,label11,label10,label13
    label12 = Label(frame,fg = "Red",font="Times 24 bold")
    label12.place(x=650,y=200)
       
    
    label9 = Label(frame,fg = "black",
                   font= "Times 20 bold")
    label9.place(x=450,y=300)    
    label10 = Label(frame,fg = "black",
                    font= "Times 20 bold")
    label10.place(x=450,y=250)
    label13 = Label(frame,fg = "black",
                    font= "Times 20 bold")
    label13.place(x=450,y=350)
    
    
    root.geometry("1366x768")
    root.mainloop()
    
if __name__=="__main__":
    main()    
