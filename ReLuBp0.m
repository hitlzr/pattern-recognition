trainima=cell2mat(struct2cell(load('numdata.mat','trainsample')));
trainlab=cell2mat(struct2cell(load('numdata.mat','trainlabel')));
% train_ima=cell2mat(struct2cell(load('class-100-1.mat','data')));
% train_lab=cell2mat(struct2cell(load('class-100-1.mat','label')));
train_ima=trainima(1:15000,:);
train_lab=trainlab(1:15000,:);
test_ima=trainima(15001:20000,:);
test_lab=trainlab(15001:20000,:);
% test_ima=cell2mat(struct2cell(load('numdata.mat','testsample')));
% test_lab=cell2mat(struct2cell(load('numdata.mat','testlabel')));
max01=max(max(train_ima));
min01=min(min(train_ima));
[row1,col1]=size(train_ima);
% train_ima=(train_ima-min01*ones(row1,col1))/(max01-min01);%%数值归一化
train_ima=(train_ima-((min01+max01)/2)*ones(row1,col1))/((max01-min01)/2);

max02=max(max(test_ima));
min02=min(min(test_ima));
[row2,col2]=size(test_ima);
% test_ima=(test_ima-min02*ones(row2,col2))/(max02-min02);%%数值归一化
test_ima=(test_ima-((min02+max02)/2)*ones(row2,col2))/((max02-min02)/2);
sam_sum =15000;%%训练样本数
test_sum=5000;%%测试样本数
input = 79;%输入维数
output = 10;%输出维数
hid = 400;%隐含层数
w1=cell2mat(struct2cell(load('class-400-3.mat','w1')));
w2=cell2mat(struct2cell(load('class-400-3.mat','w2')));
bias1=cell2mat(struct2cell(load('class-400-3.mat','bias1')));
bias2=cell2mat(struct2cell(load('class-400-3.mat','bias2')));
% w1=cell2mat(struct2cell(load('class-40-7.92.mat','w1')));
% w2=cell2mat(struct2cell(load('class-40-7.92.mat','w2')));
% bias1=cell2mat(struct2cell(load('class-40-7.92.mat','bias1')));
% bias2=cell2mat(struct2cell(load('class-40-7.92.mat','bias2')));
% w1 = randn([input,hid])*sqrt(2/79);
% w2 = randn([hid,output])*sqrt(2/79);
% bias1 = zeros(hid,1);
% bias2 = zeros(output,1);
rate1 = 0.005;
rate2 = 0.005; 

net1=zeros(hid,1);
y=zeros(hid,1);
net2=zeros(output,1);
z=zeros(output,1);
Error=[];
Sumerror=[];
p=0.5;
for num=1:3000
    sumerror=0;
    for i=1:sam_sum
        label = zeros(10,1);
        label(train_lab(i)+1,1) = 1;
        net1=w1'*train_ima(i,:)'+bias1;
        y=ReLu(net1);
%         r=randsrc(hid,1,[0,1;p,(1-p)]);
%         y=y.*r;
%         y=y/p;
%         y=sigmoid(net1);
        net2=w2'*y+bias2;
%         z=sigmoid(net2);
        z=ReLu(net2);
        error=label-z;
%         delta2=error.*z.*(1-z);
%         delta1=y.*(1-y).*(delta2'*w2')';
        delta2=error.*drelu(z);
        delta1=drelu(y).*(delta2'*w2')';
        for j=1:output
            w2(:,j)=w2(:,j)+rate2*(delta2(j)*y')';
        end
        for j=1:hid
            w1(:,j)=w1(:,j)+rate1*(delta1(j)'*train_ima(i,:))';
        end
        bias2 = bias2 + rate2*delta2;
        bias1 = bias1 + rate1*delta1; 
        
%         pp=label-train_lab(i,:);
%         sumerror=sumerror+norm(pp);
    end
%     sumerror
%     Sumerror=[Sumerror sumerror];
%     
    for i=1:test_sum
        net1=w1'*test_ima(i,:)'+bias1;
        y=ReLu(net1);
        net2=w2'*y+bias2;
        z0(i,:)=ReLu(net2);
    end
    
    label00=[];
    for i=1:test_sum
        [K,Index]=sort(z0(i,:));
        label00(i,1)=Index(10)-1;
    end
    
    M=[];
    sum=0;
    for i=1:test_sum
        if test_lab(i)~=label00(i)
            sum=sum+1;
%             M=[M;i];
        end
    end
    errorrate=sum/5000
    
    Error=[Error errorrate];
    
    
    if errorrate>0.2
        rate1=0.05;
        rate2=0.05;
    else if errorrate>0.1 
            rate2=0.02;
            rate1=0.02;
        else if errorrate>0.07
                rate2=0.01;
                rate1=0.01;
            else
                rate2=0.005;
                rate1=0.005;
            end
        end
    end
end


%     for i=1:sam_sum
%         net1=w1'*train_ima(i,:)'+bias1;
%         y=ReLu(net1);
%         net2=w2'*y+bias2;
%         z1(i,:)=ReLu(net2);
%     end
%     label01=[];
%     for i=1:sam_sum
%         [K,Index]=sort(z1(i,:));
%         label01(i,1)=Index(10)-1;
%     end
%     sum0=0;
%     for i=1:sam_sum
%         if train_lab(i)~=label01(i)
%             sum0=sum0+1;
% %             M=[M;i];
%         end
%     end