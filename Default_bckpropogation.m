%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Program to predict goldprices %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear all variables in workspace and close all figures
clear all;
close all hidden;
clc;

% Gold data prices into workspace

a=xlsread('gold price');
b=xlsread('gold date');

plot(a,'b.-')
set(gca,'XTick',[0 200 360 500 716])    % Setting the x-axis labels
set(gca,'XTickLabel',{'1/2/2007','10/15/2007','06/04/2008','12/18/2008','10/29/2009'})
grid;
title('Gold Prices in USD (all data)');
ylabel('USD per troy ounce');
xlabel('Date (Month/Day/Year)');


% Separating training data and test data

traindata=a(1:360);
testdata=a(361:716);

%Pre-processing data

[Inputn,minp,maxp,Targetn,mint,maxt]=premnmx(traindata,testdata);

% Estimation 
% Forming U and z
ntrain=length(Inputn);
numfuturedays=1;       % Number of future days to predict
numpastdays=80;         % Number of past days to use for reference
maxiter=numpastdays+numfuturedays;

% Creating input and output matrix
for i=1:1:ntrain-maxiter-1
    tempU=[];
    for jj=1:1:numpastdays
        tempU=[Inputn(i+jj-1) tempU];
    end
    U2(i,:)=tempU;
    tempZ=[];
    for kk=1:1:numfuturedays     
        tempZ=[Inputn(i+numpastdays+kk-1) tempZ];
    end
    z2(i,:)=tempZ;
end


%Creating a feedforward network
net=newff(U2',z2',2); 

%%Train the network
%net.trainParam.lr = 0.01;

net.trainParam.epochs = 500;

net = train(net,U2',z2');

% Testing on training data
figure(2);hold on; plot(z2,'r-d'); 
figure(2);plot(Inputn,'b.-')
set(gca,'XTick',[0 100 200 360])
set(gca,'XTickLabel',{'1/2/2007','05/24/2007','10/15/2007','06/04/2008'})
grid;
ylabel('USD per troy ounce');
xlabel('Date (Month/Day/Year)');
ss=cat(2,'Training data and estimation,',num2str(numfuturedays),' days in the future.');
title(ss);


% Testing on test data
 tempU=[];
    for jj=1:1:numpastdays
        tempU=[tempU traindata(ntrain-jj+1)];
    end
    U3(1,:)=tempU;

ypredict = sim(net,U3');  % using training data to predict future data

ypredictnorm=postmnmx(ypredict,minp,maxp); %post-processing 

figure(3); plot(testdata,'b.-');
set(gca,'XTick',[0 100 200 356]);
set(gca,'XTickLabel',{'06/05/2008','10/24/2008','3/19/2009','10/29/2009'});
grid;
title('Testdata');
hold on;
plot(ypredictnorm,'rx-')
ylabel('USD per troy ounce');
xlabel('Date (Month/Day/Year)');
legend('Actual','Backpropagation');

%Error
d=[z2-ypredict].^2;
mse=mean(d)