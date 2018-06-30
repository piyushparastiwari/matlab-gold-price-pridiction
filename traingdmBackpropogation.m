%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Program to predict goldprices %%%%%%%%%%%%%%%%
%By Using Gradient Descent With Momentum Backpropogation%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear all variables in workspace and close all figures
clear all;
close all hidden;
clc;

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

figure(2); plot(Inputn,'b.-')
set(gca,'XTick',[0 100 200 360])    % Setting the x-axis labels
set(gca,'XTickLabel',{'2 Jan 2007','24 May 2007','15 Oct 007','04 June 2008'})
grid;
title('Gold Prices in USD (train data)');
ylabel('USD per troy ounce');
xlabel('Date (Month/Day/Year)');

figure(3); plot(Targetn,'b.-')
set(gca,'XTick',[0 100 200 356])    % Setting the x-axis labels
set(gca,'XTickLabel',{'05 June 2008','24 Oct 2008','19 March 2009','29 Oct 2009'})
grid;
title('Gold Prices in USD (test data)');
hold on;
plot(Targetn,'g+-')
ylabel('USD per troy ounce');
xlabel('Date (Month/Day/Year)');

% Estimation 
% Forming U and z
ntrain=length(Inputn);
numfuturedays=1;         % Number of future days to predict
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
net=newff(minmax(U2'),[3 10],{'tansig' 'purelin'},'traingdm');  %Using Gradient Descent With Momentum Backpropogation


%%Train the network

net.trainParam.epochs =1000;

net = train(net,U2',z2');


% Testing on training data
figure(5);hold on; plot(z2,'r-d'); 
figure(5);plot(Inputn,'b.-')
set(gca,'XTick',[0 100 200 360])
set(gca,'XTickLabel',{'1/04/2007','05/24/2007','10/15/2007','06/04/2008'})
grid;
ylabel('USD per troy ounce');
xlabel('Date (Month/Day/Year)');
ss=cat(2,'Training data and estimation,',num2str(numfuturedays),' days in the future.');
title(ss);


% Testing on test data
 tempU=[];
    for jj=1:1:numpastdays
        tempU=[tempU Inputn(ntrain-jj+1)];
    end
    U3(1,:)=tempU;

ypredict = sim(net,U3');  % using training data to predict future data

ypredictnorm=postmnmx(ypredict,minp,maxp); %post-processing 

figure(6); plot(testdata,'b.-');
set(gca,'XTick',[0 100 200 356]);
set(gca,'XTickLabel',{'06/05/2008','10/24/2008','3/19/2009','10/29/2009'});
grid;
title('Testdata');
hold on;
plot(ypredictnorm,'r*-')
ylabel('USD per troy ounce');
xlabel('Date (Month/Day/Year)');
legend('Actual','Backpropagation');

%error
d=[z2-ypredict].^2;
mse=mean(d)

