clear;clc;close all;
fontsize=16;
figure;
x=1:1:7;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
FH1=[0.8915 0.8926	0.8772	0.8507	0.8452	0.8491	0.8133]; %a数据y值
plot(x,FH1,'-*','LineWidth',1); %线性，颜色，标记
axis([1,7,min(FH1)-0.1,min(FH1)+0.2])  %确定x轴与y轴框图大小
set(gca,'XTick',[1:1:7],fontsize) %x轴范围1-6，间隔1
set(gca,'YTick',[min(FH1)-0.1:0.1:min(FH1)+0.2]) %y轴范围0-700，间隔100
legend('Neo4j','MongoDB');   %右上角标注
xlabel('深度')  %x轴坐标描述
ylabel('时间（ms）') %y轴坐标描述