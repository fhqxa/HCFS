clear;clc;close all;
fontsize=16;
figure;
x=1:1:7;%x���ϵ����ݣ���һ��ֵ�������ݿ�ʼ���ڶ���ֵ��������������ֵ������ֹ
FH1=[0.8915 0.8926	0.8772	0.8507	0.8452	0.8491	0.8133]; %a����yֵ
plot(x,FH1,'-*','LineWidth',1); %���ԣ���ɫ�����
axis([1,7,min(FH1)-0.1,min(FH1)+0.2])  %ȷ��x����y���ͼ��С
set(gca,'XTick',[1:1:7],fontsize) %x�᷶Χ1-6�����1
set(gca,'YTick',[min(FH1)-0.1:0.1:min(FH1)+0.2]) %y�᷶Χ0-700�����100
legend('Neo4j','MongoDB');   %���ϽǱ�ע
xlabel('���')  %x����������
ylabel('ʱ�䣨ms��') %y����������