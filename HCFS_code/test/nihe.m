clear

clf

x=1810:10:1900;

y=[74.875 92.552 107.231 120.153 130.879 152.427 180.383 202.352 227.485 250.597];

plot(x,y,'s','markersize',3)

grid on

%��ͼ���۲���ɢ���ݵ�����

p=polyfit(x,y,1);

%��1�ζ���ʽ�������

f = polyval(p,x);

hold on

plot(x,f,'r');

xlabel('���')

ylabel('�˿�')

title('�������')

%��ͬһ�������ڻ���������ߺ�ԭ����ɢ����



%��ʾ��϶���ʽϵ��

p1865=polyval(p,[1865])

%����1865����˿�����