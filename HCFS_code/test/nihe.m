clear

clf

x=1810:10:1900;

y=[74.875 92.552 107.231 120.153 130.879 152.427 180.383 202.352 227.485 250.597];

plot(x,y,'s','markersize',3)

grid on

%画图并观察离散数据的特性

p=polyfit(x,y,1);

%用1次多项式进行拟合

f = polyval(p,x);

hold on

plot(x,f,'r');

xlabel('年份')

ylabel('人口')

title('拟合曲线')

%在同一个坐标内画出拟合曲线和原有离散数据



%显示拟合多项式系数

p1865=polyval(p,[1865])

%估计1865年的人口数量