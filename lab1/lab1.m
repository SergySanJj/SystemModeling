clc;
clear;
data = dlmread("f11.txt", " ");
[p,N] = size(data);
T = 5;
dt = 0.01;
t = 0:dt:(T);
saveas(plot(t,data),"images/startData.png");

frequency = zeros(1,N);

for point_id = 1:N
    sin_freq = 0;
    cos_freq = 0;

    for signal = 1:N
        sin_freq = sin_freq + data(signal) * sin(2 * pi * point_id * signal / N);
        cos_freq = cos_freq + data(signal) * cos(2 * pi * point_id * signal / N);
    end
    
    sin_freq = sin_freq / N;
    cos_freq = cos_freq / N;

    frequency(point_id) = sqrt(sin_freq *sin_freq + cos_freq * cos_freq);
end
saveas(plot(t,frequency),"images/fur.png");


biggest_value = zeros(1,0);
for i = 5: N / 2
    maxim = frequency(i-3);
    for sm = (i-3):(i+3)
        if (frequency(sm)>maxim)
            maxim = frequency(sm);
        end
    end
    
    if maxim == frequency(i)
        biggest_value = [biggest_value, i];
        disp(frequency(i));
    end
end
main_frequency = frequency(biggest_value(1))/T;
disp(main_frequency);

b = [sum(data .* (t .^ 3)), sum(data .* (t.^2)), sum(data .* t), sum(data .* sin(2 * pi * main_frequency * t)), sum(data)];

a = zeros(5,5);
functions = [t .^ 3; t .^ 2; t; sin(2 * pi * main_frequency * t); ones(1,N)];

for i = 1:5
    for j = 1:5
        %a(i, j) = sum(functions(i) .* functions(j));
        for k = 1:N
            a(i,j) = a(i,j) + functions(i,k)*functions(j,k);
        end
    end
end  

solution = a\b';

appr = functions .* solution;
cc = zeros(1,N);
for i=1:N
    for j =1:5
        cc(1,i) = cc(1,i) + appr(j,i);
    end
end

saveas(plot(t,cc),"images/appr.png");

immse(cc,data)

