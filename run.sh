nohup python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 30 --dp_clip 10 --gpu 3 > running/1.log &
nohup python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 20 --dp_clip 10 --gpu 1 > running/2.log &
nohup python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 10 --dp_clip 10 --gpu 1 > running/3.log &
nohup python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 5 --dp_clip 10 --gpu 1 > running/4.log &
nohup python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 1 --dp_clip 10 --gpu 3 > running/5.log &
nohup python3 -u main.py --dataset mnist --dp_mechanism no_dp --gpu 3 > running/6.log &
#nohup python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 100000 --dp_clip 10 --gpu 3 > running/3.log &
#nohup python3 main.py --dataset femnist --epsilon 20 --clip 10 > running/femnist_20.log &