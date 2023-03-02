python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 30 --dp_clip 10 --gpu 3
python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 20 --dp_clip 10 --gpu 1
python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 10 --dp_clip 10 --gpu 1
python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 5 --dp_clip 10 --gpu 1
python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 1 --dp_clip 10 --gpu 3
python3 -u main.py --dataset mnist --dp_mechanism no_dp --gpu 3 > running/6.log &


python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 30 --dp_clip 10 --gpu 3 --dp_sample 0.01
python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 20 --dp_clip 10 --gpu 1 --dp_sample 0.01
python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 10 --dp_clip 10 --gpu 1 --dp_sample 0.01
python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 5 --dp_clip 10 --gpu 1 --dp_sample 0.01
python3 -u main.py --dataset mnist --dp_mechanism MA --dp_epsilon 1 --dp_clip 10 --gpu 3 --dp_sample 0.01
#nohup python3 -u main.py --dataset mnist --dp_mechanism Gaussian --dp_epsilon 100000 --dp_clip 10 --gpu 3 > running/3.log &
#nohup python3 main.py --dataset femnist --epsilon 20 --clip 10 > running/femnist_20.log &