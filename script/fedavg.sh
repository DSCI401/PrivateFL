cd .. &&
python FedAverage.py --data='imdb' --nclient=100 --nclass=2 --ncpc=2 --model='SentimentClassifier' --mode='LDP' --round=60 --epsilon=2 --sr=1 --lr=5e-3 --flr=1e-2 --physical_bs=64 --E=1
