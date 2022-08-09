# move towards data folder first
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
mkdir raw
mv 'cornell_movie_dialogs_corpus.zip' 'raw'
unzip 'raw/cornell_movie_dialogs_corpus.zip'
!mv "/content/deep-learning/NLP/implement/temp/data/cornell movie-dialogs corpus" raw