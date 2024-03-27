### Downloading and Pre-processing the data
[Data](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view)
```
mkdir data
cd ./data
tar -xvzf LAMOL.tar.gz
cd ../src
python preprocess.py
```
### Training RnD model (Full options)
```

python runRnD.py --tasks ag yelp amazon yahoo dbpedia --epochs 1 1 1 1 1 --batch_size 2 --disen True --reg True --reggen 0.5 --regspe 0.5 --clus gmm --n-labeled -1 --n-val 500
```
### Training Fine-tune model 
```

# Example for length-3 task sequence
python runFinetune.py --tasks ag yelp yahoo --epochs 4 3 2 --batch_size 2 

# Example for length-5 task sequence
python runFinetune.py --tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1 --batch_size 2
```
### Training Replay model
```
# Example for length-3 task sequence
python runReplay.py --tasks ag yelp yahoo --epochs 4 3 2 --batch_size 2  

# Example for length-5 task sequence
python runReplay.py --tasks ag yelp amazon yahoo dbpedia --epochs 4 3 3 2 1 --batch_size 2
```
 

