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
# 3 tasks
python runRnD.py --env RnD_3step

# 5 tasks
python runRnD.py --env RnD_5step
```
### Training Fine-tune model 
```
# 3 tasks
python runFinetune.py --env Finetune_3step

# 5 task
python runFinetune.py --env Finetune_5step
```
### Training Replay model
```
# 3 tasks
python runFinetune.py --env Replay_3step

# 5 task
python runFinetune.py --env Replay_5step
```
 

