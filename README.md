### File Structure:
+ Within here is are two files for the CNN-LSTM model that include the script to download all the data as well as the model itself. In addition, we have the code for the GNN in its own gnn folder minus the data. The data comes from NASA's website which holds geospatial simulations of particulate matter behavior. For this project, we processed and stored all of our data on USC's CARC cluster. 

### README:
CNN-LSTM
+ To run this code locally, first run async_dl_preprocess.py to collect the relevant data, and then run python3 cnnlstm_final.py in terminal.
	
### To run on CARC: 
+ First set up an interactive allocation with the command: salloc --partition=gpu --gres=gpu:v100:1 --ntasks=1 --cpus-per-task=4 --mem=184G --time=02:00:00
+ Then navigate with cd to the directory that contains all the relevant files:/project/dilkina_565/
+ Enter the venv with the command source venv/bin/activate.
+ Now run python3 narrow3.py 


### PM2.5-GNN
+ Go to GPU (If you're doing low epochs, you prob could skip and just use CPU)
+ salloc --partition=gpu --gres=gpu:v100:1 --ntasks=1 --cpus-per-task=4 --mem=184G --time=02:00:00

+ cd /project/dilkina_565/PM2.5-GNN && source venv/bin/activate

- module purge
- module load gcc/11.3.0
- module load python/3.9.12
- module load cuda/11.8.0
- module load cudnn/8.4.0.27-11.6
- module load git/2.36.1

Good to go!

+ Run code with:
python train.py
