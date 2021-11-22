import matplotlib.pyplot as plt
import argparse
import os
import shutil

def get_file(dir):
    for name in next(os.walk(dir))[2]:
        if name.endswith('.txt'):
            print(os.path.join(dir, name))
            return os.path.join(dir, name)


# first list are names to be put in legend, second list are paths to experiments
filenames = [
    [''],
    ['retr_best']
]

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str)

args = parser.parse_args()
dest_dir = './result' if args.result_dir is None else args.result_dir

def main(filenames=filenames, dest_dir=dest_dir):
    mae_fig = plt.figure()
    subfig_mae = mae_fig.add_subplot(111)
    cs_fig = plt.figure()
    subfig_cs = subfig = cs_fig.add_subplot(111)

    for name, filename in zip(filenames[0], filenames[1]):
        max = 0
        ep = []

        acc = []
        mae = []
        cs = []
        with open(get_file(filename), mode='r') as f:
            for line in f:
                if line[:6] == 'Epochs':
                    max = int(line.split(':')[-1])
                elif line[:5] == 'Epoch':
                    n = int(line[6:].split('/')[0])
                    ep.append(n)
                elif line[:10] == 'VALIDATION':
                    line = line.split('|')[1:]
                    acc.append(round(float(line[0].split(':')[-1]), ndigits=3))
                    mae.append(round(float(line[1].split(':')[-1]), ndigits=3))
                    cs.append(round(float(line[2].split(':')[-1]), ndigits=3))

        
        if name == 'classification':
            # if classification then plots accuracy and CS[1]
            subfig_mae.plot(ep, acc, label=name)
            subfig_mae.set_xlabel('Epochs')
            subfig_mae.set_ylabel('acc')
            subfig_mae.legend()
            
            subfig_cs.plot(ep, cs, label=name)
            subfig_cs.set_xlabel('Epochs')
            subfig_cs.set_ylabel('CS[1]')
            subfig_cs.legend()

        else:
            # if regression plots MAE and CS[5]
            subfig_mae.set_ylim(bottom=2, top=15) 
            subfig_mae.plot(ep, mae, label=name, color='red')
            subfig_mae.set_xlabel('Epochs')
            subfig_mae.set_ylabel('MAE')
            subfig_mae.legend()
            
            subfig_cs.plot(ep, cs, label=name)
            subfig_cs.set_xlabel('Epochs')
            subfig_cs.set_ylabel('CS[5]')
            subfig_cs.legend()


    try:
        os.mkdir(dest_dir)
    except FileExistsError:
        txt = None
        txt = input("Warning: overwrite result directory? ")
        while txt not in ['y', 'n']:
            txt = input("Can only accept y/n as response; retry: ")
        if txt == 'y':
            shutil.rmtree(dest_dir)
            os.mkdir(dest_dir)
        elif txt == 'n':
            exit(1)

    mae_fig.savefig(dest_dir + '/mae.jpg')
    cs_fig.savefig(dest_dir + '/cs.jpg')

if __name__ == '__main__':
    main()