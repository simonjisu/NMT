import argparse

def build_config_file(run_path='./runtrain.sh', config_path='./settings.py', notebook=False):
    with open(run_path, 'r') as file:
        data = file.read().splitlines()
    if notebook:
        log_file = data[-1].split('>')[1].strip()[1:-2]
    else:
        log_file = data[-1].split('>')[1].strip()[:-2]
    with open(log_file, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()[0][10:-1]
        data = [x.strip() for x in data.split(',')]
        for i, x in enumerate(data):
            if ('SAVE' in x) & ('_PATH' in x):
                data[i] = x.split("'.")[0] +"'./model" + x.split("'.")[1]
            elif 'ROOTPATH' in x:
                data[i] = "ROOTPATH='./data/'"
            elif 'RETURN_W' in x:
                data[i] = "RETURN_W=True"

        with open(config_path, 'w', encoding='utf-8') as f:
            for x in data:
                print(x, file=f)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='build config parser')
    parser.add_argument('-r', '--RUN_PATH', help='run path', type=str, default='./runtrain.sh')
    parser.add_argument('-c', '--CONFIG_PATH', help='config path', type=str, default='./settings.py')
    parser.add_argument('-n', '--NOTEBOOK', help='whether try this in notebook', action='store_true')
    config = parser.parse_args()
    build_config_file(run_path=config.RUN_PATH, config_path=config.CONFIG_PATH, notebook=config.NOTEBOOK)
    print('done!')