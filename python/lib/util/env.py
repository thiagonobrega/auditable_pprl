import os

def getbase_dir(subpath):
    try:
        os.environ['WINDIR']
        # os.path.join('path','paths')
        base_dir = 'D:\\Dados\\OneDrive\\Doutorado\\workspace\\bc-playground\\'+subpath+'\\'
    except KeyError:
        base_dir = '/home/thiagonobrega/workspace/bc-playground/'+subpath+'/'
    return base_dir