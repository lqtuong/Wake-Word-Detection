import os
import shutil
import urllib.request
import zipfile
import glob

def download_dataset(name):
    """Download the dataset into current working directory."""
    if not os.path.exists(name):
        os.mkdir(name)
        urllib.request.urlretrieve('https://github.com/karoldvl/{0}/archive/master.zip'.format(name), '{0}/{0}.zip'.format(name))

        with zipfile.ZipFile('{0}/{0}.zip'.format(name)) as package:
            package.extractall('{0}/'.format(name))

        os.unlink('{0}/{0}.zip'.format(name))

        for src in glob.glob('{0}/{0}-master/*'.format(name)):
            shutil.move(src, '{0}/{1}'.format(name, os.path.basename(src)))

        os.rmdir('{0}/{0}-master'.format(name))

download_dataset('ESC-10')
download_dataset('ESC-50')