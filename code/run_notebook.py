from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
path=root/'code/Vaidik_Maheshwari_Anant_Gupta_DA2141.ipynb'
nb=nbformat.read(path,as_version=4)


def save(*args,**kwargs):
    nbformat.write(nb,path)


def progress(cell_index,**kwargs):
    print('Executing cell',cell_index,flush=True)


client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root/'code')}},on_cell_complete=save,on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('NOTEBOOK EXECUTED',flush=True)
