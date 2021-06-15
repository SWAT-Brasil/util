import pytest
import os
import subprocess

here = os.path.abspath(os.path.dirname(__file__))
# Caminho para executavel
EXEC_PATH = os.path.realpath(os.path.join(here, '../pcpswat.py'))
EXAMPLE_DIR = 'example_1'
EXAMPLE_DIR = 'example_2'
#EXAMPLE_DIR = 'example_error_empty_line'
#EXAMPLE_DIR = 'example_error_malformed'



def test_interpolation():
    cmd = ['python', EXEC_PATH, os.path.realpath('./' + EXAMPLE_DIR + '/pcp/pcp.txt'), os.path.realpath('./' + EXAMPLE_DIR + '/out/'),
           '-interpolate', os.path.realpath('./' + EXAMPLE_DIR + '/pcp/pcp_to_interpolate.txt'), '-method', 'idw']

    print('\nComando teste -> ' + ' '.join(cmd))
    output = subprocess.run(cmd)
    print(output.stdout)
    if output.returncode != 0:
        # Processo retornou erro
        assert False
    else:
        assert True
