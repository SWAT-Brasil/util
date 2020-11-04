import pytest
import os
import subprocess

here = os.path.abspath(os.path.dirname(__file__))
# Caminho para executavel
EXEC_PATH = os.path.realpath(os.path.join(here, '../pcpswat.py'))


def test_get_netcdf_to_db():
    cmd = ['python', EXEC_PATH, os.path.realpath('./example_1/pcp/pcp.txt'), os.path.realpath('./out/'),
           '-interpolate', os.path.realpath("./example_1/pcp/pcp_to_interpolate.txt"), "-method", 'nearest']
    cmd = ['python', EXEC_PATH, os.path.realpath('./example_samantha/pcp/pcp.txt'), os.path.realpath('./out/'),
           '-interpolate', os.path.realpath("./example_samantha/pcp/pcp_to_interpolate.txt"), "-method", 'nearest']


    print('\nComando teste -> ' + ' '.join(cmd))
    output = subprocess.run(cmd)
    print(output.stdout)
    if output.returncode != 0:
        # Processo retornou erro
        assert False
    else:
        assert True
