# pcpswat.py 

Rotinas para interpolação de precipitação para format de dados do SWAT. 

Para maior facilidade de uso utilize o pcpswat_colab.ipynb, usando o link no arquivo. Este arquivo pode
ser utilizado de forma online no Colab - Google, e é mais facil de utilizar pois já instala todas as 
depedencias automaticamente.

Para informações mais detalhadas de uso olhar o jupyter notebook 'tutorial pcpswat.ipynb'

## Formato arquivo de saída
Existe diferença entre arquivos de texto no modo unix e windows, que podem gerar problemas na leitura dos dados, 
especialmetne no Windows - Notepad. O formato de saida dos arquivos foi ajustada para o formato Windows,
ou seja, com fim de linha '\r\n'. Caso tenha algum tipo de problema utilize o notepad++, ele mostra em 
qual formato o arquivo esta sendo interpretado, e oferece informações sobre qual pode ser o problema.

No linux é possível verificar o tipo do arquivo utilizando o comando ```file <nome do arquivo>```. 
No formato unix a saída é ASCII text, em windows a saida é ASCII text, with CRLF line teminator.
