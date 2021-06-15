import pandas as pd
import os
import datetime
import argparse
import math
import numpy as np
import sys


# Valor de NO_DATA utilizado pelo SWAT
NO_DATA = -99.0
DELIMITER = ','
ENCODING = 'utf-8'
DECIMAL = '.'
# Versao major.minor.patch
# major: incompatible changes
# minor: adiciona funcoes compativeis com versoes anteriores
# patch: correcao de bugs que mantem compatibilidade
__version_info__ = (0, 1, 0)
__version__ = '.'.join(map(str, __version_info__))


def read_pcp_file(file):
    """
    Read a single precipitation file in SWAT format

    :param file: precipitation file path
    :return: pandas series where the index is datetime and precipitation is value
    """
    print('Lendo {}'.format(file))
    df = pd.read_csv(file, delimiter=DELIMITER, encoding=ENCODING, decimal=DECIMAL, skip_blank_lines=False)
    start_date = pd.to_datetime(df.columns[0], format='%Y%m%d')
    temp_var = df[df.columns[0]]
    serie_temp = []
    index_temp = []
    for index, row in temp_var.iteritems():
        index_temp.append(start_date + datetime.timedelta(days=index))
        if row == NO_DATA:
            serie_temp.append(np.nan)
        elif np.isnan(row):
            raise ValueError('Erro ao ler arquivo. Verifique linhas em branco ou valores incorretos.')
        else:
            serie_temp.append(row)
    var = pd.Series(serie_temp, index_temp)
    var.name = os.path.basename(file)
    return var


def read_pcp_index(file):
    """
    Read pcp.txt (index) file in SWAT format

    O arquivo pcp.txt é utilizado com um índice pelo SWAT e fornece infromações sobre as estações disponiveis

    :param file: caminho para o arquivo. Geralemte um arquivo com o nome pcp.txt
    :return: um dataframe pandas com as informções
    """
    print('Lendo {}'.format(file))
    data = None
    try:
        data = pd.read_csv(file)
    except Exception as e:
        raise ValueError('Erro abrindo arquivo {}: {}. Verifique formato.'.format(file, str(e)))

    # Corrige tamanho das letras do cabecalho para caixa alta, e retira espacos sobrando
    data.columns = [x.upper().strip() for x in data.columns]

    return data


def get_pcp_pts(pcp_index):
    pts = np.column_stack((pcp_index['LONG'], pcp_index['LAT']))
    z = pcp_index['ELEVATION']
    name = pcp_index['NAME']
    ident = pcp_index['ID']
    return ident, name, pts, z

def read_pcp(file):
    """
    Carrega o arquivo de indice e todos os arquivos indicados por ele.

    :param file: arquivo indice, geralmente pcp.txt
    :return: dataframe para o indice, uma lista com series pandas dos dados, e uma array com as coordenadas dos pontos
    """
    pcp_index = read_pcp_index(file)
    folder = os.path.dirname(file)

    observed_data = []
    for index, row in pcp_index.iterrows():
        #data = pd.read_csv(os.path.join(observed_data_folder, i['NAME']+'txt'))
        file_name = row['NAME']+'.txt'
        #print('reading {}'.format(file_name))
        file_path = os.path.join(folder, file_name )
        df = read_pcp_file(file_path)
        # modifica o dataframe para colocar a data com indice - isso facilita detectar problema de sincronia dos dados e permite automatizar
        # a a separacao dos dados por periodo.

        observed_data.append(df)
        
        #pts_observed = np.column_stack((pcp_index['LONG'], pcp_index['LAT']))
        #id, name, pts, z
        _, _, pts_observed, _ = get_pcp_pts(pcp_index)
    
    return pcp_index, observed_data, pts_observed


def get_centroid_from_shape(file, crs=4326):
    """
    abre um arquivo shape e encontra os centroides

    :param file: arquivo shapefile
    :param crs: projeção na qual o shapefile deve ser convertido para ser compativel com as coordenadas lat,lon do SWAT
    :return: array numpy com as cooreenadas dos centroides e o shape em geopandas
    """
    try:
        import geopandas
    except ImportError:
        raise ImportError(
            'Processamente de shape requer geopandas. Instale o pacote geopandas ou forneça '
            'os pontos a serem iterpolados manualmente')
    # Converte shape para projecao utilizada no swat
    shape = geopandas.read_file(file)
    
    shape = shape.to_crs(epsg=crs)
    centroids = shape.centroid
    pts_to_interpolate = np.array(list(shape.centroid.map(lambda p: [p.x, p.y])))
    return pts_to_interpolate, shape


def create_pcp_index(pcp_pts, pcp_name=None, pcp_z=None, pcp_id=None):
    """
    cria um indice de dados de precipitação a partir dos dados fornecidos
    :param pcp_pts:  pontos com os dados
    :param pcp_name: nome dos pontos
    :param pcp_z: altitude dos pontos
    :return: dataframe em pandas com o indice
    """
    columns = ['ID', 'NAME', 'LAT', 'LONG', 'ELEVATION']
    pcp_index = []
    for i, p in enumerate(pcp_pts):
        lat = p[1]
        lon = p[0]

        if pcp_id is None:
            id = i+1
        else:
            id = pcp_id[i]
        if pcp_name is None:
            name = 'ipcp{:04}'.format(id)
        else:
            name = pcp_name[i]
        if pcp_z is None:
            elevation = NO_DATA
        else:
            elevation = pcp_z[i]
        pcp_index.append([id, name, lat, lon, elevation])
    pcp_index = pd.DataFrame(pcp_index, columns=columns)
    return pcp_index


def save_pcp(pcp_index, pcp_data, pcp_folder):
    """
    salva os dados de precpitacao

    :param pcp_index: indice da precipitacao
    :param pcp_data: dados da precipitacao
    :param pcp_folder: diretorio de saida onde serao salvos os arquivos
    :return:
    """
    file = os.path.join(pcp_folder, 'pcp.txt')
    print('Salvando {}'.format(file))
    pcp_index.to_csv(file, index=False, decimal='.', sep=',', line_terminator='\r\n')
    for i in range(len(pcp_index)):
        index = pcp_index.iloc[i]
        serie = pcp_data[i]
        start_date = serie.index[0]
        name = index['NAME']
        file = os.path.join(pcp_folder,  name + '.txt')
        print('Salvando {}'.format(file))
        with open(file, 'w', newline='\r\n') as fo:
            fo.write('{}\n'.format(start_date.strftime('%Y%m%d')))
            serie.to_csv(fo, index=False, header=False, line_terminator='\n', na_rep=NO_DATA)


def harvesine(lon1, lat1, lon2, lat2):
    """
    calcula a distância aproximada entre dois pontos fornecidos em lat,lon.
    :param lon1:
    :param lat1:
    :param lon2:
    :param lat2:
    :return: distancia entre o pontos, em km??
    """
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def idwr(x, y, z, xi, yi):
    """
    Calcula interpolacao IDW
    :param x: coordenada x ponto de entrada
    :param y: coordenada y ponto de entrada
    :param z: valor no ponto (x,y)
    :param xi: coordenada x no ponto a ser interpolado
    :param yi: coordenda y no ponto a ser interpolado
    :return: array numpy com pontos interpolados
    """
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.sum(sumsup)
        sumsup = np.sum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        #xyzi = [xi[p], yi[p], u]
        #lstxyzi.append(xyzi)
        lstxyzi.append(u)
    return np.array(lstxyzi)


def idw(points, val, ipts):
    """
    wrapper para utilizar idwr num formato compativel com o interpolador grid no scipy
    :param points: coordenadas dos pontos de entrada
    :param val: valores
    :param ipts: coordenadas dos pontos a serem interpolados
    :return: dados interpolados
    """
    return idwr(points[:, 0], points[:, 1], val, ipts[:, 0], ipts[:, 1])


def interpolate(points, values, xi, method='nearest'):
    """
    wrapper que lida com a naturaza temporal dos dado de precipiação. Ele extrai a precipitacao em cada
    ponto em um determiando tempo , interpola e formata a saida para ficar consistente
    :param points:  pontos de entradas
    :param values:  valores de entrada
    :param xi: pontos a serem interpolados
    :param method: metodo de interpolacao
    :return: dados interpoladors
    """
    # Esse eh o index utilziado para os dados
    index = values[0].index
    if method == 'idw':
        t_values = np.array(values).T
        temporal_size = len(t_values)
        xi_size = len(xi)
        out = np.empty([temporal_size, xi_size])
        #for i in range(temporal_size):
        for i in progress_bar(range(temporal_size)):
            val = t_values[i]
            #out[i] = griddata(points, val , xi, method=method)
            # Utiliza somente dados validos para interpolar (remove NaN e inf)
            valid_num_id = np.isfinite(val)
            if len(valid_num_id) != 0:
                out[i] = idw(points[valid_num_id], val[valid_num_id], xi)
            else:
                out[i] = np.NaN
        out = pd.DataFrame(out)
        out.index = index
        return out
    elif method == 'nearest':
        from scipy.interpolate import griddata
        t_values = np.array(values).T
        temporal_size = len(t_values)
        xi_size = len(xi)
        out = np.empty([temporal_size, xi_size])
        #for i in range(temporal_size):
        for i in progress_bar(range(temporal_size)):
            val = t_values[i]
            out[i] = griddata(points, val, xi, method='nearest')
        out = pd.DataFrame(out)
        out.index = index
        return out
    else:
        raise ValueError('Metodo de interpolacao indisponivel: {}'.format(method))
        

def progress_bar(iterable, prefix='', suffix='', decimals=1, length=50, fill='#', printend="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        #print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printend)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        #sys.stdout.flush()
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


def save_figure(file, obs_pts, int_pts, shape=None):
    import matplotlib.pyplot as plt
    import geopandas as gpd
    f, ax = plt.subplots(1, figsize=(15, 8))
    #f.suptitle('Shape com pontos obsrvados e interpolados')
    pcp_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(obs_pts[:, 0], obs_pts[:, 1]))
    ipcp_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(int_pts[:, 0], int_pts[:, 1]))
    if shape is not None:
        shape.plot(ax=ax)
    pcp_pts.plot(ax=ax, color='red')
    ipcp_pts.plot(ax=ax, color='black')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='black', lw=4)]

    # fig, ax = plt.subplots()
    # lines = ax.plot(data)
    ax.legend(custom_lines, ['observado', 'interpolado'])
    #plt.show()
    # tem um bug se chamado assim, sem borda
    #plt.savefig(file, bbox_inches='tight')
    plt.savefig(file)


def check_input_data(values):
    # TODO: Precisa ver como que fica os casos onde os arquivos tem dados de periodos diferentes
    # Verifica se todos os pontos correspondem ao mesmo periodo de tempo.
#    data_len = len(values[0])
#    first_index = values[0].index[0]
#    for i in range(len(values)):
#        if data_len != len(values[i]):
#            raise ValueError('Quantidade de pontos encontrados ({}) diferente do esperado ({}) no arquivo {}. Verifique se dados correspondeM ao mesmo periodo'.format(len(values[i]), data_len, values[i].name))
#        if first_index != values[i].index[0]:
#            raise ValueError('Data de inicio ({}) diferente do esperado ({}) no arquivo {}. Verifique se dados correspondeM ao mesmo periodo'.format(values[i].index[0], first_index, values[i].name))
    # Encontra a faixa de tempo em que existe dados para todos os pontos
    data_len = len(values[0])
    start_index = values[0].index[0]
    end_index = values[0].index[-1]
    period_mismatch = False
    #print('Periodo de dados: {} - {}'.format(start_index, end_index))
    for i in range(len(values)):
        serie_start = values[i].index[0]
        print('Serie {}, periodo: ({} - {}), numero de pontos: {}'.format(values[i].name, values[i].index[0], values[i].index[-1], len(values[i]) ))
        if start_index != serie_start:
            period_mismatch = True
            if start_index < serie_start:
                start_index = serie_start
            #print("AVISO: serie {} com data inicial {} diferente de periodo. Utilizando periodo: {} - {}".format(values[i].name, serie_start, start_index, end_index))
        serie_end = values[i].index[-1]
        if end_index != serie_end:
            period_mismatch = True
            if end_index > serie_end:
                end_index = serie_end
            #print("AVISO: serie {} com data final {} diferente de periodo. Utilizando periodo: {} - {}".format(values[i].name, serie_end, start_index, end_index))

    if period_mismatch:
        print("AVISO: periodo das series diferente. Utilizando somente dados presentes em todas series.")
    # Fica somente com o overlap de tempo
    print('Periodo de dados final: {} - {}'.format(start_index, end_index))
    for i in range(len(values)):
        values[i] = values[i][start_index:end_index]

    # Covnerte para numerico
    for i in range(len(values)):
        try:
            values[i] = pd.to_numeric(values[i])
        except Exception as e:
            raise ValueError('Erro ao converter serie {}: {}. Verifique formato.'.format(values[i].name, str(e)))


    # Checa valores

    return values


def cmd_line():
    """
    Prepara o parsser para ler os dados inseridos pela linha de comando.
    :return: variaveis de configuracao
    """
    parser = argparse.ArgumentParser(prog="pcpswat.py", description="Ferramenta de interpolação de precipitação para SWAT")

    parser.add_argument(
        'pcp_file', help="pcp index file", type=str)
    parser.add_argument(
        'destination_folder', help="destination folder", type=str)

    parser.add_argument(
        '-shape',
        action='store',
        help="shape file para obter o centroide automaticamente",
        type=str)
    parser.add_argument(
        '-interpolate',
        action='store',
        help="arquivo com os pontos a serem interpolados",
        type=str)
    parser.add_argument(
        '-method',
        action='store',
        default='nearest',
        help="metodo de interpolacao",
        type=str)
    parser.add_argument(
        '-savefig',
        action='store_true',
        help="salva arquivo com imagem dos pontos observados, interpolados e shape")

    args = parser.parse_args()

    if not (args.shape or args.interpolate):
        parser.error('-shape or -interpolate required')

    pcp_file = args.pcp_file
    dst_folder = args.destination_folder
    interpolate_file = args.interpolate
    shape_file = args.shape
    interpolate_method = args.method
    save_fig = args.savefig
    return pcp_file, dst_folder, interpolate_file, shape_file, interpolate_method, save_fig


def main():
    """
    rotina principal caso o seja chamado pela linha de comando. As funcoes tb podem ser chamadas externamentes.
    :return:
    """
    import warnings
    warnings.filterwarnings("ignore")

    print('Ferramenta de interpolacao de precipitacao para SWAT {}'.format(__version__))
    pcp_file, dst_folder, interpolate_file, shape_file, interpolate_method, save_fig = cmd_line()

    shape_available = False
    interpolate_from_shape = True
    if interpolate_file is not None:
        print('Pontos a interpolar definidos manualmente')
        interpolate_from_shape = False
        int_index = read_pcp_index(interpolate_file)
        pts_id, pts_name, pts_to_interpolate, pts_z = get_pcp_pts(int_index)

    if shape_file is not None:
        if interpolate_from_shape:
            print('Obtendo pontos a interpolar do arquivo shape')
            pts_to_interpolate, shape = get_centroid_from_shape(shape_file)
        else:
            print('Carregando arquivo shape')
            _, shape = get_centroid_from_shape(shape_file)
        shape_available = True

    print('Carregando dados de precipitacao')
    pcp_index, observed_data, pts_observed = read_pcp(pcp_file)

    if save_fig:
        file = os.path.join(dst_folder, 'fig.png')
        print('Salvando figura {}'.format(file))
        if shape_available:
            save_figure(file, pts_observed, pts_to_interpolate, shape)
        else:
            save_figure(file, pts_observed, pts_to_interpolate)
    print('Verificando dados...')
    observed_data = check_input_data(observed_data)

    print('Interpolando utilizando metodo {}'.format(interpolate_method))
    interpolated_data = interpolate(pts_observed, observed_data, pts_to_interpolate, method=interpolate_method)
    if interpolate_from_shape:
        interpolated_index = create_pcp_index(pts_to_interpolate)
    else:
        interpolated_index = create_pcp_index(pts_to_interpolate, pts_name, pts_z, pts_id)

    print('Salvando dados interpolados')
    save_pcp(interpolated_index, interpolated_data, dst_folder)


if __name__ == "__main__":
    main()
