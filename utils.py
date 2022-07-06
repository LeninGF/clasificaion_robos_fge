import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import html
import re
from datetime import datetime
from tqdm import tqdm
from sqlalchemy import create_engine, text
from string import Formatter
tqdm.pandas()


# involucrados dictionary
dict_involucrados_tipo ={
        'PRESUNTA_VICTIMA':['VICTIMA', 'VICTIMA NO RECONOCIDA', 'DESAPARECIDO', 'DESAPARECIDO NO RECONOCID', 'FALLECIDO NO RECONOCIDO', 'FALLECIDO', 'PERJUDICADO', 'PERJUDICADO NO RECONOCIDO'],
        'DENUNCIANTE':['DENUNCIANTE', 'VICTIMA DENUNCIANTE', 'VICTIMA-DENUNCIANTE'],
        'SOSPECHOSO':['SOSPECHOSO NO RECONOCIDO', 'SOSPECHOSO', 'PROCESADO', 'PROCESADO NO RECONOCIDO', 'APREHENDIDO', 'DETENIDO', 'APREHENDIDO NO RECONOCIDO']
    }


def dataframe_difference(df_left, df_right, which=None):
    """
    This function compares two dataframes and returns a new dataframes with the comparison between the rows of each
    of them according to the following logic:
    1. Which rows are only present in df1 --> right_only
    2. Which rows are only present in df_right --> left_only
    3. which rows are present in both --> which = both
    4. Which rows are not present in both Dataframes, but present in one of them --> which = None
    url: https://hackersandslackers.com/compare-rows-pandas-dataframes/
    :param df_left: a dataframe with data
    :param df_right: a dataframe with data
    :param which: None, both, left_only, right_only
    :return: dataframe with rows compared between two dataframes
    """
    comparison_df = df_left.merge(df_right, indicator=True, how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df


def columna_mes_nombre(bbdd, campo, abrev, etiqueta_mes_abrv="MES_ABRV", etiqueta_mes_cmpl="MES_CMPL"):
    """
    Esta función permite añadir al dataframe las columnas MES_ABRV y MES_CMPL. La primera contiene nombres de meses
    abreviados (e.g. ene, feb, mar, etc.). Mientras que la segunda los nombres completos
    :param bbdd: es el dataframe con los datos
    :param campo: campo que contiene al mes como un entero de 1 al 12
    :param abrev: booleano. False genera columnas con nombre completo y abreviado
    :return:
    """
    "campo apunto a uno en el que tengamos el numero del mes"
    "abrev: False Genera dos columnas con nombre completo y abreviado"
    "abrev: True genera solo abreviados"
    mes_num = np.arange(1.0, 13.0, step=1., dtype=np.float64)
    mes_nomb_completo = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre',
                         'Octubre', 'Noviembre', 'Diciembre']
    mes_nomb_abrev = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
    mes_dic_abrev = dict(zip(mes_num, mes_nomb_abrev))
    mes_dic_comp = dict(zip(mes_num, mes_nomb_completo))
    if abrev:
        bbdd[etiqueta_mes_abrv] = bbdd[campo].apply(lambda x: mes_dic_abrev[x])
    else:
        bbdd[etiqueta_mes_abrv] = bbdd[campo].apply(lambda x: mes_dic_abrev[x])
        bbdd[etiqueta_mes_cmpl] = bbdd[campo].apply(lambda x: mes_dic_comp[x])


def format_fecha_registro_datetime(bbdd, campo, formato_t='%Y-%m-%d'):
    """
    Función que da formato datetime al str de fecha leído de un csv
    :param bbdd: dataframe obtenido de un csv
    :param campo: apunta al campo que contiene las fechas en str
    :param formato_t: es el formato en que está escrita la fecha:'%Y-%m-%d', '%d/%m/%Y'
    :return: devuelve el campo en formato datetime
    """
    bbdd[campo] = pd.to_datetime(bbdd[campo], format=formato_t)


def crear_mes_registro(bbdd, campo, nombre_campo='MES_R'):
    """
    Función que permite crear un campo con el número de Mes (i.e. 1,2,...12)
    :param bbdd: dataframe con la información
    :param campo: datetime que contiene la fecha
    :param nombre_campo: nombre del campo de salida por defecto MES_R
    :return: devuelve datagrame con campo Mes que contiene el número del mes
    """
    bbdd[nombre_campo] = bbdd[campo].apply(lambda x: x.month)


def crear_anio_registro(bbdd, campo, nombre_campo="AÑO_R"):
    """
    Función que devuelve el año correspondiente a la fecha del campo
    :param bbdd: dataframe con los datos
    :param campo: debe estar en datetime
    :param nombre_campo: nombre del campo de salida. por defecto AÑO_R
    :return:
    """
    bbdd[nombre_campo] = bbdd[campo].apply(lambda x: x.year)


def convert2datetime(bbdd, campo_date):
    """
    Convertir los timestamps de un campo de datetimes a datetimes
    
    """
    # bbdd[campo_date] = bbdd[campo_date].apply(lambda x: x.to_pydatetime() if type(x) is  pd._libs.tslibs.timestamps.Timestamp else x)
    bbdd[campo_date] = pd.to_datetime(bbdd[campo_date])


def crear_dia_semana(bbdd, fecha_dt, dia_num, dia_nombre):
    """
    Función que devuelve el día de la semana correspondiente a la fecha del campo
    :parm bbdd: dataframe con los datos
    :param fecha_dt: campo que contiene la fecha en datetime
    :dia_num: nombre del campo que contendrá el número del día de la semana. 0 is Monday
    :dia_nombre: nombre del día de la semana
    return: columnas dia_num y dia_nombre según los valores provistos 
    """
    bbdd[fecha_dt] = pd.to_datetime(bbdd[fecha_dt], errors='ignore')
    bbdd[dia_num] = bbdd[fecha_dt].apply(lambda x: x.weekday())
    dayOfWeek_dict = {0:'LUNES',
                      1:'MARTES',
                      2:'MIERCOLES',
                      3:'JUEVES',
                      4:'VIERNES',
                      5:'SABADO',
                      6:'DOMINGO'}
    bbdd[dia_nombre] = bbdd[fecha_dt].apply(lambda x: dayOfWeek_dict[x.weekday()])
    
    

def cambiar_fecha_spss_py(bbdd_df, campo):
    ini_spss = datetime(1582,10,14)
    fini_py = datetime(1970,1,1)
    ajuste = (fini_py-ini_spss).total_seconds()
    bbdd_df[campo] = bbdd_df[campo].apply(lambda x: pd.to_datetime(x-ajuste, unit='s'))


def cambiar_fecha_spss_py_progress(bbdd_df, campo):
    ini_spss = datetime(1582,10,14)
    fini_py = datetime(1970,1,1)
    ajuste = (fini_py-ini_spss).total_seconds()
    bbdd_df[campo] = bbdd_df[campo].progress_apply(lambda x: pd.to_datetime(x-ajuste, unit='s'))


def grupo_etario(edad):
    if (edad >= 0) and (edad <= 11):
        return 'NIÑ@S (DE 0 A 11 AÑOS)'
    elif (edad >= 12) and (edad <= 17):
        return 'ADOLESCENTES (DE 12 A 17 AÑOS)'
    elif (edad >= 18) and (edad <= 29):
        return 'JÓVENES (DE 18 A 29 AÑOS)'
    elif (edad >= 30) and (edad <= 64):
        return 'ADULTOS (DE 30 A 64 AÑOS)'
    elif (edad >= 65) and (edad <= 100):
        return 'ADULTOS MAYORES (DESDE 65 AÑOS)'
    elif edad is np.nan:
        return 'SIN INFORMACION'
    else:
        return 'SIN INFORMACION'
    

def grupo_etario_forma2(edad):
    if (edad >= 0) and (edad < 15):
        return 'MENORES DE 15 AÑOS'
    elif (edad >= 15) and (edad < 25):
        return 'ENTRE 15 Y 24 AÑOS'
    elif (edad >= 25) and (edad < 35):
        return 'ENTRE 25 Y 34 AÑOS'
    elif (edad >= 35) and (edad < 45):
        return 'ENTRE 35 Y 44 AÑOS'
    elif (edad >= 45) and (edad < 65):
        return 'ENTRE 45 Y 64 AÑOS'
    elif (edad >= 65) and (edad <=100):
        return 'MAYORES DE 65 AÑOS'
    elif edad is np.nan:
        return 'SIN INFORMACION'
    else:
        return 'SIN INFORMACION'


def es_menor_edad(edad):
    if (edad >= 0) and (edad < 18):
        return 'MENOR DE EDAD'
    elif edad is np.nan:
        return 'SIN INFORMACION'
    elif edad < 0:
        return 'SIN INFORMACION'
    else:
        return 'MAYOR DE EDAD'
    
    
def sort_pandas_crosstab(df, columna_sort='TOTAL'):
    """
    Funcion que permite ordenar un pandas crosstab de mayor a menor según el valor de la columna
    Total
    :param df: es el dataframe esto es el crosstab
    :param columna_sort: es la columna que se quiere re-ordenar
    """
    df_sorted = df.sort_values(columna_sort, ascending=False )
    idx = df_sorted.index.tolist()
    idx.pop(0) # NOTAL QUE 0 CORRESPONDE A TOTAL
    df_sorted = df_sorted.reindex(idx+[columna_sort])
    return df_sorted


def sort_pandas_crosstab_more_rows(df, columna_sort='TOTAL'):
    """
    Funcion que permite ordenar un pandas crosstab de mayor a menor según el valor de la columna
    Total
    :param df: es el dataframe esto es el crosstab
    :param columna_sort: es la columna que se quiere re-ordenar
    """
    df_sorted = df.sort_values(columna_sort, ascending=False )
    idx = df_sorted.index.tolist()
    idx.pop(0) # NOTAL QUE 0 CORRESPONDE A TOTAL
    df_sorted = df_sorted.reindex(idx+[(columna_sort,'')])
    return df_sorted


def masking_nnds(df, field):
    ndds_uniques = df[field].unique().tolist()
    t = np.random.random(size=len(ndds_uniques))
    t_= t*111111111111111
    masked_ndd = t_.astype(np.int64)
    ndd_mask_dict = dict(zip(ndds_uniques, masked_ndd))
    df["NDD_MASKED"] = df[field].apply(lambda x: ndd_mask_dict[x])
    df["NDD_MASKED"] = df["NDD_MASKED"].apply(lambda x: str(x))
    df["NDD_MASKED"] = df["NDD_MASKED"].str.zfill(15)


def save_df_in_sql(name_table, dataf, index=False, where="bigData"):
    """
    This function saves a dataframe as a Table in a MariaDB MySQL database
    """    
    if where=="bigData":
        engine_maria_db = create_engine("mysql+pymysql://falconiel:BebuSuKO@192.168.20.238:3306/reportes"+"?charset=utf8mb4")
    else:
        engine_maria_db = create_engine("mysql+pymysql://falconiel:F0s!Hu63@192.168.20.217:3306/reportes"+"?charset=utf8mb4")
    print("conexion con base es: {}".format(engine_maria_db.connect()))
    dataf.to_sql(name_table, engine_maria_db, if_exists='replace', index=index, chunksize=1000)


def grupos_horarios(hour_as_string):
    """
    Función que permite generar grupos horarios de horas
    :param hours_as_string: la hora debe ingresar como un string para ser
    convertida al grupo respectivo

    Ejemplo de uso: siaf_bbdd["Hora_Incidente_Grupo_Horario"] = siaf_bbdd.Hora_Incidente.apply(lambda x: grupos_horarios(x))
    """
    t = pd.to_datetime(hour_as_string)
    limit_inf_t1 = pd.to_datetime('00:00:00')
    limit_sup_t1 = pd.to_datetime('05:59:59')
    limit_inf_t2 = pd.to_datetime('06:00:00')
    limit_sup_t2 = pd.to_datetime('11:59:59')
    limit_inf_t3 = pd.to_datetime('12:00:00')
    limit_sup_t3 = pd.to_datetime('17:59:59')
    limit_inf_t4 = pd.to_datetime('18:00:00')
    limit_sup_t4 = pd.to_datetime('23:59:59')
    if (t<=limit_sup_t1) and (t>=limit_inf_t1):
        return "00:00 - 05:59"
    elif (t<=limit_sup_t2) and (t>=limit_inf_t2):
        return "06:00 - 11:59"
    elif (t<=limit_sup_t3) and (t>=limit_inf_t3):
        return "12:00 - 17:59"
    elif (t<=limit_sup_t4) and (t>=limit_inf_t4):
        return "18:00 - 23:59"
    else:
        print("SD for: ",hour_as_string)
        return "S.D."


def conectar_sql(big_data_bbdd=True):
    # F0s!Hu63
    if big_data_bbdd:
        engine_maria_db = create_engine("mysql+pymysql://falconiel:BebuSuKO@192.168.152.197")
        print("conectando con big data database....")
    else:
        engine_maria_db = create_engine("mysql+pymysql://falconiel:F0s!Hu63@192.168.152.8")  # N27a34v1
        print("conectando con proxy database....")
    print(engine_maria_db.connect())
    return engine_maria_db


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can 
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the  
    default, which is a datetime.timedelta object.  Valid inputtype strings: 
        's', 'seconds', 
        'm', 'minutes', 
        'h', 'hours', 
        'd', 'days', 
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


# El siguiente código valida cédulas ecuatorianas
# http://blog.espol.edu.ec/ccpg1001/s2eva_it2008_t2-validar-cedula-ecuatoriana/
def validar_cedula(texto):
    if texto is None:
        return 0
    if not texto.isnumeric():
        return 0
    if len(texto) < 10 and len(texto) > 0:
        return 0
    if len(texto) > 10:
        return 0
    # sin ceros a la izquierda
    # nocero = texto.strip("0")
    nocero = texto.lstrip("0")
    if len(nocero) == 0:
        return 0
    cedula = int(nocero,0)
    verificador = cedula%10
    numero = cedula//10
    
    # mientras tenga números
    suma = 0
    while (numero > 0):
        
        # posición impar
        posimpar = numero%10
        numero   = numero//10
        posimpar = 2*posimpar
        if (posimpar  > 9):
            posimpar = posimpar-9
        
        # posición par
        pospar = numero%10
        numero = numero//10
        
        suma = suma + posimpar + pospar
    
    decenasup = suma//10 + 1
    calculado = decenasup*10 - suma
    if (calculado  >= 10):
        calculado = calculado - 10

    if (calculado == verificador):
        validado = 1
    else:
        validado = 0
        
    return (validado)


def validar_ruc(texto):
    """Validar Ruc
    Se sigue la misma rutina de validar cédula una vez que se ha retirado el 001
    Args:
        texto (_type_): _description_

    Returns:
        _type_: _description_
    """
    if texto is None:
        return 0
    if not texto.isnumeric():
        return 0
    if len(texto) < 13 and len(texto) > 0:
        return 0
    if len(texto) > 13:
        return 0
    # sin ceros a la izquierda
    # nocero = texto.strip("0")
    texto = re.sub('[^0-9]+', '', texto) 
    tmp = re.findall(r"(\d{10})001", texto, re.MULTILINE)
    if len(tmp) > 0:
        texto = tmp[0]
    else:
        return 0
    validado = validar_cedula(texto)    
    return (validado)


def femicidios_process(csv_femicidios):
    """
    This function reads the Femicidios Matrix and formats it to
    be used.
    """
    bdd_femicidios = pd.read_excel(csv_femicidios,
                               sheet_name='matriz',
                               header=1,
                               parse_dates=['fecha_i', 
                                            'fecha_m', 
                                            'fecha_acta', 
                                            'fecha_ndd', 
                                            'fecha_proceso', 
                                            'fecha_actualizacionCJ'],
                               converters = {'NDD':str,
                                             'cod_unico':str,
                                             'numero_juicio':str})
    bdd_femicidios_report = bdd_femicidios[bdd_femicidios.Tipo_muerte_SRMCE=="FEMICIDIO"]
    return bdd_femicidios_report


def extraer_relato(lista_ndds, sql_connection):
    """
    Devuelve un dataframe que contiene la NDD y el relato de los hechos
    @lista_ndds: lista con las ndds a ser consultadas
    @sql_connection: objeto que crea la conexion con la base
    return dataframe con NDD y Relato
    """
    sql_query = text("""
                 SELECT den.codfisc AS 'NDD', bdd_enlace_externo.fnStripTags(den.obserinc) AS 'RELATO'
                 FROM fgn.denuncia_fiscalia AS den
                 WHERE den.estado=1 AND den.anulada='NO' AND den.codfisc IN :ndds_list
                 GROUP BY den.codfisc;
                 """)
    sql_query = sql_query.bindparams(ndds_list=tuple(lista_ndds))
    relatos = pd.read_sql(sql_query, sql_connection)
    relatos.RELATO = relatos.RELATO.str.lower()
    relatos.RELATO = relatos.RELATO.apply(lambda x: html.unescape(x))
    # Removiendo xa0 que proviende de encodificacion Latin1 ISO8859-1
    relatos.RELATO = relatos.RELATO.str.replace(u'\xa0', u' ')
    return relatos


def mostrar_todas_filas_columnas():
    """
    Configura el entorno para desplegar todas las filas y columnas
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)


def ofuzcar_nombres(dataf, campo_nombre, nombre_campo_out="NOMBRE_OFUZCADO"):
    """
    Genera una columna nueva en el DataFrame con los valores del string de
    nombres ofuzcados. Para esto se genera un diccionario clave el nombre original
    y valor el nombre ofuzcado
    @dataf: es el dataframe
    @campo_nombre: es el campo en donde se encuentran los valores de nombres
    @nombre_campo_out: es el nombre que tendrá el nuevo campo
    return el dataframe con una columna de nombres ofuzcados
    """
    nombres_unicos_list = dataf[campo_nombre].unique().tolist()
    t = np.random.random(size=len(nombres_unicos_list))
    t_ = t*111111111111111
    t_ = t_.astype(np.int64)
    oculta_nombre = ['nombre_'+str(x) for x in t_]
    codigo_nombres_dict = dict(zip(nombres_unicos_list, oculta_nombre))
    dataf[nombre_campo_out] = dataf[campo_nombre].apply(lambda x: codigo_nombres_dict[x])


def ofuscar_ndd(dataf, campo_ndd, nombre_campo_out="NDD_OFUSCADA"):
    ndds_lista = dataf[campo_ndd].unique().tolist()
    t = np.random.random(size=len(ndds_lista))
    t_ = t*111111111111111
    masked_ndd = t_.astype(np.int64)
    ndd_mask_dict = dict(zip(ndds_lista, masked_ndd))
    dataf[nombre_campo_out] = dataf[campo_ndd].apply(lambda x: ndd_mask_dict[x])
    return ndd_mask_dict


def extraer_numeroDiligenciasImpulsos(sql_conn, lista_ndds):
    """
    Devuelve la información de la NDD con su estado procesal, número de diligencias, número de impulsos, aprehendidos,
    sospechosos, etc.
    """
    sql_query = text("""
                 SELECT 
                 df.codfisc AS NDD, 
                 df.fecha AS Fecha_Registro,
                 df.hora AS Hora_Registro,
                 df.fechainc AS Fecha_Incidente,
                 df.horainc AS Hora_Incidente,
                 If(df.tentativa='N','No','Si') AS Tentativa,
                 df.direccion AS Direccion,
                 del.gen_delito_tipopenal AS Presunto_Delito,
                 del.gen_delito_concatenado AS 'Presunto Delito (Circunstancia Modificatoria)',
                 (
                     SELECT fisc.fxz_descripcion
                     FROM fgn.fiscalias AS fisc
                     WHERE df.fiscalias = fisc.fxz_codigo
                     ) AS 'Fiscalia',
                     fzxu.descripcion AS Fiscalia_Especializada,
                     (
                         SELECT fiscalia.gen_canton.can_descripcion
                         FROM fiscalia.gen_canton
                         WHERE fiscalia.gen_canton.can_codigo = df.ciudades) AS 'Ciudad',
                         fzxu.pro_descripcion AS PROVINCIA,
                         (
                             SELECT fiscalia.gen_canton.can_descripcion
                             FROM fiscalia.gen_canton
                             WHERE fiscalia.gen_canton.can_iso = substr(df.parroquias,1,4) AND fiscalia.gen_canton.can_estado = '1') AS 'Canton',
                             (
                                 SELECT par_nombre
                                 FROM fiscalia.gen_parroquia AS parr
                                 WHERE df.parroquias = parr.par_codigo) AS 'Parroquia',
                                 (
                                     SELECT ti.descripcion
                                     FROM fgn.tipoincidente AS ti
                                     WHERE df.tipoincidente=ti.cdg) AS 'Tipo',
                                     if (df.tipodelito=0, 'No Flagrante','Flagrante') AS NyF, CONCAT ('Fiscalia ',fzxu.fzxu_num_fiscalia) AS numero_fiscalia,
                                     fzxu.fxz_descripcion AS edificio,
                                     fzxu.fzxu_nombre_fiscal AS nombre_fiscal,
                                     (
                                         SELECT tp2_descripcion
                                         FROM fiscalia.gen_tipos2 AS fue
                                         WHERE df.fuero = fue.tp2_codigo) AS 'Fuero', CASE fso.fso_etapas WHEN 265 THEN CONCAT('INVESTIGACION PREVIA') WHEN 266 THEN CONCAT('INSTRUCCION FISCAL') WHEN 267 THEN CONCAT('PREPARATORIA DE JUICIO') WHEN 268 THEN CONCAT('JUICIO') WHEN 1065 THEN CONCAT('IMPUGNACION') ELSE 'SIN ACCIONES' END AS Etapa_procesal, CASE WHEN tip2.tp2_descripcion IS NULL THEN 'POR APERTURAR INVETIGACION PREVIA' ELSE tip2.tp2_descripcion END AS Estado_Procesal,
                                         (
                                             SELECT COUNT(DISTINCT exndd.codigo_exn) AS numero_diligencias
                                             FROM milenium.proceso_ndd AS pn
                                             INNER JOIN milenium.experticiaxndd exndd ON exndd.cod_uni_exp=pn.codigo_proc
                                             WHERE pn.estado = 1 AND exndd.estado = 1 AND exndd.NDD=df.codfisc) AS 'IMPULSOS_DILIGENCIAS',
                                             (
                                                 SELECT COUNT(pn.ndd) AS impulsos
                                                 FROM milenium.proceso_ndd AS pn
                                                 WHERE pn.estado = 1 AND pn.ndd=df.codfisc) AS 'IMPULSOS',
                                                 ult.nua_ult_accion AS Ultima_accion,
                                                 ult.nua_fecha_ult_accion AS Fecha_Ultima_Accion,
                                                 (
                                                     SELECT COUNT(inv1.id_involucrado)
                                                     FROM fgn.involucrado inv1
                                                     WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (34,193)) AS APREHENDIDO,
                                                     (
                                                         SELECT COUNT(inv1.id_involucrado)
                                                         FROM fgn.involucrado inv1
                                                         WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (37,35)) AS SOSPECHOSO,
                                                         (
                                                             SELECT COUNT(inv1.id_involucrado)
                                                             FROM fgn.involucrado inv1
                                                             WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (53,194)) AS PROCESADO,
                                                             (
                                                                 SELECT COUNT(inv1.id_involucrado)
                                                                 FROM fgn.involucrado inv1
                                                                 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (995)) AS DETENIDO,
                                                                 (
                                                                     SELECT COUNT(inv1.id_involucrado)
                                                                     FROM fgn.involucrado inv1
                                                                     WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (49)) AS IMPUTADO,
                                                                     df.numeroinf,
                                                                     (
                                                                         SELECT tipinc.descripcion
                                                                         FROM fgn.tipoincidente tipinc
                                                                         WHERE df.tipoincidente=tipinc.cdg
                                                                         LIMIT 1) AS Tipo_incidente,
                                                                         ip.ind_fecha_ini,
                                                                         ip.ind_fecha_fin,
                                                                         ins.ins_fecha_inicio,
                                                                         ins.ins_fecha_cierre,
                                                                         bdd_enlace_externo.quitaEntities(bdd_enlace_externo.fnStripTags(df.obserinc)) AS Relato
                                                                         FROM fgn.denuncia_fiscalia df
                                                                         INNER JOIN fgn.fiscal_sorteado fso ON fso.ndd_codigo=df.codfisc
                                                                         INNER JOIN fgn.gen_fiscalia_zonal_unidad_vista fzxu ON fzxu.fzxu_codigo=fso.fzxu_codigo
                                                                         INNER JOIN fgn.gen_delitos del ON del.gen_delito_secuencia=df.infraccion
                                                                         LEFT JOIN fgn.fa2_ndd_ult_accion ult ON ult.nua_ndd=df.codfisc
                                                                         LEFT JOIN fiscalia.gen_tipos2 tip2 ON tip2.tp2_codigo=fso.fso_estado_procesal
                                                                         LEFT JOIN fgn.fa2_indagacion_previa ip ON ip.ndd_codigo=fso.ndd_codigo AND ip.ind_estado=1
                                                                         LEFT JOIN fgn.fa2_instruccion ins ON ins.ndd_codigo=fso.ndd_codigo AND ip.ind_estado=1
                                                                         WHERE df.estado=1 AND df.anulada='NO' AND fso.fso_estado<3
                                                                         -- AND df.fecha BETWEEN '2019-11-01' AND '2019-11-20'
                                                                         -- AND df.fechainc BETWEEN '2019-10-02' AND '2019-10-13'
                                                                         -- AND df.codfisc IN (
                                                                             AND df.codfisc IN :ndds_list
                                                                             GROUP BY df.codfisc;
                                                                             """)
    sql_query = sql_query.bindparams(ndds_list=tuple(lista_ndds))
    num_diligencias_ep = pd.read_sql(sql_query, sql_conn)
    return num_diligencias_ep


def extraer_detalle_diligencia(sql_conn, lista_ndds):
    """
    Función para traer el detalle de las diligencias por NDD
    @sql_conn: elemento de conexión con la base de datos
    @lista_ndds: listado de ndds a realizar la consulta (únicas)
    """
    resp_sql_ac = []
    for ndd in tqdm(lista_ndds):
        sql_query = text("""
                     SELECT
                     (select concat(f.ape_paterno_fun, ' ', f.ape_materno_fun, ' ', f.nombres_fun) from talento.funcionario f where f.codigo_fun = dcm.dcm_aprobado) as FISCAL,
                     (select concat(f.ape_paterno_fun, ' ', f.ape_materno_fun, ' ', f.nombres_fun) from talento.funcionario f WHERE f.identificacion_fun = dcm.dcm_razon_cedula_secre) as SECRETARIO,
                     expendd.NDD,
                     -- expendd.fxzu_codigo,
                     vis.pro_descripcion AS PROVINCIA,
                     vis.can_descripcion AS CANTON,
                     vis.fxz_descripcion AS EDIFICIO,
                     vis.concat_descripcion AS ESPECIALIDAD,
                     case  when expendd.estado_dil = 0 then 'SOLICITADO' ELSE 'RECIBIDO/ESCANEO' END AS ESTADO_DLIGENCIA,
                     dcm.dcm_numero AS NUMERO_OFICIO,
                     DATE_FORMAT(dcm.fecha_crea, '%Y-%m-%d') AS FECHA,
                     -- case when fs.fso_fecha>'2014-08-10' then pndd.no_resolucion else 0 end as 'NO_IMPULSO',
                     -- dcm.numero_secuencial,
                     pndd.no_resolucion as 'NO_IMPULSO',
                     x.descripcion_exp as 'TIPO_DILIGENCIA'
                     FROM milenium.experticias expe
                     INNER JOIN milenium.experticiaxndd expendd ON expendd.NDD IN ('{}') AND expe.codigo_exp=expendd.codigo_exp AND expe.estado=1 AND expendd.estado_dil!=-1
                     INNER JOIN milenium.experticias x ON x.codigo_exp = expendd.codigo_exp
                     LEFT JOIN fgn.fiscal_sorteado fs ON fs.ndd_codigo=expendd.NDD AND fs.fzxu_codigo=expendd.fxzu_codigo AND fs.fso_estado<3
                     INNER JOIN  fisdoc.documento_misional dcm  ON  dcm.dcm_codigo=expendd.cod_doc AND dcm.dcm_estado=17  and dcm.dcm_codigo_tabla_catalogo = 525 AND dcm.ndd_codigo=expendd.NDD AND dcm.fzxu_codigo=expendd.fxzu_codigo
                     INNER JOIN milenium.proceso_ndd pndd ON pndd.ndd=expendd.NDD AND pndd.fxzu_codigo=expendd.fxzu_codigo AND pndd.codigo_proc=expendd.cod_uni_exp AND pndd.estado=1
                     INNER JOIN fgn.gen_fiscalia_zonal_unidad_vista vis ON vis.fzxu_codigo=dcm.fzxu_codigo
                     """.format(ndd))
        resp_sql_ac.append(pd.read_sql(sql_query, con=sql_conn))
        detalle_diligencias_df = pd.concat(resp_sql_ac, ignore_index=True)
    return detalle_diligencias_df


def consultar_estadoprocesal(sql_conn, lista_ndds):
    """
    Devuelve un dataframe con los estados procesales, etapas
    y fuero por ndd
    @sql_conn: conexion a la base de datos
    @lista_ndds: lista de ndds unicas a consultar
    """
    sql_query = text("""
                 SELECT DISTINCT(df.codfisc) AS NDD,
                 (
                     SELECT tp2_descripcion
                     FROM fiscalia.gen_tipos2 AS fue
                     WHERE df.fuero = fue.tp2_codigo) AS 'Fuero', CASE fso.fso_etapas WHEN 265 THEN CONCAT('INVESTIGACION PREVIA') WHEN 266 THEN CONCAT('INSTRUCCION FISCAL') WHEN 267 THEN CONCAT('PREPARATORIA DE JUICIO') WHEN 268 THEN CONCAT('JUICIO') WHEN 1065 THEN CONCAT('IMPUGNACION') ELSE 'SIN ACCIONES' END AS Etapa_Procesal, CASE WHEN tip2.tp2_descripcion IS NULL THEN 'POR APERTURAR INVETIGACION PREVIA' ELSE tip2.tp2_descripcion END AS Estado_Procesal
                     FROM fgn.denuncia_fiscalia df
                INNER JOIN fgn.fiscal_sorteado fso ON fso.ndd_codigo=df.codfisc
                LEFT JOIN fiscalia.gen_tipos2 tip2 ON tip2.tp2_codigo=fso.fso_estado_procesal
                WHERE df.estado=1 AND df.anulada='NO' AND fso.fso_estado<3 AND df.codfisc IN :ndds_list""")
    sql_query = sql_query.bindparams(ndds_list=tuple(lista_ndds))
    estadop = pd.read_sql(sql_query, sql_conn)
    estadop.rename(columns={'codfisc':'NDD'}, inplace=True)
    return estadop


def remove_prefix_analitica_columns(dataframe):
    """Devuelve los nombres de las columnas del dataset
    modificadas. Las columnas en el analitica empiezan con f_ o d_
    y el nombre sule estar en mayúsculas no tildadas

    Args:
        dataframe (dataframe): dataframe con la data para ser modificado los
        nombres de las columnas

    Returns:
        dictionary: diccionario con los pares clave valor para convertir
        los nombres de las columnas originales
    """
    new_colname = []
    for col in dataframe.columns:
        colh = re.findall(r"[df]_([\w_Ñ]+)", col, re.MULTILINE)[0]
        colh = str.upper(colh)
        new_colname.append(colh)
    colname_dict = dict(zip(dataframe.columns.to_list(), new_colname))
    dataframe.rename(columns=colname_dict, inplace=True)
    return colname_dict


def consultar_no_delitos(bdd_con, fecha_ini='2000-01-01'):
    """Consulta los no delitos desde una fecha inicial hasta la fecha actual
    
    Keyword arguments:
    bdd_con -- es el conector a la base de datos mysql
    fecha_ini -- es la fecha de inicio desde la que se ejecuta el query
    Return: return_description
    """
    
    query = text("""
                 SELECT a.avd_fecha_ini AS FECHA,
                    pro.pro_descripcion AS PROVINCIA,
                    can.can_descripcion AS CANTON,
                    edi.fxz_descripcion AS EDIFICIO,
                    IFNULL(nodel.dai_nodelito_descripcion,'ATENCION AL PUBLICO') AS NO_DELITO,
                    a.avd_abreviado AS ABREVIADO,
                    a.avd_hora_ini horaInicio,
                    (SELECT (usu.usu_nombre_completo) from fgn.gen_usuario usu where usu.usu_codigo=a.usu_codigo LIMIT 1) AS 'NOMBRE',
                    (SELECT (usu.usu_cedula) from fgn.gen_usuario usu where usu.usu_codigo=a.usu_codigo LIMIT 1) AS 'CEDULA'
                    from fgn.dai_registroatenciovictima_detalle a
                    INNER JOIN fiscalia.gen_provincia pro ON pro.pro_codigo=a.pro_codigo
                    INNER JOIN fiscalia.gen_canton can ON can.can_codigo=a.can_codigo
                    INNER JOIN fgn.fiscalias edi ON  edi.fxz_codigo=a.edi_codigo 
                    LEFT JOIN fgn.dai_nodelito nodel ON  nodel.dai_nodelito_abreviado=a.avd_abreviado
                    WHERE a.avd_fecha_ini BETWEEN :fecha_ini AND NOW()
                    AND a.avd_estado IN (1,2)
                 """)
    query = query.bindparams(fecha_ini=fecha_ini)
    nodelito_df = pd.read_sql(query, bdd_con)
    return nodelito_df


def consultar_actos_administrativos(bdd_con, fecha_ini='2000-01-01'):
    """consultar actos administrativos
    
    Keyword arguments:
    bdd_con -- conexion con la base de datos mysql
    fecha_ini -- string que indica la fecha de inicio para la recolección de datos 
    Return: dataframe con la información solicitada por la consulta sql
    """
    

    query = text("""
                 SELECT a.avd_fecha_ini AS 'FECHA_INICIO', 
                    b.pro_descripcion AS 'PROVINCIA', 
                    c.can_descripcion AS 'CANTON', 
                    fis.fxz_descripcion AS 'EDIFICIO', 
                    d.dai_actuacionAdministracion_descripcion AS 'DESCRIPCION_AVD', 
                    a.avd_abreviado, 
                    (SELECT (usu.usu_nombre_completo) from fgn.gen_usuario usu where usu.usu_codigo=a.usu_codigo LIMIT 1) nombreUsuario,
                    (SELECT (usu.usu_cedula) from fgn.gen_usuario usu where usu.usu_codigo=a.usu_codigo LIMIT 1) cedulaUsuario
                    from fgn.dai_registroatenciovictima_detalle a 
                    INNER JOIN fiscalia.gen_provincia b ON a.pro_codigo=b.pro_codigo
                    INNER JOIN fiscalia.gen_canton c ON a.can_codigo=c.can_codigo
                    INNER JOIN fgn.fiscalias fis ON fis.fxz_codigo=a.edi_codigo
                    INNER JOIN fgn.dai_actuacionadministrativa d ON a.avd_abreviado=d.dai_actuacionAdministracion_abreviatura
                    where
                    a.avd_fecha_ini BETWEEN :fecha_ini AND NOW()
                    and a.avd_estado!=0
                    AND a.avd_aa!=0;

    """
    )

    query = query.bindparams(fecha_ini=fecha_ini)
    actoadmin_df = pd.read_sql(query, bdd_con)
    return actoadmin_df


def devolver_involucrado_sencillo(x):
    """funcion que devuelve tres tipos de involucrados
    dependiendo del diccionario de involucrados,
    esta diseniada para ser usada dentro de un dataframe

    Args:
        x (str): valor que esta en un dataframe

    Returns:
        _str_: cambia el string de involucrado
    """
    for tipo_inv in dict_involucrados_tipo.keys():
        if x in dict_involucrados_tipo[tipo_inv]:
            return tipo_inv


def crear_columna_involucrado_sencillo(dataf, col_involucrado='TIPO_INVOLUCRADO'):
    """crear columna involucrado senccillo
    toma un dataframe de involucrados y genera la columna involucrado sencillo
    que resume los tipos de involucrados en 3 tipos denunciante, sospechoso y vicitma

    Args:
        dataf (_dataframe_): dataframe de involucrados
        col_involucrado (str, optional): nombre de la columna con los valores de tipo de inovlucrado. Defaults to 'TIPO_INVOLUCRADO'.
    """
    dataf['INVOLUCRADO_SENCILLO'] = dataf[col_involucrado].apply(lambda x: devolver_involucrado_sencillo(x))
    
    
def camel_case_string_noPoint(string):
    """
    This function permits to reformat the name of the columns of a dataframe
    in camel case style

    example: df.columns = [camel_case_string(x) for x in df.columns]
    """
    string =  re.sub(r"(_|-|\.)+", " ", string).title().replace(" ", "")
    string = string[0].lower() + string[1:]
    return string