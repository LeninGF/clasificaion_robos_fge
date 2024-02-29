"""
This script is going to be designed to be a generalization 
of the work carried out in predictFgeRobberyValidSeguRobosMallaOnlyNews_Mensual.ipynb

Coder: LeninGF
Date: 2024-02-16


"""
# TO DOs
# 1. automatizar el sql para poder escoger la tabla sobre la cual realizar las predicciones
# 2. automatizar para que exista la operacion de sampling en el sql
# 3. Se podría automatizar para ingresar las direcciones de los modelos en vez de usar por defecto
# 4. autmatizar el nombrado de la tabla y archivos csv que se guardan en la tabla o sql

import pandas as pd
import os
import numpy as np
import re
from time import time
from sqlalchemy import text, create_engine
from src.utils import extraer_relato, conectar_sql, save_df_in_sql
from src.utils import format_crimestory
from src.utils import words_qty, predict_text_class_tqdm, predict_text_class_only_new_tqdm
from src.utils import predictLabelAndScore
from datasets import Dataset
from src.utils import load_text_classification_model
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm


PATH_MODEL_SEGUIMIENTOS = '/home/falconiel/ML_Models/robbery_tf20221113'
PATH_MODEL_VALIDADOS = '/home/falconiel/ML_Models/robbery_tf20230213'
model_ckpt = "distilbert-base-multilingual-cased"
SEQ_LEN = 400
THRESHOLD_WORDS_QTY = 50
DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT = {'predictions':'predictionsDelitosSeguimiento',
                                            'label':'labelDelitosSeguimiento',
                                            'score':'scoreDelitosSeguimiento'}
                                            
DELITOS_VALIDADOS_COLUMNS_NAMES_DICT = {'predictions':'predictionsDelitosValidados',
                                        'label':'labelDelitosValidados',
                                        'score':'scoreDelitosValidados'}


def read_all_database_with_crime_story():
    query = text("""SELECT 
                    robos.NDD,
                    robos.Tipo_Delito_PJ as 'Tipo_Delito_PJ_comision',
                    robos.delitos_seguimiento as 'delitos_seguimiento_comision',
                    robos.delitos_validados as 'delitos_validados_comision',
                    -- den.infraccion,
                    gen.gen_delito_tipopenal,
                    gen.gen_delito_numart,
                    robos.Fecha_Registro,
                    (select min(dh.fecha) FROM fgn.denuncia_fiscalia dh where dh.codfisc = robos.NDD) as FECHA_PS,
                    (select max(den.fecha) from fgn.denuncia_fiscalia den) as 'FechaCorte',
                    bdd_enlace_externo.fnStripTags(den.obserinc) AS 'RELATO_SIAF'
                    FROM reportes.robos_2014_08012023 robos 
                    LEFT JOIN fgn.denuncia_fiscalia AS den ON robos.NDD = den.codfisc
                    LEFT JOIN fgn.gen_delitos as gen on gen.gen_delito_secuencia = den.infraccion
                    WHERE den.estado=1 
                    AND den.anulada='NO'
                    AND robos.Tipo_Delito_PJ = 'ROBO'
                    -- LIMIT 1000;
                    """)
    return pd.read_sql(query, conectar_sql())


def read_csv(file_path, extension='.csv', read_sample=False):
    try:
        # Check if the file is a CSV file
        if file_path.endswith(extension):
            if read_sample:
                df = pd.read_csv(file_path, nrows=1000, converters={'NDD':str})
            else:
                df = pd.read_csv(file_path, converters={'NDD':str})
            print(f"Archivo leído exitosamente: {df.shape}, sampling:{read_sample}")
            return df
        else:
            print("Tipo de archivo inválido. Por favor provea de un archivo CSV.")
    except Exception as e:
        print(f"El siguiente error ocurrió: {e}")


def read_sav(file_path, extension='.sav', read_sample=False):
    try:
        # Check if the file is a CSV file
        if file_path.endswith(extension):
            df = pd.read_spss(file_path)
            if read_sample:
                df = df.head(1000)
            print(f"Archivo leído exitosamente: {df.shape}, sampling:{read_sample}")
            return df
        else:
            print("Tipo de archivo inválido. Por favor provea de un archivo CSV.")
    except Exception as e:
        print(f"El siguiente error ocurrió: {e}")


def predict_robbery_class(dataframe, 
                          path_model, 
                          label_name, 
                          label_relato, 
                          score_label, 
                          words_qty_label,
                          new_ndds_list, 
                          model_name):
    print(f"Modelo utilizado: {path_model}")
    print("Cargando modelo...")
    model = load_text_classification_model(path2model=path_model,
                                            seq_len=SEQ_LEN,
                                            threshold_words_qty=THRESHOLD_WORDS_QTY)
    print(f"SEQ_LEN:{SEQ_LEN}\nTHRESHOLD_WORDS_QTY:{THRESHOLD_WORDS_QTY}")
    print(f"Inicia prediccion segun modelo {model_name} a las {datetime.now()}")
    time_start = time()
    predict_text_class_only_new_tqdm(dataf=dataframe,
                                     model=model,
                                     label_relato=label_relato,
                                     label_name= label_name, #'delitos_seguimiento_predicted',
                                     score_label=score_label,
                                     ndd_label="NDD",
                                     new_ndds_list=new_ndds_list,
                                     words_qty_label=words_qty_label,
                                     threshold_words_qty=THRESHOLD_WORDS_QTY)
    time_end = time()
    print(f"Predicción {model_name} concluida {datetime.now()}\nDuración: {time_end-time_start} segundos")
    
    
def preprocessing(df, sql_connection, source_type):
    print("Descargando relatos en SIAF...")
    relatos_siaf = extraer_relato(sql_connection=sql_connection, lista_ndds=list(set(df.NDD.to_list())))
    relatos_siaf.rename(columns={"RELATO": "relato_siaf"}, inplace=True)
    print(relatos_siaf.columns)
    df = pd.merge(df, relatos_siaf, on="NDD", how="left")
    
    if source_type == "sav":
        # when the file comes from sav there is a preloaded relato that is sent and compared 
        # if the file is not sav and there is no previous relato in dataframe, all relatos willbe downloaded and the next condition skipped
        if len(df[df.relato.isna()]) != len(df[df.relato_siaf.isna()]):
            print("No existe la misma cantidad de relatos en SIAF que en el archivo. Se procesa por defecto los relatos del archivo suministrado")
        print("Preprocesando texto")
        format_crimestory(relato_label='relato', dataf=df) # viene del archivo spss
        words_qty(dataf=df, relato_label='relato')
    elif source_type == "csv":
        print("Preprocesando Texto")
        format_crimestory(relato_label='relato_siaf', dataf=df)
        words_qty(dataf=df, relato_label='relato_siaf')
        df.rename(columns={'relato_siaf':'relato'}, inplace=True)
    else:
        print(f"Extension de archivo no conocida {source_type}")
    df.rename(columns={'d_CANTIDAD_PALABRAS':'cantidad_palabras'}, inplace=True)
    print("Columnas del dataset {}".format(df.columns))
    print(f"Características estadísticas de la cantidad de palabras:\n{df.cantidad_palabras.describe()}")
    return df


def read_previous_table(previous_table, sql_connection):
    database, table = previous_table.split('.')
    print(f"Leyendo tabla anterior desde: {database}.{table}")
    query = f"select * from {database}.{table}"
    xtest_anterior = pd.read_sql(query, sql_connection)
    print(f"Tamaño del dataset anterior {xtest_anterior.shape}")
    ndds_anterior_vacias = xtest_anterior[
    xtest_anterior.delitos_seguimiento.isna()].NDD.to_list()
    print(f"Existen {len(ndds_anterior_vacias)} Ndds vacias en el dataset anterior")
    return xtest_anterior, ndds_anterior_vacias


def print_robbery_kinds_qty(df, predicted_label):
    print(f"Cantidad de categorias de {predicted_label}:\n{df[predicted_label].value_counts()}")
    print(f"Cantidad de categorias vacias {predicted_label}:{df[predicted_label].isna().sum()}")
    print(f"Total de registros: {df[predicted_label].value_counts().sum()}")

# def check4missingpredictions(previous_df,
#                              model,
#                              predicted_label,score_label,
#                              relato_label = "relato",
#                              words_qty_label="cantidad_palabras",
#                              ndds_label="NDD"):
#     if previous_df[predicted_label].isna().sum() > 0:
#         print(f"Existen Ndds sin prediccion de {predicted_label} en el dataset anterior")
#         unpredicted_ndds_previous_set_list = previous_df[previous_df[predicted_label].isna()][ndds_label].to_list()
#         index_vacias = previous_df.index[previous_df[ndds_label].isin(unpredicted_ndds_previous_set_list)].to_list()
#         for i in tqdm(index_vacias):
#             if previous_df[words_qty_label].iloc[i] >= THRESHOLD_WORDS_QTY:
#                 delito_i, score_i = predictLabelAndScore(
#                     previous_df[relato_label].iloc[i], model
#                 )
#             else:
#                 delito_i, score_i = ("N/A", 0)
    
#             previous_df[predicted_label].iloc[i] = delito_i
#             previous_df[score_label].iloc[i] = score_i
    

def check4missingpredictions(previous_df,
                             path_model,
                             predicted_label,
                             score_label,
                             relato_label = "relato",
                             words_qty_label="cantidad_palabras",
                             ndds_label="NDD"):
    missing_predictions = previous_df[predicted_label].isna()
    if missing_predictions.sum() > 0:
        model = load_text_classification_model(path2model=path_model,
                                            seq_len=SEQ_LEN,
                                            threshold_words_qty=THRESHOLD_WORDS_QTY)
        print(f"Existen Ndds sin prediccion de {predicted_label} en el dataset anterior")
        unpredicted_ndds_previous_set_list = previous_df.loc[missing_predictions, ndds_label].tolist()
        unpredicted_indices = previous_df.loc[previous_df[ndds_label].isin(unpredicted_ndds_previous_set_list)].index
        for i in tqdm(unpredicted_indices):
            if previous_df.loc[i, words_qty_label] >= THRESHOLD_WORDS_QTY:
                prediction, score = predictLabelAndScore(previous_df.loc[i, relato_label], model)
            else:
                prediction, score = ("N/A", 0)
    
            previous_df.loc[i, predicted_label] = prediction
            previous_df.loc[i, score_label] = score
    else:
        print(f"No existen casos anteriores sin etiqueta de {predicted_label}")


def remove_columns(df, columns2remove_list):
    try:
        df.drop(columns=columns2remove_list, axis=1, inplace=True)
    except KeyError:
        print(f"No fue posible eliminar las columnas: {columns2remove_list}. Revise los nombres")


def merging_results(xtest_actual, xtest_anterior, label_ndd, new_ndds_list, columnas_prediccion, en_delitos_seguimiento, en_delitos_validados):
    # It must be executed when we work with previous data and not in a new prediction
    # looking for columns new that are not neccessary
    columns2remove = list(set(xtest_actual.columns).difference(xtest_anterior.columns))
    print(f"Columnas adicionales: {columns2remove}")
    remove_columns(df=xtest_actual, columns2remove_list=columns2remove)    
    # Setting FechaActualizacion in xtest_actual=xtest_df
    xtest_actual["FechaActualizacion"] = np.nan
    xtest_actual["FechaActualizacion"][xtest_actual[label_ndd].isin(new_ndds_list)] = datetime.now()
    xtest_merged = pd.merge(xtest_actual, 
                            xtest_anterior[columnas_prediccion[:-1]],
                            on="NDD",
                            how="left",
                            suffixes=["_actual", "_anterior"],
                            )
    print(f"dataset_actual: {xtest_actual.shape}\ndataset_anterior:{xtest_anterior.shape}\ndataset_merged:{xtest_merged.shape}")
    print(xtest_merged.columns)
    # ajuste de la fecha
    mask = ~xtest_merged[label_ndd].isin(new_ndds_list)
    if en_delitos_seguimiento:
        xtest_merged.loc[mask, 'delitos_seguimiento_actual'] = xtest_merged.loc[mask, 'delitos_seguimiento_anterior']
        # xtest_merged.delitos_seguimiento_actual[-xtest_merged[label_ndd].isin(new_ndds_list)] = xtest_merged.delitos_seguimiento_anterior
        # xtest_merged.score_delitos_seguimiento_actual[-xtest_merged[label_ndd].isin(new_ndds_list)] = xtest_merged.score_delitos_seguimiento_anterior
        xtest_merged.loc[mask, 'score_delitos_seguimiento_actual'] = xtest_merged.loc[mask, 'score_delitos_seguimiento_anterior']
    if en_delitos_validados:
        xtest_merged.loc[mask, 'delitos_validados_actual'] = xtest_merged.loc[mask, 'delitos_validados_anterior']
        # xtest_merged.delitos_validados_actual[-xtest_merged[label_ndd].isin(new_ndds_list)] = xtest_merged.delitos_validados_anterior
        # xtest_merged.score_delitos_validados_actual[-xtest_merged[label_ndd].isin(new_ndds_list)] = xtest_merged.score_delitos_validados_anterior
        xtest_merged.loc[mask, 'score_delitos_validados_actual'] = xtest_merged.loc[mask, 'score_delitos_validados_anterior']
    if en_delitos_validados and en_delitos_seguimiento:
        xtest_merged.loc[mask, 'compare_actual'] = xtest_merged.loc[mask, 'compare_anterior']
        # xtest_merged.compare_actual[-xtest_merged[label_ndd].isin(new_ndds_list)] = xtest_merged.compare_anterior
    
    # xtest_merged.FechaActualizacion_actual[-xtest_merged[label_ndd].isin(new_ndds_list)] = xtest_merged.FechaActualizacion_anterior
    xtest_merged.loc[mask, 'FechaActualizacion_actual'] = xtest_merged.loc[mask, 'FechaActualizacion_anterior']
    # Dropping columns
    if en_delitos_seguimiento and en_delitos_validados:
        xtest_merged.drop(
            columns=[
                "delitos_seguimiento_anterior",
                "delitos_validados_anterior",
                "score_delitos_seguimiento_anterior",
                "score_delitos_validados_anterior",
                "FechaActualizacion_anterior",
                "compare_anterior",
                ],
            inplace=True,
            )
    elif en_delitos_seguimiento:
        xtest_merged.drop(
            columns=[
                "delitos_seguimiento_anterior",
                # "delitos_validados_anterior",
                "score_delitos_seguimiento_anterior",
                # "score_delitos_validados_anterior",
                "FechaActualizacion_anterior",
                # "compare_anterior",
                ],
            inplace=True,
            )
    elif en_delitos_validados:
        xtest_merged.drop(
            columns=[
                # "delitos_seguimiento_anterior",
                "delitos_validados_anterior",
                # "score_delitos_seguimiento_anterior",
                "score_delitos_validados_anterior",
                "FechaActualizacion_anterior",
                # "compare_anterior",
                ],
            inplace=True,
            )
    else:
        print(f"No se puede retirar las columnas anteriores. Revise Linea 245 y las columnas del dataframe:{xtest_merged.columns}")
    # removing additional label "_actual"
    for col in xtest_merged.columns:
        xtest_merged.rename(columns={col: col.replace("_actual", "")}, inplace=True)   
    xtest_merged["compare"] = xtest_merged.apply(lambda x: "OK"
                                                 if x["delitos_seguimiento"] == x["delitos_validados"] 
                                                 else "CHECK", 
                                                 axis=1,)
    
    remove_columns(df=xtest_merged, columns2remove_list=['edad'])
    print(xtest_merged["compare"].value_counts())
    print(xtest_merged["FechaActualizacion"].value_counts())
    return xtest_merged


def print_words_qty_less_than_threshold(df, threshold, words_qty_label):
    less_than_threshold = df[df[words_qty_label]< threshold].shape[0]
    print(f"En el dataset existen {less_than_threshold} registros con cantidad de palabras menores a {threshold}")


def main(source,
         predict_all_ndds,
         predict_validados, 
         predict_seguimiento, 
         path_source,
         previous_table,
         sample,
         save2csv,
         save2sql,
         save2xlsx,
         save_path_files,
         save_sql_ddbb_table_name):
    print("Prediccion de Etiquetas Delitos Seguimiento y Delitos Validados Robos")
    print("Conectando a base de datos por favor espere...")
    conx = conectar_sql()
    if source == "sav":
        print("Leyendo datos desde Archivo de SPSS...")
        xtest_df = read_sav(file_path=path_source, read_sample=sample)
        print(f"Total de registros: {xtest_df.shape}")
    elif source == "csv":
        print("Leyendo datos desde Archivo de CSV...")
        xtest_df = read_csv(file_path=path_source, read_sample=sample)
        print(f"Total de registros: {xtest_df.shape}")
    else:
        print("Formato de archivo no registrado")
    
    # preprocessing
    xtest_df = preprocessing(df=xtest_df, sql_connection=conx, source_type=source)
    if predict_all_ndds:
        # predecir todas las ndds del archivo
        print("Ejecuntado prediccion sobre todas las Ndds...")
        new_ndds = xtest_df.NDD.to_list()
    else:
        # cargando datos anteriores de comision si se predice solo nuevos
        print("Ejecutando la prediccion sobre Ndds Nuevas...")
        xtest_anterior, lista_ndds_anteriores_sin_prediccion = read_previous_table(previous_table=previous_table, sql_connection=conx)
        new_ndds = list(set(xtest_df.NDD.to_list()).difference(xtest_anterior.NDD.to_list()))
        columnas_prediccion = list(set(xtest_anterior.columns).difference(xtest_df.columns))
        columnas_prediccion.extend(["NDD"])
        columnas_prediccion.extend(["FechaActualizacion"])
        print("======== VALORES DE CATEGORIS DEL DATASET ANTERIOR ========")
        print_robbery_kinds_qty(df=xtest_anterior, predicted_label='delitos_seguimiento')
        print_robbery_kinds_qty(df=xtest_anterior, predicted_label='delitos_validados')
        
    print(f"Existen un total de {len(new_ndds)} Ndds para prediccion")
    
    if predict_seguimiento:
        # predict seguimiento
        predict_robbery_class(
            dataframe=xtest_df,
            path_model=PATH_MODEL_SEGUIMIENTOS,
            label_name="delitos_seguimiento",
            label_relato="relato",
            score_label="score_delitos_seguimiento",
            words_qty_label="cantidad_palabras",
            new_ndds_list=new_ndds,
            model_name="delitos_seguimiento",
            )
        print_robbery_kinds_qty(df=xtest_df, predicted_label='delitos_seguimiento')
        # check for empty predictions of prior dataset from previous table
        if not predict_all_ndds:
            check4missingpredictions(previous_df=xtest_anterior,
                                     path_model=PATH_MODEL_SEGUIMIENTOS,
                                     predicted_label="delitos_seguimiento",
                                     score_label="score_delitos_seguimiento",
                                     relato_label="relato",
                                     words_qty_label="cantidad_palabras",
                                     ndds_label="NDD")
    if predict_validados:
        # predict validados
        predict_robbery_class(
            dataframe=xtest_df,
            path_model=PATH_MODEL_VALIDADOS,
            label_name="delitos_validados",
            label_relato="relato",
            score_label="score_delitos_validados",
            words_qty_label="cantidad_palabras",
            new_ndds_list=new_ndds,
            model_name="delitos_validados",
            )
        print_robbery_kinds_qty(df=xtest_df, predicted_label='delitos_validados')
        # check for empty predictions of prior dataset from previous table    
        if not predict_all_ndds:
            check4missingpredictions(previous_df=xtest_anterior,
                                     path_model=PATH_MODEL_VALIDADOS,
                                     predicted_label="delitos_validados",
                                     score_label="score_delitos_validados",
                                     relato_label="relato",
                                     words_qty_label="cantidad_palabras",
                                     ndds_label="NDD")    
    
    # Merging datasets
    # TO-DO A comparison between delitos_seguimiento and delitos_validados is no longer necessary it must be deprecated
    if predict_seguimiento and predict_validados:
        xtest_df["compare"] = xtest_df.apply(
            lambda x: "OK" 
            if x["delitos_seguimiento"] == x["delitos_validados"] 
            else "CHECK", 
            axis=1,)
    
    if predict_all_ndds:
        xtest_merged = xtest_df
    else:
        print(xtest_df.shape, xtest_anterior.shape)
        xtest_merged = merging_results(xtest_actual=xtest_df,
                                       xtest_anterior=xtest_anterior,
                                       label_ndd="NDD",
                                       new_ndds_list=new_ndds,
                                       columnas_prediccion=columnas_prediccion,
                                       en_delitos_seguimiento=predict_seguimiento,
                                       en_delitos_validados=predict_validados)
    # print(xtest_merged.shape)
    # print(xtest_merged.columns)
    if predict_seguimiento:
        print(print_robbery_kinds_qty(df=xtest_merged,predicted_label='delitos_seguimiento'))
    if predict_validados:
        print(print_robbery_kinds_qty(df=xtest_merged, predicted_label='delitos_validados'))
    print_words_qty_less_than_threshold(df=xtest_merged,
                                        threshold=THRESHOLD_WORDS_QTY,
                                        words_qty_label='cantidad_palabras')
    # ++++ saving results ++++
    name_of_original_file = path_source.split("/")[-1].split(".")[0]
    output_file_name = save_path_files+"prediccionesDelitosSeguimientoValidados_"+name_of_original_file+ "_"+ datetime.now().strftime("%Y-%m-%d")
    if save2sql:
        if save_sql_ddbb_table_name is None:
            pattern = r'INEC_(\d+)_(\d+)(?:.*)'
            match = re.search(pattern, name_of_original_file)
            if match:
                fechaini = match.group(1)
                fechafin = match.group(2)
                table = "robosAI_"+fechaini[6:] + fechaini[4:6] + fechaini[2:4]+"_"+fechafin[6:] + fechafin[4:6] + fechafin[2:4]
            else:
                table = "RobosAI"
            database = "reportes"
            save_df_in_sql(name_table=table, database=database, dataf=xtest_merged)
        else:
            database, table = save_sql_ddbb_table_name.split('.')
            save_df_in_sql(name_table=table, database=database, dataf=xtest_merged) 
        print(f"Resultados guardados en {database}.{table}")
    if save2xlsx:
        print("Escribi el archivo a xlsx puede generar errores e interrumpirse")
        xlsx_out = output_file_name+".xlsx"
        writer = pd.ExcelWriter(xlsx_out, engine="xlsxwriter")
        xtest_merged.to_excel(writer, sheet_name="RobosAI")
        writer.close()
        print(f"Resultados guardados en {xlsx_out}")
    if save2csv:
        csv_out = output_file_name+".csv"
        xtest_merged.to_csv(csv_out, index=False)
        print(f"Resultados guardados en {csv_out}")

    # print("##### FIN #####")
    return 0

if __name__=="__main__":
    parser = ArgumentParser(prog='predict_all_database_robos.py',
                            description="Este programa realiza la predicición de las etiquetas de delitos seguimiento y \
                            delitos_validados, sobre la base de datos de la comsión de 2014 a 2022 con fecha de corte \
                            8 de enero de 2023. Por defecto, el programa predice la etiqueta de delitos seguimiento. Para \
                            predecir delitos validados se debe ingresar el parámetro correspondiente en la línea de comandos.\
                            Por defecto los resultados se guardan en archivos csv a disco duro.\nDesarrollado por Lenin G. Falconi (lenin.g.falconi@gmail.com)",
                            add_help=True)
    
    parser.add_argument('--sample', action="store_true", help="Si se declara, consulta los 1.000 registros de la base para pruebas")
    parser.add_argument('--seguimiento', action="store_true", help="Si se declara, realiza la predicción de delitos seguimiento")
    parser.add_argument('--validados', action="store_true", help="Si se declara, realiza la predicción de etiquetas de delitos validados")
    parser.add_argument('--full_predict', action="store_true", help="Si se declara, realiza la predicción de todos los casos suministrados en el archivo,\
    caso contario sólo predice los casos nuevos. Por defecto solo predice casos nuevos. Se debe activar la prediccion para delitos_seguimiento o delitos_validados")
    parser.add_argument('--source', default='sav', help="Especifica el tipo de fuente de origen de los datos entre csv, sql o sav")
    parser.add_argument('--path_source', help="Especifica la ubicacion del archivo que contiene los datos fuente")
    parser.add_argument('--previous_table', default='reportes.robosAI', help="Especifica la base y \
    la tabla donde reposan los datos anteriores a comparar a menos \
    que se pida predicción completa. El formato es BASE.TABLA")
    parser.add_argument('--save2sql', action="store_true", help="Si se declara, se guarda los resultados obtenidos en tabla de SQL. Por defecto guarda en [DATABASE].[TABLE]")
    parser.add_argument('--save2csv', action="store_true", help="Si se declara, se guarda los resultados obtenidos en archivo CSV. La ubicacion se declara en save_path_files")
    parser.add_argument('--save2xlsx', action="store_true", help="Si se declara, se guarda los resultados obtenidos en archivo XLSX. La ubicacion se declara en save_path_files")
    parser.add_argument('--save_path', default='data/processed/', help="Especifica la ubicacion en que se guardaran los resultados obtenidos")
    parser.add_argument('--sql_table_name', help="Especifica la base y la tabla donde se guardaran los resultados. El formato es BASE.TABLA. \
    Si no se declara, el programa calcula automaticamente un nombre siempre que el nombre del archivo fuente tenga el formato: \
    INEC_AAAAMMDD_AAAAMMDD_AAAAMMDD_.*")
    
    args = parser.parse_args()
    main(predict_validados=args.validados,
         predict_seguimiento=args.seguimiento,
         source=args.source,
         path_source=args.path_source,
         predict_all_ndds=args.full_predict,
         previous_table=args.previous_table,
         sample=args.sample,
         save2csv=args.save2csv,
         save2xlsx=args.save2xlsx,
         save2sql=args.save2sql,
         save_path_files=args.save_path,
         save_sql_ddbb_table_name=args.sql_table_name)


"""
An  example on how to execute the program
python predictRobberyClass_fromSource.py --seguimiento --validados --path_source data/raw/requests/INEC_20230101_20240208_20240209_MALLA_ROBO_0123_0124.sav --source sav --previous_table reportes.robosAI_010122_100124 --save2xlsx --save2sql

"""


# Report dictionary structure
# data = {
#     'execution_data': {
#         'date': str(datetime.now()),
#         'total_registers': 100,  # replace with actual value
#     },
#     'model_performance': [
#         {
#             'model_name': 'Model 1',  # replace with actual model name
#             'time_started': '2024-01-01 00:00:00',  # replace with actual time
#             'time_ended': '2024-01-01 01:00:00',  # replace with actual time
#             'duration': '1 hour',  # replace with actual duration
#         },
#         {
#             'model_name': 'Model 2',  # replace with actual model name
#             'time_started': '2024-01-01 02:00:00',  # replace with actual time
#             'time_ended': '2024-01-01 03:00:00',  # replace with actual time
#             'duration': '1 hour',  # replace with actual duration
#         },
#     ],
#     'file_locations': {
#         'csv_file': '/path/to/csv/file',  # replace with actual path
#         'xlsx_file': '/path/to/xlsx/file',  # replace with actual path
#         'sql_table': 'database_name.table_name',  # replace with actual table location
#     },
#     'total_unpredicted_values': 10,  # replace with actual value
# }

# to_save = os.path.join(os.getcwd(), 'reports', 'robos_2014_08012023_relatos.csv')
# print(f"Guardando archivo con Relatos a disco en {to_save}")
# # El archivo generado puede ser enviado a colab para usar GPU
# xtest_df.to_csv(to_save, index=False)
# # ahora se debe realizar las predicciones de acuerdo a las caracteristicas
# # limitantes: que sea ndd de robos y con un relato de 50 palabras
# time_report_dict = {}
#     time_report_dict['modelo_validados'] = time_end-time_start
# time_report_dict['modelo_seguimientos'] = time_end-time_start
# # salida del programa        
# # print(f"Predicciones Terminadas: {time_report_dict['modelo_validados']+time_report_dict['modelo_seguimientos']}")
# # print(time_report_dict.items())
# to_save = os.path.join(os.getcwd(), 'reports', 'robos_2014_08012023_predicted.csv')
# print(f"Salvando resultados a csv {to_save}")
# xtest_df.to_csv(to_save, index=False)

# if sql:
#     name_table = 'robos_2014_08012023_predicted'
#     print(f"Guardando en sql {name_table}")
#     save_df_in_sql(name_table=name_table, dataf=xtest_df)