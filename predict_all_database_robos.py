# TO DOs
# 1. automatizar el sql para poder escoger la tabla sobre la cual realizar las predicciones
# 2. automatizar para que exista la operacion de sampling en el sql
# 3. Se podría automatizar para ingresar las direcciones de los modelos en vez de usar por defecto
# 4. autmatizar el nombrado de la tabla y archivos csv que se guardan en la tabla o sql

import pandas as pd
import os
from time import time
from sqlalchemy import text, create_engine
from src.utils import extraer_relato, conectar_sql, save_df_in_sql
from src.utils import format_crimestory
from src.utils import words_qty, predict_text_class_tqdm
from datasets import Dataset
from src.utils import load_text_classification_model
from datetime import datetime
from argparse import ArgumentParser


PATH_MODEL_SEGUIMIENTOS = '/home/falconiel/ML_Models/robbery_tf20221113'
PATH_MODEL_VALIDADOS = '/home/falconiel/ML_Models/robbery_tf20230213'
model_ckpt = "distilbert-base-multilingual-cased"
SEQ_LEN = 300
THRESHOLD_WORDS_QTY = 50
# DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT = {'predictions':'predictionsDelitosSeguimiento',
#                                             'label':'labelDelitosSeguimiento',
#                                             'score':'scoreDelitosSeguimiento'}
                                            
# DELITOS_VALIDADOS_COLUMNS_NAMES_DICT = {'predictions':'predictionsDelitosValidados',
#                                         'label':'labelDelitosValidados',
#                                         'score':'scoreDelitosValidados'}


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


def main(predict_delitos_validados, sql):
    print("Prediccion de Etiquetas Delitos Seguimiento y Delitos Validados Robos")
    xtest_df = read_all_database_with_crime_story()
    print(f"Total de registros: {xtest_df.shape}")
    format_crimestory(relato_label='RELATO_SIAF', dataf=xtest_df)
    words_qty(dataf=xtest_df, relato_label='RELATO_SIAF')
    print("Columnas del dataset {}".format(xtest_df.columns))
    print(f"Características de la Canidad de palabras\n:{xtest_df.d_CANTIDAD_PALABRAS.describe()}")
    to_save = os.path.join(os.getcwd(), 'reports', 'robos_2014_08012023_relatos.csv')
    print(f"Guardando archivo con Relatos a disco en {to_save}")
    # El archivo generado puede ser enviado a colab para usar GPU
    xtest_df.to_csv(to_save, index=False)
    # ahora se debe realizar las predicciones de acuerdo a las caracteristicas
    # limitantes: que sea ndd de robos y con un relato de 50 palabras
    time_report_dict = {}
    if predict_delitos_validados:
        print(f"Cargando Modelo Delitos Validados: {PATH_MODEL_VALIDADOS}")
        # carga de modelo de delitos validados
        modelo_delitos_validados = load_text_classification_model(path2model=PATH_MODEL_VALIDADOS,
                                                                  seq_len=SEQ_LEN,
                                                                  threshold_words_qty=THRESHOLD_WORDS_QTY)
        print(f"SEQ_LEN:{SEQ_LEN}\n THRESHOLD_WORDS_QTY:{THRESHOLD_WORDS_QTY}")
        print(f"Ejecuntando prediccion Delitos Validados {datetime.now()}")
        time_start = time()
        predict_text_class_tqdm(dataf=xtest_df,
                           model=modelo_delitos_validados,
                           label_relato='RELATO_SIAF',
                           label_name='delitos_validados_predicted',
                           words_qty_label='d_CANTIDAD_PALABRAS',
                           threshold_words_qty=THRESHOLD_WORDS_QTY)
        time_end = time()
        print(f"Predicción Delitos Validados concluida {datetime.now()}\nDuración: {time_end-time_start} ")
        time_report_dict['modelo_validados'] = time_end-time_start
        xtest_df['FechaActualizacionDelitosValidados'] = datetime.now()
    # predicción por defecto de delitos seguimiento
    
    print(f"Cargando Modelo Delitos Seguimiento: {PATH_MODEL_SEGUIMIENTOS}")
    modelo_delitos_seguimiento = load_text_classification_model(path2model=PATH_MODEL_SEGUIMIENTOS,
                                                                seq_len=SEQ_LEN,
                                                                threshold_words_qty=THRESHOLD_WORDS_QTY)
    print(f"SEQ_LEN:{SEQ_LEN}\n THRESHOLD_WORDS_QTY:{THRESHOLD_WORDS_QTY}")
    print(f"Ejecutando prediccion Delitos Seguimiento {datetime.now()}")
    time_start = time()
    predict_text_class_tqdm(dataf=xtest_df,
                       model=modelo_delitos_seguimiento,
                       label_relato='RELATO_SIAF',
                       label_name='delitos_seguimiento_predicted',
                       words_qty_label='d_CANTIDAD_PALABRAS',
                       threshold_words_qty=THRESHOLD_WORDS_QTY)
    time_end = time()
    print(f"Predicción Delitos Seguimiento concluida {datetime.now()}\nDuración: {time_end-time_start} ")
    time_report_dict['modelo_seguimientos'] = time_end-time_start
    xtest_df['FechaActualizacionDelitosSeguimiento'] = datetime.now()
    # salida del programa        
    # print(f"Predicciones Terminadas: {time_report_dict['modelo_validados']+time_report_dict['modelo_seguimientos']}")
    # print(time_report_dict.items())
    to_save = os.path.join(os.getcwd(), 'reports', 'robos_2014_08012023_predicted.csv')
    print(f"Salvando resultados a csv {to_save}")
    xtest_df.to_csv(to_save, index=False)
    
    if sql:
        name_table = 'robos_2014_08012023_predicted'
        print(f"Guardando en sql {name_table}")
        save_df_in_sql(name_table=name_table, dataf=xtest_df)
    
    print("##### FIN #####")
    return 0

if __name__=="__main__":
    parser = ArgumentParser(prog='predict_all_database_robos.py',
                            description="Este programa realiza la predicición de las etiquetas de delitos seguimiento y \
                            delitos_validados, sobre la base de datos de la comsión de 2014 a 2022 con fecha de corte \
                            8 de enero de 2023. Por defecto, el programa predice la etiqueta de delitos seguimiento. Para \
                            predecir delitos validados se debe ingresar el parámetro correspondiente en la línea de comandos.\
                            Por defecto los resultados se guardan en archivos csv a disco duro",
                            add_help=True)
    
    parser.add_argument('--sample', action="store_true", help="Si se declara, consulta los 1.000 registros de la base para pruebas")
    # parser.add_argument('--seguimiento', action="store_true", help="Si se declara, realiza la predicción de delitos seguimiento")
    parser.add_argument('--validados', action="store_true", help="Si se declara, realiza la predicción de etiquetas de delitos validados")
    parser.add_argument('--sql', action="store_true", help="Si se declara, se guarda los resultados obtenidos tabla de SQL. Por defecto guarda en 192.168.152.197")
    
    args = parser.parse_args()
    main(predict_delitos_validados=args.validados, sql=args.sql)