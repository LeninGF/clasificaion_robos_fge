# TO DOs
# 1. automatizar el sql para poder escoger la tabla sobre la cual realizar las predicciones
# 2. automatizar para que exista la operacion de sampling en el sql
# 3. Se podría automatizar para ingresar las direcciones de los modelos en vez de usar por defecto
# 4. autmatizar el nombrado de la tabla y archivos csv que se guardan en la tabla o sql, es decir definir por el usuario
# 5. Generalizar para que se pueda escoger la base de datos en la que trabajar

import pandas as pd
import os
from time import time
from sqlalchemy import text, create_engine
from src.utils import extraer_relato, conectar_sql, save_df_in_sql
from src.utils import format_crimestory
from src.utils import words_qty, predict_text_class_DaaS_tqdm
from datasets import Dataset
from src.utils import load_text_classification_model
from datetime import datetime
from argparse import ArgumentParser


PATH_MODEL_SEGUIMIENTOS = '/home/falconiel/ML_Models/robbery_tf20221113'
PATH_MODEL_VALIDADOS = '/home/falconiel/ML_Models/robbery_validados_tf20231211'
model_ckpt = "distilbert-base-multilingual-cased"
SEQ_LEN = 400
THRESHOLD_WORDS_QTY = 50
# base de datos que contiene la tabla desde la que se va a leer los datos
DATABASE_FROM = 'DaaS'  # generalizar para que apunte a cualquier Base de Datos
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
    
    
def read_daas_robosML(sample, table_in, database_in):
    # query = "select * from `DaaS`.`robosML`"
    query = f"select * from `{database_in}`.`{table_in}`"
    if sample:
        query += " limit 1000"
    query += ";"
    query = text(query)
    daas_df = pd.read_sql(query, conectar_sql())
    print(f"Total de registros: {daas_df.shape}")
    # dando formato al relato de la noticia del delito
    format_crimestory(relato_label='RELATO', dataf=daas_df)
    # genera la cuenta de cantidad de palabras, en columna d_CANTIDAD_PALABRAS
    words_qty(dataf=daas_df, relato_label='RELATO')
    # puede ser adecuado omitir esto luego pero podemos sobrescribir la columna \
    # cantidad de palabras original para luego hacer un drop de esa columna y no
    # afectar la estructura de la tabla original.
    daas_df['CANTIDAD_PALABRAS'] = daas_df['d_CANTIDAD_PALABRAS']
    print("Columnas del dataset {}".format(daas_df.columns))
    print(f"Características de la Canidad de palabras\n:{daas_df.d_CANTIDAD_PALABRAS.describe()}")
    # hacer un drop de d_CANTIDAD_PALABRAS????
    return daas_df, 'RELATO'


def main(predict_delitos_validados,csv,sql, sample, table_in, update):
    print(f"Prediccion de Etiquetas Delitos Seguimiento y Delitos Validados Robos en {DATABASE_FROM}.{table_in}")
    xtest_df, LABEL_RELATO = read_daas_robosML(sample=sample, table_in=table_in, database_in=DATABASE_FROM)
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
        predict_text_class_DaaS_tqdm(dataf=xtest_df,
                                    model=modelo_delitos_validados,
                                    label_relato=LABEL_RELATO,
                                    label_name='delitos_validados_predicted',
                                    words_qty_label='d_CANTIDAD_PALABRAS',
                                    threshold_words_qty=THRESHOLD_WORDS_QTY,
                                    status='ESTADO_ML')
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
    predict_text_class_DaaS_tqdm(dataf=xtest_df,
                                 model=modelo_delitos_seguimiento,
                                 label_relato=LABEL_RELATO,
                                 label_name='delitos_seguimiento_predicted',
                                 label_score='delitos_seguimiento_predicted_SCORE',
                                 words_qty_label='d_CANTIDAD_PALABRAS',
                                 threshold_words_qty=THRESHOLD_WORDS_QTY,
                                 status='ESTADO_ML')
    time_end = time()
    print(f"Predicción Delitos Seguimiento concluida {datetime.now()}\nDuración: {time_end-time_start} ")
    time_report_dict['modelo_seguimientos'] = time_end-time_start
    xtest_df['FechaActualizacionDelitosSeguimiento'] = datetime.now()
    # salida del programa        
    # print(f"Predicciones Terminadas: {time_report_dict['modelo_validados']+time_report_dict['modelo_seguimientos']}")
    # print(time_report_dict.items())
    if csv:
        to_save = os.path.join(os.getcwd(), 'reports', 'DaaS_RobosML_predicted.csv')
        print(f"Salvando resultados a csv {to_save}")
        xtest_df.to_csv(to_save, index=False)
    if sql:
        table_out = table_in+'_predicted_tmp'
        print(f"Guardando en base de datos DaaS en tabla sql {table_out}")
        save_df_in_sql(name_table=table_out, dataf=xtest_df, database='DaaS')
        # implementar la actualizacion de la tabla
        if update:
            query = f"""UPDATE {DATABASE_FROM}.{table_in} del1
                       INNER JOIN {DATABASE_FROM}.{table_out} del2
                       ON del1.NDD = del2.NDD
                       SET del1.delitos_seguimiento_predicted = del2.delitos_seguimiento_predicted,
                       del1.delitos_seguimiento_predicted_SCORE = del2.delitos_seguimiento_predicted_SCORE,
                       del1.ESTADO_ML = del2.ESTADO_ML,
                       del1.FechaActualizacionDelitosSeguimiento = del2.FechaActualizacionDelitosSeguimiento;"""

            engine_maria_db = create_engine(f"mysql+pymysql://falconiel:BebuSuKO@192.168.152.197:3306/{DATABASE_FROM}"+"?charset=utf8mb4")
            with engine_maria_db.connect() as conn:
                conn.execute(query)
        
    print("##### FIN #####")
    return 0

if __name__=="__main__":
    parser = ArgumentParser(prog='predict_daasRobosML.py',
                            description="Este programa realiza la predicición de las etiquetas de delitos seguimiento y \
                            delitos_validados, sobre la tabla DaaS.robosML de la FGE. \
                            Por defecto, el programa predice la etiqueta de delitos seguimiento. Para \
                            predecir delitos validados se debe ingresar --validados en la línea de comandos.\
                            El programa procesa sólo aquellos que tienen EstadoML=0. Por defecto guarda en SQL.\
                            Para guardar en archivo declarar --csv",
                            add_help=True)
    
    parser.add_argument('--tablein', default='robosML', type=str, help='Nombre de la tabla sql de la que se debe leer los datos')
    parser.add_argument('--sample', action="store_true", help="Si se declara, consulta los 1.000 registros de la base para pruebas")
    # parser.add_argument('--seguimiento', action="store_true", help="Si se declara, realiza la predicción de delitos seguimiento")
    parser.add_argument('--validados', action="store_true", help="Si se declara, realiza la predicción de etiquetas de delitos validados")
    parser.add_argument('--csv', action="store_true", help="Si se declara, se guarda los resultados obtenidos en archivo csv")
    parser.add_argument('--sql', action="store_true", help="Si se declara, se guarda los resultados obtenidos tabla de SQL temporal. Por defecto guarda en 192.168.152.197")
    parser.add_argument('--update', action="store_true", help="Si se declara, realiza la actualizacion de la base de datos original leida")
    args = parser.parse_args()
    main(predict_delitos_validados=args.validados, csv=args.csv, sample=args.sample, table_in=args.tablein, sql=args.sql, update=args.update)