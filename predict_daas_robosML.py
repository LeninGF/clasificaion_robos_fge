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
from src.utils import seconds_to_readable_time, print_robbery_kinds_qty
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
# DATABASE_FROM = 'DaaS'  # generalizar para que apunte a cualquier Base de Datos
TABLE_FROM = 'robosML'
# Los siguientes nombres deben guardar con la declaracion que tienen en la base de datos
DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT = { 'label_name':'delitos_seguimiento_predicted',
                                            'label_score':'delitos_seguimiento_predicted_score',
                                            'label_fecha': 'FechaActualizacionDelitosSeguimiento'
                                            }
                                            
DELITOS_VALIDADOS_COLUMNS_NAMES_DICT = {'label_name':'delitos_validados_predicted',
                                        'label_score':'delitos_validados_predicted_score',
                                        'label_fecha':'FechaActualizacionDelitosValidados'
                                        }


def predict_robbery_class_daas(dataframe, 
                          path_model,
                          label_relato, 
                          label_name, 
                          label_score, 
                          words_qty_label, 
                          status,
                          fecha_label, 
                          model_name,
                          ):
    print(f"Cargando Modelo {model_name} desde: {path_model}")
    model = load_text_classification_model(path2model=path_model,
                                           seq_len=SEQ_LEN,
                                           threshold_words_qty=THRESHOLD_WORDS_QTY)
    print(f"SEQ_LEN:{SEQ_LEN}\nTHRESHOLD_WORDS_QTY:{THRESHOLD_WORDS_QTY}")
    print(f"Inicia ejecución de prediccion con modelo {model_name}: {datetime.now()}")
    time_start = time()
    predict_text_class_DaaS_tqdm(dataf=dataframe,
                                 model=model,
                                 label_relato=label_relato,
                                 label_name=label_name,
                                 label_score=label_score,
                                 words_qty_label=words_qty_label,
                                 threshold_words_qty=THRESHOLD_WORDS_QTY,
                                 status=status,
                                 )
    time_end = time()
    delta_time = time_end-time_start
    dataframe[fecha_label] = datetime.now()
    readable_time = seconds_to_readable_time(delta_time)
    
    print(f"Predicción {model_name} concluída {datetime.now()}\nDuración: {readable_time}")


def read_sql_comision_estadistica(database_table):
    database, table = database_table.split('.')
    print(f"Leyendo datos desde {database}.{table}")
    query = text(f"""SELECT 
                    robos.NDD,
                    robos.Tipo_Delito_PJ as 'Tipo_Delito_PJ_comision',
                    robos.delitos_seguimiento as 'delitos_seguimiento_comision',
                    robos.delitos_validados as 'delitos_validados_comision'
                    FROM {database}.{table} robos
                    WHERE robos.Tipo_Delito_PJ = 'ROBO';
                    """)
    return pd.read_sql(query, conectar_sql())
    
    
def read_daas_robosML(sample, database_in, table_in):
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


def fix_estadoml(estadoml, estado_seguimiento, estado_validados):
    """
    Fixes ESTADO_ML output. if in a row_i, estado_seguimiento
    and estado_validados both have value of 1, then we return 1
    if only estado seguimiento or estado_validado was performed, 
    3 and 2 are returned, respectively. In any other case, the original
    value of ESTADO_ML is returned
    """
    if estado_seguimiento==1 and estado_validados==1:
        return 1
    elif estado_validados ==1:
        return 2
    elif estado_seguimiento==1:
        return 3
    else:
        return estadoml


def main(predict_delitos_validados, 
         predict_delitos_seguimiento, 
         csv,
         sql, 
         sample, 
         bbdd_in, 
         update,
         save_path,
         load_data_comision,
         bbdd_comision):
    DATABASE_IN, TABLE_IN = bbdd_in.split('.')
    print(f"Prediccion de Etiquetas Delitos Seguimiento y Delitos Validados Robos en {DATABASE_IN}.{TABLE_IN}")

    xtest_df, LABEL_RELATO = read_daas_robosML(sample=sample, database_in=DATABASE_IN, table_in=TABLE_IN)
    # ahora se debe realizar las predicciones de acuerdo a las caracteristicas
    # limitantes: que sea ndd de robos y con un relato de 50 palabras
    time_report_dict = {}
    xtest_df['ESTADO_ML_SEGUIMIENTO'] = xtest_df['ESTADO_ML']
    xtest_df['ESTADO_ML_VALIDADOS'] = xtest_df['ESTADO_ML']

    if predict_delitos_validados:
        predict_robbery_class_daas(dataframe=xtest_df,
                              path_model=PATH_MODEL_VALIDADOS,
                              label_relato=LABEL_RELATO,
                              label_name=DELITOS_VALIDADOS_COLUMNS_NAMES_DICT.get('label_name'),
                              label_score=DELITOS_VALIDADOS_COLUMNS_NAMES_DICT.get('label_score'),
                              words_qty_label='d_CANTIDAD_PALABRAS',
                              status='ESTADO_ML_VALIDADOS',
                              fecha_label=DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_fecha'],
                              model_name='delitos_validados',
                              )
    print_robbery_kinds_qty(xtest_df, predicted_label=DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_name'])
    # time_report_dict['modelo_validados'] = time_end-time_start

    # predicción por defecto de delitos seguimiento
    
    if predict_delitos_seguimiento:
        predict_robbery_class_daas(dataframe=xtest_df,
                              path_model=PATH_MODEL_SEGUIMIENTOS,
                              label_relato=LABEL_RELATO,
                              label_name=DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT.get('label_name'),
                              label_score=DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT.get('label_score'),
                              words_qty_label='d_CANTIDAD_PALABRAS',
                              status='ESTADO_ML_SEGUIMIENTO',
                              fecha_label=DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_fecha'],
                              model_name='delitos_seguimiento',
                              )
        print_robbery_kinds_qty(xtest_df, predicted_label=DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_name'])
    # adjusting ESTADO_ML

    xtest_df['ESTADO_ML'] = xtest_df.apply(lambda x: fix_estadoml(estadoml=x['ESTADO_ML'], 
                                                                  estado_seguimiento=x['ESTADO_ML_SEGUIMIENTO'], 
                                                                  estado_validados=x['ESTADO_ML_VALIDADOS']), 
                                                                  axis=1)
    # loading data from comision
    if load_data_comision:
        comission_df = read_sql_comision_estadistica(database_table=bbdd_comision)
        print(f"Comision de Estadística tiene {comission_df.shape[0]} registros de {comission_df.columns}")
        # merging dataframes
        print(f"Caracteristicas del dataframe obtenido:\n{xtest_df.info()}")
        output_df = pd.merge(xtest_df, comission_df, on="NDD", how='left', suffixes=['','_comision'])
        print(output_df.shape, xtest_df.shape, comission_df.shape)
        print(f"Caracteristicas del dataframe obtenido:\n{output_df.info()}")
    else:
        output_df = xtest_df
    # salida del programa        
    # print(f"Predicciones Terminadas: {time_report_dict['modelo_validados']+time_report_dict['modelo_seguimientos']}")
    # print(time_report_dict.items())
    if csv:
        output_file_name = 'DaaS_RobosML_predicted-'+datetime.now().strftime('%Y-%m-%d')+'.csv'
        to_save = os.path.join(save_path, output_file_name )
        print(f"Salvando resultados a csv {to_save}")
        output_df.to_csv(to_save, index=False)
    if sql:
        table_out = TABLE_IN+'_predicted_tmp'+'_'+datetime.now().strftime('%Y_%m_%d')
        print(f"Guardando en base de datos DaaS en tabla sql {table_out}")
        save_df_in_sql(name_table=table_out, dataf=output_df, database='DaaS')
        # implementar la actualizacion de la tabla
        if update:
            query = f"""UPDATE {DATABASE_IN}.{TABLE_IN} del1
                       INNER JOIN {DATABASE_IN}.{table_out} del2
                       ON del1.NDD = del2.NDD
                       SET del1.{DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_name']} = del2.{DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_name']},
                       del1.{DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_score']} = del2.{DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_score']},
                       del1.{DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_fecha']} = del2.{DELITOS_SEGUIMIENTOS_COLUMNS_NAMES_DICT['label_fecha']},
                       del1.{DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_name']} = del2.{DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_name']},
                       del1.{DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_score']} = del2.{DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_score']},
                       del1.{DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_fecha']} = del2.{DELITOS_VALIDADOS_COLUMNS_NAMES_DICT['label_fecha']},
                       del1.ESTADO_ML = del2.ESTADO_ML;"""
                       

            engine_maria_db = create_engine(f"mysql+pymysql://falconiel:BebuSuKO@192.168.152.197:3306/{DATABASE_IN}"+"?charset=utf8mb4")
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
    
    parser.add_argument('--bbddin', default='DaaS.robosML', type=str, help='Nombre de la base y tabla sql de la que se debe leer los datos de robos a actualizar. El formato es BASE.TABLA')
    parser.add_argument('--sample', action="store_true", help="Si se declara, consulta los 1.000 registros de la base para pruebas")
    parser.add_argument('--validados', action="store_true", help="Si se declara, realiza la predicción de etiquetas de delitos validados")
    parser.add_argument('--seguimiento', action="store_true", help="Si se declara, realiza la predicción de etiquetas de delitos seguimiento")
    parser.add_argument('--load_data_comision', action="store_true", help="Si se declara, realiza la carga de los datos de delitos seguimiento de la comision para 2014 a 2022. Delitos Validados se sugiere tomar del modelo")
    parser.add_argument('--bbdd_comision', default='reportes.robos_2014_08012023', help="Especifica la base y \
    la tabla donde reposan los datos de la comisión para los años 2014 - 2022. El formato es BASE.TABLA")
    parser.add_argument('--save2sql', action="store_true", help="Si se declara, se guarda los resultados obtenidos en tabla de SQL. Por defecto guarda en [DATABASE].[TABLE]")
    parser.add_argument('--save2csv', action="store_true", help="Si se declara, se guarda los resultados obtenidos en archivo CSV. La ubicacion se declara en save_path_files")
    parser.add_argument('--save_path', default='data/processed/', help="Especifica la ubicacion en que se guardaran los resultados obtenidos")
    parser.add_argument('--update', action="store_true", help="Si se declara, realiza la actualizacion de la base de datos original leida")
    args = parser.parse_args()
    main(predict_delitos_validados=args.validados, 
         predict_delitos_seguimiento=args.seguimiento,
         csv=args.save2csv,
         save_path=args.save_path, 
         sample=args.sample, 
         bbdd_in=args.bbddin, 
         sql=args.save2sql, 
         update=args.update,
         load_data_comision=args.load_data_comision,
         bbdd_comision=args.bbdd_comision)



# query = f"""UPDATE {DATABASE_IN}.{TABLE_IN} del1
#                        INNER JOIN {DATABASE_IN}.{table_out} del2
#                        ON del1.NDD = del2.NDD
#                        SET del1.delitos_seguimiento_predicted = del2.delitos_seguimiento_predicted,
#                        del1.delitos_seguimiento_predicted_score = del2.delitos_seguimiento_predicted_score,
#                        del1.ESTADO_ML = del2.ESTADO_ML,
#                        del1.FechaActualizacionDelitosSeguimiento = del2.FechaActualizacionDelitosSeguimiento;"""